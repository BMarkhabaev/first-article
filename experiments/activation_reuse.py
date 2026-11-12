"""Идея 6: выражается ли MLP слоя b через активации MLP слоя a (соседнего)?

Для пары (a, b): признаки — z_a = act(gate_a x_a)·up_a x_a (I) и x_b (вход MLP b, d); цель — y_b = MLP_b(x_b).
Гребневая регрессия (робастные строки: без позиции 0 и массивных ||z|| > tau·медиана) в трёх вариантах:
  lin   — y ≈ x_b B           (линейная замена, d·d умножений на токен, база);
  reuse — y ≈ z_a A           (переиспользование, I·d = 1/3 исходного MLP);
  joint — y ≈ z_a A + x_b B   ((I+d)·d).
Отчёт: R² на отложенных окнах и KL/PPL при подстановке в модель (ExactPos0): по каждой паре отдельно и для
непересекающихся поздних пар вместе.

  python activation_reuse.py --model 0.6b
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from detlayers.common import calibration_windows, load_model, pick_device, save_json, seed_all, wikitext_windows  # noqa: E402
from detlayers.methods import Replacement  # noqa: E402
from detlayers.metrics import compare_lm  # noqa: E402
from detlayers.mlp import ExactPos0, _z, robust_rows_z  # noqa: E402


class CacheMLP(nn.Module):
    """MLP слоя a: считает как исходный, но запоминает z для слоя b."""

    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp
        self.z = None

    def forward(self, x):
        z = _z(self.mlp, x)
        self.z = z
        return self.mlp.down_proj(z)


class ReuseMLP(nn.Module):
    """MLP слоя b: y = z_a A + x B + c (любая из A, B может отсутствовать)."""

    def __init__(self, cache: CacheMLP | None, A, B, c):
        super().__init__()
        self.cache = cache
        self.A = None if A is None else nn.Parameter(A, requires_grad=False)
        self.B = None if B is None else nn.Parameter(B, requires_grad=False)
        self.c = nn.Parameter(c, requires_grad=False)

    def forward(self, x):
        y = self.c.expand(*x.shape[:-1], -1).clone()
        if self.B is not None:
            y = y + x @ self.B
        if self.A is not None:
            y = y + self.cache.z[:, :x.shape[1]] @ self.A
        return y


def mlp_inputs(model, ids, layers_wanted):
    cache, hooks = {}, []
    for l in layers_wanted:
        hooks.append(model.model.layers[l].mlp.register_forward_pre_hook(
            lambda m, inp, l=l: cache.__setitem__(l, inp[0].detach())))
    try:
        model(ids)
    finally:
        for h in hooks:
            h.remove()
    return cache


def parse_pairs(spec, L):
    if spec != "auto":
        return [tuple(int(v) for v in p.split("-")) for p in spec.split(",")]
    late = [(L - 2, L - 1), (L - 3, L - 2), (L - 4, L - 3), (L - 5, L - 4), (L - 6, L - 5)]
    early = [(1, 2), (2, 3), (L // 2 - 1, L // 2)]
    return list(dict.fromkeys(late + early))  # без дубликатов, порядок сохранён


def _solve(G, R, lam_rel):
    n = G.shape[0]
    lam = lam_rel * G.diagonal().mean()
    return torch.linalg.solve(G + lam * torch.eye(n, dtype=G.dtype), R)


@torch.no_grad()
def fit_pair(model, calib, a, b, device, args):
    """Гребневые решения для трёх вариантов из одной совместной матрицы Грама [1, x_b, z_a]."""
    layers = model.model.layers
    d, I = model.config.hidden_size, model.config.intermediate_size
    n_fit = calib.shape[0] - args.holdout
    F = 1 + d + I
    G = torch.zeros(F, F, dtype=torch.float64)
    R = torch.zeros(F, d, dtype=torch.float64)
    held = []  # признаки и цели отложенных окон
    for s in range(0, calib.shape[0], args.bs):
        ids = calib[s:s + args.bs].to(device)
        xs = mlp_inputs(model, ids, [a, b])
        za = _z(layers[a].mlp, xs[a]).float()
        xb = xs[b].float()
        yb = layers[b].mlp(xs[b]).float()
        keep = robust_rows_z(za, args.tau) & robust_rows_z(_z(layers[b].mlp, xs[b]).float(), args.tau)
        f = torch.cat([torch.ones(*xb.shape[:-1], 1, device=device), xb, za], -1)[keep].cpu().double()
        y = yb[keep].cpu().double()
        if s < n_fit:
            G += f.T @ f
            R += f.T @ y
        else:
            held.append((f, y))
    sol = {}
    idx = {"lin": list(range(0, 1 + d)), "reuse": [0] + list(range(1 + d, F)), "joint": list(range(F))}
    fh = torch.cat([f for f, _ in held]); yh = torch.cat([y for _, y in held])
    var = ((yh - yh.mean(0)) ** 2).sum()
    for name, ii in idx.items():
        W = _solve(G[ii][:, ii], R[ii], args.lam)
        r2 = 1 - ((yh[:, :] - fh[:, ii] @ W) ** 2).sum() / var
        sol[name] = (W, float(r2))
    return sol, d, I


def make_modules(model, a, b, W, variant, d, I, device):
    layers = model.model.layers
    W = W.float().to(device)
    c = W[0]
    if variant == "lin":
        A, B, cache = None, W[1:1 + d], None
    elif variant == "reuse":
        A, B, cache = W[1:1 + I], None, CacheMLP(layers[a].mlp)
    else:
        A, B, cache = W[1 + d:1 + d + I], W[1:1 + d], CacheMLP(layers[a].mlp)
    return cache, ExactPos0(layers[b].mlp, ReuseMLP(cache, A, B, c))


@torch.no_grad()
def run(model, tok, device, args):
    L = len(model.model.layers)
    pairs = parse_pairs(args.pairs, L)
    calib = calibration_windows(tok, args.calib_windows * 512, 512, seed=0)
    wins = wikitext_windows(tok, "wikitext-2-raw-v1", "test", 1024, max_windows=args.eval_windows)
    t0 = time.time()
    out = {"model": args.model, "pairs": {}, "joint_sets": {}}
    fitted = {}
    for a, b in pairs:
        sol, d, I = fit_pair(model, calib, a, b, device, args)
        fitted[(a, b)] = sol
        row = {}
        for variant in ("lin", "reuse", "joint"):
            W, r2 = sol[variant]
            cache, mod = make_modules(model, a, b, W, variant, d, I, device)
            rep = Replacement(model)
            if cache is not None:
                rep.add(a, "mlp", cache)
            rep.add(b, "mlp", mod)
            r = compare_lm(model, rep, wins, device, bs=1)
            rep.disable()
            row[variant] = {"r2_holdout": r2, "kl": r["kl"], "dppl_pct": 100 * (r["ppl"] / r["ppl_teacher"] - 1),
                            "top1": r["top1_agree"]}
            print(f"пара ({a},{b}) {variant:5s}: R² {r2:.3f}  KL {r['kl']:.5f}  dPPL {row[variant]['dppl_pct']:.2f}%",
                  flush=True)
        out["pairs"][f"{a}-{b}"] = row
    # непересекающиеся поздние пары вместе (reuse и joint)
    d, I = model.config.hidden_size, model.config.intermediate_size
    disjoint = [(a, b) for a, b in pairs if b >= L - 6 and (L - b) % 2 == 1][:3]  # (L-2,L-1), (L-4,L-3), (L-6,L-5)
    for variant in ("reuse", "joint"):
        rep = Replacement(model)
        for a, b in disjoint:
            W, _ = fitted[(a, b)][variant]
            cache, mod = make_modules(model, a, b, W, variant, d, I, device)
            rep.add(a, "mlp", cache)
            rep.add(b, "mlp", mod)
        r = compare_lm(model, rep, wins, device, bs=1)
        rep.disable()
        out["joint_sets"][variant] = {"pairs": disjoint, "kl": r["kl"],
                                      "dppl_pct": 100 * (r["ppl"] / r["ppl_teacher"] - 1), "top1": r["top1_agree"]}
        print(f"вместе {disjoint} {variant}: KL {r['kl']:.5f}  dPPL {out['joint_sets'][variant]['dppl_pct']:.2f}%",
              flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    out["args"] = vars(args)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="0.6b")
    ap.add_argument("--pairs", default="auto", help="auto или список a-b через запятую")
    ap.add_argument("--calib-windows", type=int, default=64)
    ap.add_argument("--holdout", type=int, default=16, help="последние окна калибровки — для R²")
    ap.add_argument("--eval-windows", type=int, default=8)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--tau", type=float, default=10.0)
    ap.add_argument("--lam", type=float, default=1e-3, help="гребень относительно среднего диагонали Грама")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seed_all(0)
    device = pick_device(None)
    model, tok = load_model(args.model, device, "fp32")
    res = run(model, tok, device, args)
    out = Path(args.out) if args.out else ROOT / "results" / args.model / "criterion" / "activation_reuse.json"
    save_json(res, out)
    print("сохранено", out)


if __name__ == "__main__":
    main()
