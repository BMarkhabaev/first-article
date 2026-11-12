"""Идея 4: нужна ли поздним MLP контекстная разреженность?

Для каждого слоя считаем по калибровке вклад нейронов c_j = |z_j|·||W_down[:, j]|| на токен, множество top-25 %
на токене и статическое множество S_l (top-25 % по частоте попадания в токенные множества). Метрики по слоям:
  jaccard   — среднее |top_t ∩ S| / |top_t ∪ S|;
  cov_static — доля энергии вклада токена, покрытая S;  cov_oracle — доля, покрытая его собственным top-25 %.
Затем замена: статическое прореживание (SelectedMLP, реальная экономия 75 % умножений) против оракула top-k
(TopKMLP, экономии нет) на наборах слоёв: последние 1/2/4/6 и первые 1/2/4 (ExactPos0 везде).

  python static_late_mlp.py --model 0.6b
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from detlayers.common import calibration_windows, load_model, pick_device, save_json, seed_all, wikitext_windows  # noqa: E402
from detlayers.methods import Replacement  # noqa: E402
from detlayers.metrics import compare_lm  # noqa: E402
from detlayers.mlp import ExactPos0, SelectedMLP, TopKMLP, _z, robust_rows_z  # noqa: E402


def mlp_inputs(model, ids, layers_wanted):
    """Входы MLP (после post_attention_layernorm) выбранных слоёв за один проход учителя."""
    cache = {}
    hooks = []
    for l in layers_wanted:
        mlp = model.model.layers[l].mlp
        hooks.append(mlp.register_forward_pre_hook(lambda m, inp, l=l: cache.__setitem__(l, inp[0].detach())))
    try:
        model(ids)
    finally:
        for h in hooks:
            h.remove()
    return cache


def layer_sets(L, spec):
    out = {}
    for s in spec.split(","):
        s = s.strip()
        n = int(s[4:] if s.startswith("last") else s[5:])
        out[s] = list(range(L - n, L)) if s.startswith("last") else list(range(n))
    return out


@torch.no_grad()
def run(model, tok, device, args):
    layers = model.model.layers
    L, I = len(layers), model.config.intermediate_size
    kap = max(1, int(args.ratio * I))
    calib = calibration_windows(tok, args.calib_windows * 512, 512, seed=0)
    wins = wikitext_windows(tok, "wikitext-2-raw-v1", "test", 1024, max_windows=args.eval_windows)
    wdn = [layers[l].mlp.down_proj.weight.norm(dim=0).float() for l in range(L)]  # (I,)
    freq = torch.zeros(L, I, dtype=torch.float64)
    sumc = torch.zeros(L, I, dtype=torch.float64)
    ntok = 0
    t0 = time.time()
    # --- проход 1: частоты попадания в top-25 % на калибровке
    for s in range(0, calib.shape[0], args.bs):
        ids = calib[s:s + args.bs].to(device)
        xs = mlp_inputs(model, ids, range(L))
        for l in range(L):
            z = _z(layers[l].mlp, xs[l]).float()
            keep = robust_rows_z(z, args.tau)
            c = (z.abs() * wdn[l])[keep]  # (n, I)
            top = c.topk(kap, dim=-1).indices
            m = torch.zeros_like(c, dtype=torch.bool).scatter_(1, top, True)
            freq[l] += m.sum(0).cpu().double()
            sumc[l] += c.sum(0).cpu().double()
            if l == 0:
                ntok += c.shape[0]
    S = {l: freq[l].topk(kap).indices for l in range(L)}
    S_mean = {l: sumc[l].topk(kap).indices for l in range(L)}
    print(f"калибровка: {ntok} токенов, kap={kap}, {time.time() - t0:.0f}s", flush=True)
    # --- проход 2: покрытие и Жаккар на оценочных окнах (учитель)
    per = {k: torch.zeros(L, dtype=torch.float64) for k in ("jaccard", "cov_static", "cov_mean", "cov_oracle")}
    nb = 0
    for s in range(wins.shape[0]):
        ids = wins[s:s + 1].to(device)
        xs = mlp_inputs(model, ids, range(L))
        for l in range(L):
            z = _z(layers[l].mlp, xs[l]).float()
            keep = robust_rows_z(z, args.tau)
            c = (z.abs() * wdn[l])[keep]
            tot = c.sum(-1).clamp_min(1e-12)
            top = c.topk(kap, dim=-1)
            Sl, Sm = S[l].to(device), S_mean[l].to(device)
            inter = torch.zeros_like(c, dtype=torch.bool).scatter_(1, top.indices, True)[:, Sl].sum(-1).float()
            per["jaccard"][l] += (inter / (2 * kap - inter)).mean().item()
            per["cov_static"][l] += (c[:, Sl].sum(-1) / tot).mean().item()
            per["cov_mean"][l] += (c[:, Sm].sum(-1) / tot).mean().item()
            per["cov_oracle"][l] += (top.values.sum(-1) / tot).mean().item()
        nb += 1
    per = {k: (v / nb).tolist() for k, v in per.items()}
    for l in range(L):
        print(f"слой {l:2d}: Жаккар {per['jaccard'][l]:.3f}  покрытие статич. {per['cov_static'][l]:.3f} "
              f"(по среднему {per['cov_mean'][l]:.3f})  оракул {per['cov_oracle'][l]:.3f}", flush=True)
    # --- замены на наборах слоёв
    res_sets = {}
    for name, ls in layer_sets(L, args.sets).items():
        row = {"layers": ls}
        for kind in ("static", "oracle"):
            rep = Replacement(model)
            for l in ls:
                mlp = layers[l].mlp
                approx = SelectedMLP(mlp, S[l].to(device)) if kind == "static" else TopKMLP(mlp, kap)
                rep.add(l, "mlp", ExactPos0(mlp, approx))
            r = compare_lm(model, rep, wins, device, bs=1)
            rep.disable()
            row[kind] = {"kl": r["kl"], "dppl_pct": 100 * (r["ppl"] / r["ppl_teacher"] - 1), "top1": r["top1_agree"]}
            print(f"{name:7s} {kind:6s} KL {r['kl']:.4f}  dPPL {row[kind]['dppl_pct']:.2f}%", flush=True)
        res_sets[name] = row
    return {"model": args.model, "ratio": args.ratio, "kap": kap, "tau": args.tau, "calib_tokens": ntok,
            "eval_windows": int(wins.shape[0]), "per_layer": per, "static_sets": {l: S[l].tolist() for l in range(L)},
            "sets": res_sets, "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="0.6b")
    ap.add_argument("--ratio", type=float, default=0.25)
    ap.add_argument("--tau", type=float, default=10.0)
    ap.add_argument("--calib-windows", type=int, default=64)
    ap.add_argument("--eval-windows", type=int, default=8)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--sets", default="last1,last2,last4,last6,first1,first2,first4")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seed_all(0)
    device = pick_device(None)
    model, tok = load_model(args.model, device, "fp32")
    res = run(model, tok, device, args)
    out = Path(args.out) if args.out else ROOT / "results" / args.model / "criterion" / "static_late.json"
    save_json(res, out)
    print("сохранено", out)


if __name__ == "__main__":
    main()
