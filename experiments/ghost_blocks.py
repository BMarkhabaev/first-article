#!/usr/bin/env python
"""«Призрак» блока поздних слоёв: удалить n подряд идущих слоёв и заменить их линейной картой остаточного потока
(ReplaceMe / Ghosted Layers в закрытой форме, с робастными строками и контролем ранга).

Выбор блока: (а) по нашей карте — минимальная сумма KL одиночного удаления слоя; (б) по ShortGPT — минимальная
сумма влияния блока BI_ℓ = 1 − cos(h_in, h_out). Карты: identity (чистое удаление), full (h_b ≈ h_a W + b),
rank r на приращении (h_b − h_a ≈ h_a A B + b). Качество — KL/PPL на 32 окнах WikiText-2.
Результат: results/<model>/ghost/blocks.json."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detlayers.common import calibration_windows, embed, load_model, pick_device, run_layers, save_json, seed_all, wikitext_windows  # noqa: E402
from detlayers.methods import IdentityBlock, Replacement  # noqa: E402
from detlayers.mlp import reduced_rank  # noqa: E402

ROOT = Path(__file__).resolve().parent


class GhostBlock(nn.Module):
    """h ← h + (h A B + b) для позиций ≥1 (ранг r) или h ← h W + b (full); позиция 0 проходит без изменений."""

    def __init__(self, W=None, b=None, A=None, B=None):
        super().__init__()
        self.register_buffer("W", W if W is not None else torch.zeros(0))
        self.register_buffer("b", b if b is not None else torch.zeros(0))
        self.register_buffer("A", A if A is not None else torch.zeros(0))
        self.register_buffer("B", B if B is not None else torch.zeros(0))

    def forward(self, h, *a, **kw):
        x = h[:, 1:].float()
        if self.A.numel():
            y = x + (x @ self.A) @ self.B + self.b
        elif self.W.numel():
            y = x @ self.W + self.b
        else:
            y = x
        return torch.cat([h[:, :1], y.to(h.dtype)], 1)


@torch.no_grad()
def states(model, ids):
    h, pos, pe = embed(model, ids)
    hs = [h]
    for i in range(len(model.model.layers)):
        h = run_layers(model, model.model.layers, h, pos, pe, i, i + 1)
        hs.append(h)
    return hs


@torch.no_grad()
def kl_eval(model, reps, wins, device):
    nll_t = nll_s = kl = agree = 0.0
    n = 0
    for s in range(wins.shape[0]):
        ids = wins[s:s + 1].to(device)
        for r in reps:
            r.disable()
        lt = model(ids).logits[:, :-1].float()
        for r in reps:
            r.enable()
        ls = model(ids).logits[:, :-1].float()
        for r in reps:
            r.disable()
        tgt = ids[:, 1:]
        lpt, lps = F.log_softmax(lt, -1), F.log_softmax(ls, -1)
        nll_t += -lpt.gather(-1, tgt[..., None]).sum().item()
        nll_s += -lps.gather(-1, tgt[..., None]).sum().item()
        kl += (lpt.exp() * (lpt - lps)).sum().item()
        agree += (lt.argmax(-1) == ls.argmax(-1)).sum().item()
        n += tgt.numel()
    import math
    return {"ppl_teacher": math.exp(nll_t / n), "ppl": math.exp(nll_s / n), "dppl_pct": 100 * (math.exp(nll_s / n) / math.exp(nll_t / n) - 1),
            "kl": kl / n, "top1_agree": agree / n}


def block_reps(model, a, b, mod):
    rep = Replacement(model)
    rep.add(a, "block", mod)
    for i in range(a + 1, b):
        rep.add(i, "block", IdentityBlock())
    return [rep]


@torch.no_grad()
def fit_maps(model, calib, device, a, b, bs, tau, ranks, lam=1e-3):
    d = model.config.hidden_size
    G = torch.zeros(d + 1, d + 1, dtype=torch.float64)
    By = torch.zeros(d + 1, d, dtype=torch.float64)
    Bd = torch.zeros(d + 1, d, dtype=torch.float64)
    n = 0
    for s in range(0, calib.shape[0], bs):
        ids = calib[s:s + bs].to(device)
        h, pos, pe = embed(model, ids)
        x = run_layers(model, model.model.layers, h, pos, pe, 0, a)
        y = run_layers(model, model.model.layers, x, pos, pe, a, b)
        xf, yf = x[:, 1:].float().reshape(-1, d), y[:, 1:].float().reshape(-1, d)
        nx, ny = xf.norm(dim=-1), yf.norm(dim=-1)
        keep = (nx <= tau * nx.median()) & (ny <= tau * ny.median())
        xf, yf = xf[keep], yf[keep]
        phi = torch.cat([xf, torch.ones_like(xf[:, :1])], 1)
        G += (phi.T @ phi).cpu().double()
        By += (phi.T @ yf).cpu().double()
        Bd += (phi.T @ (yf - xf)).cpu().double()
        n += xf.shape[0]
    reg = lam * G.diagonal().mean()
    Wy = torch.linalg.solve(G + reg * torch.eye(d + 1, dtype=torch.float64), By)
    Wd = torch.linalg.solve(G + reg * torch.eye(d + 1, dtype=torch.float64), Bd)
    maps = {"identity": GhostBlock(), "full": GhostBlock(W=Wy[:-1].float().to(device), b=Wy[-1].float().to(device))}
    for r in ranks:
        A_, B_ = reduced_rank(Wd, G[:-1, :-1], r)
        maps[f"rank{r}"] = GhostBlock(b=Wd[-1].float().to(device), A=A_.float().to(device), B=B_.float().to(device))
    return maps, n


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="0.6b")
    ap.add_argument("--sizes", default="2,4")
    ap.add_argument("--ranks", default="64,256")
    ap.add_argument("--diag-windows", type=int, default=8)
    ap.add_argument("--eval-windows", type=int, default=32)
    ap.add_argument("--calib-tokens", type=int, default=131_072)
    ap.add_argument("--tau", type=float, default=10.0)
    args = ap.parse_args()
    seed_all(0)
    device = pick_device(None)
    model, tok = load_model(args.model, device, "fp32")
    L = len(model.model.layers)
    wins = wikitext_windows(tok, "wikitext-2-raw-v1", "test", 1024, max_windows=args.eval_windows)
    dwins = wins[: args.diag_windows]
    out = ROOT / "results" / args.model / "ghost" / "blocks.json"
    t0 = time.time()
    # --- статистики выбора: BI (ShortGPT) и KL удаления одного слоя
    bi = torch.zeros(L)
    for s in range(dwins.shape[0]):
        hs = states(model, dwins[s:s + 1].to(device))
        for l in range(L):
            x, y = hs[l][0, 1:].float(), hs[l + 1][0, 1:].float()
            bi[l] += (1 - F.cosine_similarity(x, y, dim=-1)).mean().item()
    bi /= dwins.shape[0]
    drop_kl = []
    for l in range(L):
        drop_kl.append(kl_eval(model, block_reps(model, l, l + 1, IdentityBlock()), dwins, device)["kl"])
    res = {"model": args.model, "bi_shortgpt": [round(float(v), 5) for v in bi], "drop_kl": [round(v, 5) for v in drop_kl],
           "blocks": {}}
    print("BI:", [round(float(v), 3) for v in bi], flush=True)
    print("drop KL:", [round(v, 4) for v in drop_kl], flush=True)
    calib = calibration_windows(tok, args.calib_tokens, 512, seed=0)
    ranks = [int(r) for r in args.ranks.split(",")]
    for n in (int(x) for x in args.sizes.split(",")):
        cands = {}
        for crit, score in (("map_kl", drop_kl), ("shortgpt_bi", bi.tolist())):
            best = min(range(1, L - n + 1), key=lambda a: sum(score[a:a + n]))  # слой 0 не трогаем
            cands[crit] = best
        for crit, a in cands.items():
            b = a + n
            key = f"n{n}_{crit}_layers{a}-{b - 1}"
            maps, ntok = fit_maps(model, calib, device, a, b, 8, args.tau, ranks)
            r = {"layers": [a, b - 1], "criterion": crit, "fit_tokens": ntok, "maps": {}}
            for mname, mod in maps.items():
                r["maps"][mname] = kl_eval(model, block_reps(model, a, b, mod), wins, device)
                print(key, mname, json.dumps({k_: round(v, 4) for k_, v in r["maps"][mname].items()}), flush=True)
            res["blocks"][key] = r
            save_json(res, out)
    res["seconds"] = round(time.time() - t0)
    save_json(res, out)
    print("saved", out)


if __name__ == "__main__":
    main()
