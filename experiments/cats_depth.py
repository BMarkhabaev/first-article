#!/usr/bin/env python
"""CATS по глубине: порог |act(gate)| (25% активных) в одном слое ℓ для всех ℓ — KL одиночной замены,
против отбора 25% по полному вкладу (kl_mlp из criterion/profile.json). Результат: results/<model>/criterion/cats_depth.json."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detlayers.common import calibration_windows, load_model, pick_device, save_json, seed_all, wikitext_windows  # noqa: E402
from detlayers.methods import Replacement  # noqa: E402
from detlayers.mlp import CatsMLP, ExactPos0, iterate_mlp_inputs  # noqa: E402

ROOT = Path(__file__).resolve().parent


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="0.6b")
    ap.add_argument("--ratio", type=float, default=0.25)
    ap.add_argument("--windows", type=int, default=8)
    args = ap.parse_args()
    seed_all(0)
    dev = pick_device(None)
    model, tok = load_model(args.model, dev, "fp32")
    layers = model.model.layers
    L = len(layers)
    calib = calibration_windows(tok, 131_072, 512, seed=0)[:32]
    g = torch.Generator().manual_seed(0)
    pools = [[] for _ in range(L)]
    for i, x, _ in iterate_mlp_inputs(model, calib, L, 8, dev):
        a = layers[i].mlp.act_fn(layers[i].mlp.gate_proj(x[:, 1:])).float().abs().reshape(-1)
        idx = torch.randperm(a.numel(), generator=g)[:500_000]
        pools[i].append(a[idx.to(a.device)].cpu())
    thr = [torch.cat(p).kthvalue(max(1, int((1 - args.ratio) * torch.cat(p).numel()))).values.item() for p in pools]
    wins = wikitext_windows(tok, "wikitext-2-raw-v1", "test", 1024, max_windows=args.windows)
    prof = json.loads((ROOT / "results" / args.model / "criterion" / "profile.json").read_text())["profile"]
    res = {"model": args.model, "ratio": args.ratio, "thresholds": thr, "kl_cats": [], "kl_topk": prof["kl_mlp"]}
    for l in range(L):
        rep = Replacement(model)
        rep.add(l, "mlp", ExactPos0(layers[l].mlp, CatsMLP(layers[l].mlp, thr[l], args.ratio)))
        kl = 0.0
        n = 0
        for s in range(wins.shape[0]):
            ids = wins[s:s + 1].to(dev)
            rep.disable()
            lpt = F.log_softmax(model(ids).logits[:, :-1].float(), -1)
            rep.enable()
            lps = F.log_softmax(model(ids).logits[:, :-1].float(), -1)
            rep.disable()
            kl += (lpt.exp() * (lpt - lps)).sum().item()
            n += lpt.shape[1]
        res["kl_cats"].append(kl / n)
        print(f"слой {l:2d}: CATS {kl / n:.4f}  отбор по вкладу {prof['kl_mlp'][l]:.4f}  отношение {kl / n / max(prof['kl_mlp'][l], 1e-9):.1f}", flush=True)
    save_json(res, ROOT / "results" / args.model / "criterion" / "cats_depth.json")


if __name__ == "__main__":
    main()
