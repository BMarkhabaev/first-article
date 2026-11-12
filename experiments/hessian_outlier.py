#!/usr/bin/env python
"""Пункт 1: доминирует ли один токен в статистиках Xᵀ X входов down_proj (их используют SVD-LLM, GPTQ, SparseGPT)?
Для слоёв 0..5 копим H = Σ z zᵀ по калибровке в двух вариантах: все позиции (как в методах) и позиции ≥1.
Считаем долю следа от позиции 0, от одного максимального токена, долю старшего собственного значения и
косинус между старшим собственным вектором и z максимального токена. Результат: results/<model>/hessian/outlier.json."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detlayers.common import calibration_windows, load_model, pick_device, save_json, seed_all  # noqa: E402
from detlayers.mlp import _z, iterate_mlp_inputs  # noqa: E402

ROOT = Path(__file__).resolve().parent


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="0.6b")
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--calib-tokens", type=int, default=131_072)
    args = ap.parse_args()
    seed_all(0)
    dev = pick_device(None)
    model, tok = load_model(args.model, dev, "fp32")
    I = model.config.intermediate_size
    calib = calibration_windows(tok, args.calib_tokens, 512, seed=0)
    k = args.layers
    H_all = [torch.zeros(I, I, dtype=torch.float64) for _ in range(k)]
    H_pos = [torch.zeros(I, I, dtype=torch.float64) for _ in range(k)]
    tr0 = [0.0] * k
    zmax = [None] * k
    nmax = [0.0] * k
    for i, x, ids in iterate_mlp_inputs(model, calib, k, 8, dev):
        z = _z(model.model.layers[i].mlp, x).float()  # (B,T,I)
        zf = z.reshape(-1, I)
        H_all[i] += (zf.T @ zf).cpu().double()
        z1 = z[:, 1:].reshape(-1, I)
        H_pos[i] += (z1.T @ z1).cpu().double()
        tr0[i] += (z[:, 0].float() ** 2).sum().item()
        nn_ = z1.norm(dim=-1)
        j = int(nn_.argmax())
        if nn_[j].item() > nmax[i]:
            nmax[i] = nn_[j].item()
            zmax[i] = z1[j].detach().cpu().double()
    res = {"model": args.model, "calib_tokens": int(calib.numel()), "layers": {}}
    for i in range(k):
        out = {}
        for tag, H in (("all_positions", H_all[i]), ("positions_ge1", H_pos[i])):
            tr = H.diagonal().sum().item()
            evals, evecs = torch.linalg.eigh(H)
            v1 = evecs[:, -1]
            out[tag] = {"trace": tr, "top_eig_share": evals[-1].item() / tr,
                        "pos0_trace_share": tr0[i] / tr if tag == "all_positions" else 0.0,
                        "max_token_trace_share": (nmax[i] ** 2) / tr,
                        "cos_topvec_maxtoken": abs(F.cosine_similarity(v1, zmax[i], dim=0).item()),
                        "max_token_norm": nmax[i]}
        med = None
        res["layers"][i] = out
        a, b = out["all_positions"], out["positions_ge1"]
        print(f"слой {i}: все позиции — доля следа от позиции 0 {100*a['pos0_trace_share']:.1f}%, от макс. токена "
              f"{100*a['max_token_trace_share']:.1f}%, λ1/tr {100*a['top_eig_share']:.1f}%, cos(v1, z_max) {a['cos_topvec_maxtoken']:.3f} | "
              f"позиции ≥1 — макс. токен {100*b['max_token_trace_share']:.1f}% следа, λ1/tr {100*b['top_eig_share']:.1f}%, "
              f"cos {b['cos_topvec_maxtoken']:.3f}, ||z_max|| {b['max_token_norm']:.0f}", flush=True)
    save_json(res, ROOT / "results" / args.model / "hessian" / "outlier.json")


if __name__ == "__main__":
    main()
