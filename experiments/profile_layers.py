#!/usr/bin/env python
"""Профилирование всех слоёв: насколько каждый слой «прост» (аппроксимируем).

Калибровка делится пополам: на половине A подгоняются позиционные ядра и ridge-карты MLP,
на половине B считаются метрики (отложенная оценка).

  python profile_layers.py --model 0.6b
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from detlayers.attention import RelPosStats, exact_attention_weights, relpos_matrix
from detlayers.common import (RESULTS_DIR, calibration_windows, embed, env_info, load_model, pick_device,
                              save_json, seed_all)
from detlayers.methods import qkv_of


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="0.6b")
    ap.add_argument("--tokens", type=int, default=65_536)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--max-offset", type=int, default=64)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    seed_all(0)
    dev = pick_device(args.device)
    model, tok = load_model(args.model, dev, args.dtype)
    cfg = model.config
    Lnum, H, d = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.hidden_size
    calib = calibration_windows(tok, args.tokens, args.seq_len, seed=1)
    half = calib.shape[0] // 2
    A, B = calib[:half], calib[half:]
    D = args.max_offset

    def layer_states(ids):
        h, pos, pe = embed(model, ids)
        for i, L in enumerate(model.model.layers):
            x = L.input_layernorm(h)
            a_out, _ = L.self_attn(x, position_embeddings=pe, attention_mask=None)
            hm = h + a_out
            xm = L.post_attention_layernorm(hm)
            mo = L.mlp(xm)
            yield i, h, x, pe, xm, mo
            h = hm + mo

    # ---------- проход A: позиционные ядра и ridge для MLP
    rel = [RelPosStats(H, D) for _ in range(Lnum)]
    G = [torch.zeros(d + 1, d + 1, dtype=torch.float64) for _ in range(Lnum)]
    Bm = [torch.zeros(d + 1, d, dtype=torch.float64) for _ in range(Lnum)]
    for s in range(0, A.shape[0], args.bs):
        ids = A[s:s + args.bs].to(dev)
        for i, h, x, pe, xm, mo in layer_states(ids):
            L = model.model.layers[i]
            q, k, _ = qkv_of(L.self_attn, x, pe)
            rel[i].update(exact_attention_weights(q, k, L.self_attn.scaling))
            xf = xm[:, 1:].float().reshape(-1, d)
            phi = torch.cat([xf, torch.ones_like(xf[:, :1])], 1)
            G[i] += (phi.T @ phi).cpu().double()
            Bm[i] += (phi.T @ mo[:, 1:].float().reshape(-1, d)).cpu().double()
    kernels = [r.finalize() for r in rel]
    Wr = []
    for i in range(Lnum):
        reg = 1e-3 * G[i].diagonal().mean()
        Wr.append(torch.linalg.solve(G[i] + reg * torch.eye(d + 1, dtype=torch.float64), Bm[i]).float().to(dev))

    # ---------- проход B: метрики
    keys = ["entropy", "mean_dist", "sink", "prev", "self", "win_mass", "tv_relpos"]
    acc = {k_: torch.zeros(Lnum, H, dtype=torch.float64) for k_ in keys}
    qn = torch.zeros(Lnum, dtype=torch.float64)
    kn = torch.zeros(Lnum, dtype=torch.float64)
    sse = torch.zeros(Lnum, dtype=torch.float64)
    sst = torch.zeros(Lnum, dtype=torch.float64)
    msum = torch.zeros(Lnum, d, dtype=torch.float64)
    pos0 = torch.zeros(Lnum + 1, dtype=torch.float64)
    rest = torch.zeros(Lnum + 1, dtype=torch.float64)
    nb = 0
    ntok = 0
    for s in range(0, B.shape[0], args.bs):
        ids = B[s:s + args.bs].to(dev)
        nb += ids.shape[0]
        T = ids.shape[1]
        t = torch.arange(T, device=dev)
        dist = (t[:, None] - t[None, :]).clamp_min(0).float()
        for i, h, x, pe, xm, mo in layer_states(ids):
            L = model.model.layers[i]
            q, k, _ = qkv_of(L.self_attn, x, pe)
            sc = L.self_attn.scaling ** 0.5
            qn[i] += (q.float() * sc).norm(dim=-1)[:, :, 1:].median().item() * ids.shape[0]
            kn[i] += (k.float() * sc).norm(dim=-1)[:, :, 1:].median().item() * ids.shape[0]
            a = exact_attention_weights(q, k, L.self_attn.scaling)[:, :, 1:]  # запросы t ≥ 1
            tt = t[1:]
            ent = -(a * (a.clamp_min(1e-12)).log()).sum(-1)
            acc["entropy"][i] += ent.mean(dim=(0, 2)).cpu().double() * ids.shape[0]
            acc["mean_dist"][i] += (a * dist[1:]).sum(-1).mean(dim=(0, 2)).cpu().double() * ids.shape[0]
            acc["sink"][i] += a[..., 0].mean(dim=(0, 2)).cpu().double() * ids.shape[0]
            acc["prev"][i] += a[:, :, torch.arange(T - 1), tt - 1].mean(dim=(0, 2)).cpu().double() * ids.shape[0]
            acc["self"][i] += a[:, :, torch.arange(T - 1), tt].mean(dim=(0, 2)).cpu().double() * ids.shape[0]
            inwin = ((dist[1:] < args.window) | (t[None, :] == 0)).float()
            acc["win_mass"][i] += (a * inwin).sum(-1).mean(dim=(0, 2)).cpu().double() * ids.shape[0]
            fixed = relpos_matrix(kernels[i][0].to(dev), kernels[i][1].to(dev), T)[:, 1:]
            acc["tv_relpos"][i] += (0.5 * (a - fixed).abs().sum(-1)).mean(dim=(0, 2)).cpu().double() * ids.shape[0]
            # линейность MLP на отложенных данных
            xf = xm[:, 1:].float().reshape(-1, d)
            y = mo[:, 1:].float().reshape(-1, d)
            pred = xf @ Wr[i][:-1] + Wr[i][-1]
            sse[i] += ((y - pred) ** 2).sum().item()
            sst[i] += (y ** 2).sum().item()
            msum[i] += y.sum(0).cpu().double()
            pos0[i] += h[:, 0].float().norm(dim=-1).sum().item()
            rest[i] += h[:, 1:].float().norm(dim=-1).mean(1).sum().item()
        ntok += ids.shape[0] * (T - 1)
    # SST относительно среднего: Σy² − n·ȳ²
    sst = sst - (msum ** 2).sum(1) / ntok
    out = {
        "model": args.model, "env": env_info(dev), "args": vars(args),
        "heads": {k_: (v / nb).tolist() for k_, v in acc.items()},
        "q_norm_scaled_median": (qn / nb).tolist(),
        "k_norm_scaled_median": (kn / nb).tolist(),
        "mlp_linear_r2": (1 - sse / sst).tolist(),
        "resid_norm_pos0": (pos0[:Lnum] / nb).tolist(),
        "resid_norm_other": (rest[:Lnum] / nb).tolist(),
    }
    path = RESULTS_DIR / args.model / "profile.json"
    save_json(out, path)
    r2 = out["mlp_linear_r2"]
    for i in range(Lnum):
        print(f"L{i:2d} tv_relpos={sum(out['heads']['tv_relpos'][i]) / H:.3f} "
              f"win={sum(out['heads']['win_mass'][i]) / H:.3f} sink={sum(out['heads']['sink'][i]) / H:.3f} "
              f"|q|={out['q_norm_scaled_median'][i]:.1f} |k|={out['k_norm_scaled_median'][i]:.1f} "
              f"mlpR2={r2[i]:.3f} |h0|={out['resid_norm_pos0'][i]:.0f}")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
