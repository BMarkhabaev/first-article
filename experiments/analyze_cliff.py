#!/usr/bin/env python
"""Диагностика обрыва на k=3: почему замена MLP первых трёх слоёв (M-O4) даёт +27% PPL,
а четырёх — только +11%. Гипотеза: слой 3 (первый слой со стоком внимания / массивной активацией)
усиливает ошибку остаточного потока, пришедшую с заменённых слоёв.

Считает для нескольких конфигураций (M-O4 k=2,3,4; окно k=3; SVD-LLM k=3):
  1) относительную ошибку остаточного потока на входе каждого слоя (без позиции 0) и
     коэффициенты усиления e_{i+1}/e_i;
  2) на первом незаменённом слое k: вклад подблоков (внимание / MLP), массу стока у учителя и ученика,
     долю энергии ошибки в «массивных» каналах (каналы массивной активации позиции 0);
  3) корректоры на входе слоя k, подогнанные по калибровке в замкнутой форме:
     diag (поканальная аффинная), full (гребневая d×d), lowrank32 (+diag), massive8 (только массивные каналы);
     для каждого — KL / PPL / top-1 на 32 окнах WikiText-2.
Результат: results/<model>/cliff/analysis.json (+ .log).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detlayers.attention import DetAttention, exact_attention_weights  # noqa: E402
from detlayers.common import (calibration_windows, embed, load_model, pick_device, run_layers,  # noqa: E402
                              save_json, seed_all, wikitext_windows)
from detlayers.methods import Replacement, build  # noqa: E402

ROOT = Path(__file__).resolve().parent


def rel(a, b):
    """Относительная ошибка по позициям ≥1, усреднённая по токенам."""
    a, b = a[:, 1:].float(), b[:, 1:].float()
    return ((a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-6)).mean().item()


@torch.no_grad()
def states(model, ids):
    """Входы всех слоёв + выход последнего (до финальной нормы) при текущем состоянии модели."""
    h, pos, pe = embed(model, ids)
    hs = [h]
    for i in range(len(model.model.layers)):
        h = run_layers(model, model.model.layers, h, pos, pe, i, i + 1)
        hs.append(h)
    return hs, pos, pe


@torch.no_grad()
def eval_lm(model, wins, device, enable, disable, bs=1):
    nll_t = nll_s = kl = agree = 0.0
    n = 0
    for s in range(0, wins.shape[0], bs):
        ids = wins[s:s + bs].to(device)
        disable()
        lt = model(ids).logits[:, :-1].float()
        enable()
        ls = model(ids).logits[:, :-1].float()
        tgt = ids[:, 1:]
        lpt, lps = F.log_softmax(lt, -1), F.log_softmax(ls, -1)
        nll_t += -lpt.gather(-1, tgt[..., None]).sum().item()
        nll_s += -lps.gather(-1, tgt[..., None]).sum().item()
        kl += (lpt.exp() * (lpt - lps)).sum().item()
        agree += (lt.argmax(-1) == ls.argmax(-1)).sum().item()
        n += tgt.numel()
    disable()
    return {"ppl_teacher": math.exp(nll_t / n), "ppl": math.exp(nll_s / n), "dppl_pct": 100 * (math.exp(nll_s / n) / math.exp(nll_t / n) - 1),
            "kl": kl / n, "top1_agree": agree / n}


class Corrected(nn.Module):
    """Слой с корректором на входе: h' = h W + b для позиций ≥ 1 (позиция 0 не трогается)."""

    def __init__(self, layer, W, b):
        super().__init__()
        self.layer = layer
        self.register_buffer("W", W)
        self.register_buffer("b", b)

    def forward(self, h, *a, **kw):
        h1 = h[:, 1:].float() @ self.W + self.b
        h = torch.cat([h[:, :1], h1.to(h.dtype)], 1)
        return self.layer(h, *a, **kw)


@torch.no_grad()
def layer_diag(model, rep, wins, device, k, n_massive=8):
    """Ошибки по слоям, усиление, подблоки слоя k, масса стока, массивные каналы."""
    L = len(model.model.layers)
    errs = torch.zeros(L + 1)
    sub = {"e_in": 0.0, "e_attn_out": 0.0, "e_mid": 0.0, "e_mlp_out": 0.0, "e_out": 0.0,
           "e_out_patch_attnT": 0.0, "e_out_patch_mlpT": 0.0}
    sink = {"mass_T": 0.0, "mass_S": 0.0, "mass_absdiff": 0.0}
    massive_frac = {"err_S_in": 0.0, "err_S_out": 0.0, "signal_T_in": 0.0}
    nb = 0
    layer = model.model.layers[k]  # незаменённый слой (один и тот же объект у учителя и ученика)
    det = DetAttention(layer.self_attn)
    massive_idx = None
    for s in range(wins.shape[0]):
        ids = wins[s:s + 1].to(device)
        with rep.teacher():
            hT, pos, pe = states(model, ids)
        rep.enable()
        hS, _, _ = states(model, ids)
        rep.disable()
        for i in range(L + 1):
            errs[i] += rel(hS[i], hT[i])
        # --- подблоки слоя k
        xT, xS = hT[k], hS[k]
        aT, _ = layer.self_attn(layer.input_layernorm(xT), position_embeddings=pe, attention_mask=None)
        aS, _ = layer.self_attn(layer.input_layernorm(xS), position_embeddings=pe, attention_mask=None)
        midT, midS = xT + aT, xS + aS
        mT, mS = layer.mlp(layer.post_attention_layernorm(midT)), layer.mlp(layer.post_attention_layernorm(midS))
        outT, outS = midT + mT, midS + mS
        sub["e_in"] += rel(xS, xT)
        sub["e_attn_out"] += rel(aS, aT)
        sub["e_mid"] += rel(midS, midT)
        sub["e_mlp_out"] += rel(mS, mT)
        sub["e_out"] += rel(outS, outT)
        # патчи: вход ученика, но внимание учителя (и наоборот)
        midP = xS + aT
        sub["e_out_patch_attnT"] += rel(midP + layer.mlp(layer.post_attention_layernorm(midP)), outT)
        sub["e_out_patch_mlpT"] += rel(midS + mT, outT)
        # --- масса стока на слое k
        for tag, x in (("T", xT), ("S", xS)):
            q, kk, _ = det.qkv(layer.input_layernorm(x), pe)
            P = exact_attention_weights(q, kk, layer.self_attn.scaling)  # (1,H,T,T)
            m = P[0, :, 1:, 0]  # масса на позиции 0 для запросов ≥1
            sink[f"mass_{tag}"] += m.mean().item()
            if tag == "T":
                mT_ = m
            else:
                sink["mass_absdiff"] += (m - mT_).abs().mean().item()
        # --- массивные каналы (по учителю на входе слоя k, позиция 0)
        if massive_idx is None:
            massive_idx = xT[0, 0].abs().topk(n_massive).indices
        dIn, dOut = (xS - xT)[0, 1:], (outS - outT)[0, 1:]
        massive_frac["err_S_in"] += (dIn[:, massive_idx] ** 2).sum().item() / (dIn ** 2).sum().item()
        massive_frac["err_S_out"] += (dOut[:, massive_idx] ** 2).sum().item() / (dOut ** 2).sum().item()
        massive_frac["signal_T_in"] += (xT[0, 1:][:, massive_idx] ** 2).sum().item() / (xT[0, 1:] ** 2).sum().item()
        nb += 1
    errs /= nb
    gain = [(errs[i + 1] / errs[i]).item() if errs[i] > 0 else None for i in range(L)]
    return {"rel_err_by_layer_input": [round(x, 5) for x in errs.tolist()],
            "gain": [round(g, 3) if g else None for g in gain],
            "sub_blocks_layer_k": {a: round(v / nb, 5) for a, v in sub.items()},
            "sink_layer_k": {a: round(v / nb, 5) for a, v in sink.items()},
            "massive_channels": {"idx": massive_idx.tolist(), **{a: round(v / nb, 4) for a, v in massive_frac.items()}}}


@torch.no_grad()
def fit_correctors(model, rep, calib, device, k, bs, massive_idx, lam=1e-3, rank=32, robust_tau=None):
    """Статистики (h_S, h_T) на входе слоя k по калибровке → несколько корректоров.
    robust_tau: строки с ||h_T|| или ||h_S|| > tau·медиана исключаются (токены с массивной активацией)."""
    d = model.config.hidden_size
    G = torch.zeros(d + 1, d + 1, dtype=torch.float64)
    B = torch.zeros(d + 1, d, dtype=torch.float64)
    sx = torch.zeros(d, dtype=torch.float64); sy = torch.zeros(d, dtype=torch.float64)
    sxx = torch.zeros(d, dtype=torch.float64); sxy = torch.zeros(d, dtype=torch.float64)
    n = 0
    for s in range(0, calib.shape[0], bs):
        ids = calib[s:s + bs].to(device)
        with rep.teacher():
            h, pos, pe = embed(model, ids)
            xT = run_layers(model, model.model.layers, h, pos, pe, 0, k)
        rep.enable()
        h, pos, pe = embed(model, ids)
        xS = run_layers(model, model.model.layers, h, pos, pe, 0, k)
        rep.disable()
        x = xS[:, 1:].float().reshape(-1, d); y = xT[:, 1:].float().reshape(-1, d)
        if robust_tau:
            nx, ny = x.norm(dim=-1), y.norm(dim=-1)
            keep = (nx <= robust_tau * nx.median()) & (ny <= robust_tau * ny.median())
            x, y = x[keep], y[keep]
        phi = torch.cat([x, torch.ones_like(x[:, :1])], 1)
        G += (phi.T @ phi).cpu().double(); B += (phi.T @ y).cpu().double()
        sx += x.sum(0).cpu().double(); sy += y.sum(0).cpu().double()
        sxx += (x * x).sum(0).cpu().double(); sxy += (x * y).sum(0).cpu().double()
        n += x.shape[0]
    reg = lam * G.diagonal().mean()
    Wf = torch.linalg.solve(G + reg * torch.eye(d + 1, dtype=torch.float64), B)  # (d+1, d)
    W_full, b_full = Wf[:-1], Wf[-1]
    # diag: y_c ≈ s_c x_c + t_c
    var = sxx - sx * sx / n
    s_ = (sxy - sx * sy / n) / var.clamp_min(1e-9)
    t_ = (sy - s_ * sx) / n
    W_diag, b_diag = torch.diag(s_), t_
    # lowrank + diag
    D = torch.diag(torch.diagonal(W_full))
    U, S, Vh = torch.linalg.svd(W_full - D)
    W_lr = D + (U[:, :rank] * S[:rank]) @ Vh[:rank]
    # только массивные каналы (выходные столбцы S из полной карты, остальные — тождество)
    W_m, b_m = torch.eye(d, dtype=torch.float64), torch.zeros(d, dtype=torch.float64)
    W_m[:, massive_idx] = W_full[:, massive_idx]; b_m[massive_idx] = b_full[massive_idx]
    # diag только на массивных каналах
    W_dm, b_dm = torch.eye(d, dtype=torch.float64), torch.zeros(d, dtype=torch.float64)
    W_dm[massive_idx, massive_idx] = s_[massive_idx]; b_dm[massive_idx] = t_[massive_idx]
    f = lambda W, b: (W.float().to(device), b.float().to(device))  # noqa: E731
    return {"diag": f(W_diag, b_diag), "full": f(W_full, b_full), f"lowrank{rank}+diag": f(W_lr, b_full),
            f"massive{len(massive_idx)}": f(W_m, b_m), f"diag_massive{len(massive_idx)}": f(W_dm, b_dm)}, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="0.6b")
    ap.add_argument("--configs", default="routed3,routed2,routed4,window3,svdllm3")
    ap.add_argument("--calib-tokens", type=int, default=65_536)
    ap.add_argument("--calib-bs", type=int, default=8)
    ap.add_argument("--diag-windows", type=int, default=8)
    ap.add_argument("--eval-windows", type=int, default=32)
    ap.add_argument("--correctors-for", default="routed3,routed4,routed2")
    ap.add_argument("--robust-corr-tau", type=float, default=None, help="робастная подгонка корректоров")
    ap.add_argument("--out-name", default="analysis.json")
    args = ap.parse_args()
    seed_all(0)
    device = pick_device(None)
    model, tok = load_model(args.model, device, "fp32")
    calib = calibration_windows(tok, 131_072, 512, seed=0)  # как в sweep (тот же набор), берём первые N
    calib = calib[: max(1, args.calib_tokens // 512)]
    wins = wikitext_windows(tok, "wikitext-2-raw-v1", "test", 1024, max_windows=args.eval_windows)
    dwins = wins[: args.diag_windows]
    CFG = {"routed": ("routed_mlp", {"active_ratio": 0.25, "predictor": "linear"}),
           "routedr": ("routed_mlp", {"active_ratio": 0.25, "predictor": "linear", "robust_tau": 10}),
           "window": ("sparse", {"window": 64}), "svdllm": ("svdllm_mlp", {"ratio": 0.5})}
    out_path = ROOT / "results" / args.model / "cliff" / args.out_name
    res = {"model": args.model, "calib_tokens": calib.numel(), "diag_windows": len(dwins),
           "eval_windows": len(wins), "configs": {}}
    for name in args.configs.split(","):
        base, k = name.rstrip("0123456789"), int(name[len(name.rstrip("0123456789")):])
        if base.endswith("r") and base not in CFG:
            base = base[:-1]
        method, params = CFG[base]
        t0 = time.time()
        full_calib = calibration_windows(tok, 131_072, 512, seed=0)
        rep, info = build(model, method, k, full_calib, device, bs=args.calib_bs, **params)
        rep.disable()
        r = {"method": method, "k": k, "build_seconds": round(time.time() - t0, 1)}
        r["lm"] = eval_lm(model, wins, device, rep.enable, rep.disable)
        r["diag"] = layer_diag(model, rep, dwins, device, k)
        print(name, json.dumps({"lm": r["lm"], "gain_k-1..k+1": r["diag"]["gain"][max(0, k - 1):k + 2],
                                "sub": r["diag"]["sub_blocks_layer_k"], "sink": r["diag"]["sink_layer_k"],
                                "massive": r["diag"]["massive_channels"]}), flush=True)
        if name in args.correctors_for.split(","):
            t0 = time.time()
            midx = torch.tensor(r["diag"]["massive_channels"]["idx"])
            corrs, ntok = fit_correctors(model, rep, calib, device, k, args.calib_bs, midx, robust_tau=args.robust_corr_tau)
            r["correctors"] = {"fit_tokens": ntok, "fit_seconds": round(time.time() - t0, 1), "robust_tau": args.robust_corr_tau}
            for cname, (W, b) in corrs.items():
                rep2 = Replacement(model)
                rep2.add(k, "block", Corrected(model.model.layers[k], W, b))
                en = lambda: (rep.enable(), rep2.enable())  # noqa: E731
                dis = lambda: (rep2.disable(), rep.disable())  # noqa: E731
                lm = eval_lm(model, wins, device, en, dis)
                # ошибка после слоя k с корректором
                rep.enable(); rep2.enable()
                e_after = 0.0
                for s in range(len(dwins)):
                    ids = dwins[s:s + 1].to(device)
                    hS, _, _ = states(model, ids)
                    rep2.disable(); rep.disable()
                    with torch.no_grad():
                        hT, _, _ = states(model, ids)
                    rep.enable(); rep2.enable()
                    e_after += rel(hS[k + 1], hT[k + 1])
                rep2.disable(); rep.disable()
                lm["rel_err_after_layer_k"] = round(e_after / len(dwins), 5)
                lm["extra_macs_per_token"] = int((W != 0).sum().item()) if cname != "full" else W.numel()
                r["correctors"][cname] = lm
                print(f"   корректор {cname}: dPPL {lm['dppl_pct']:.1f}%  KL {lm['kl']:.4f}  "
                      f"err после слоя {k}: {lm['rel_err_after_layer_k']:.4f}", flush=True)
        res["configs"][name] = r
        save_json(res, out_path)
        del rep
    print("saved", out_path)


if __name__ == "__main__":
    main()
