#!/usr/bin/env python
"""Теория отравления: влияние одной строки на гребневую регрессию (формула Шермана–Моррисона).
Для слоёв 0..3 по калибровке (путь учителя, позиции ≥1): φ = [x̂, 1] — вход MLP, y = MLP(x̂) — цель.
Строка-выброс: максимальная ||y||. Величины: доля энергии s = ||y_out||²/Σ||y||²; рычаг h = φ_outᵀA⁻¹φ_out
(A = G_без_выброса + λI); невязка r_out = y_out − φ_outᵀW_robust; связь c_t = φ_tᵀA⁻¹φ_out с обычными токенами;
«утечка» в предсказания обычных токенов: Δŷ_t = c_t·r_out/(1+h), относительно ||y_t||. Проверяется против прямой
разности W_naive − W_robust. Результат: results/<model>/trap/leverage.json."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detlayers.common import calibration_windows, load_model, pick_device, save_json, seed_all  # noqa: E402
from detlayers.mlp import _ridge, iterate_mlp_inputs  # noqa: E402

ROOT = Path(__file__).resolve().parent


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="0.6b")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--calib-tokens", type=int, default=131_072)
    ap.add_argument("--lam", type=float, default=1e-3)
    args = ap.parse_args()
    seed_all(0)
    dev = pick_device(None)
    model, tok = load_model(args.model, dev, "fp32")
    d = model.config.hidden_size
    calib = calibration_windows(tok, args.calib_tokens, 512, seed=0)
    k = args.layers
    G = [torch.zeros(d + 1, d + 1, dtype=torch.float64) for _ in range(k)]
    B = [torch.zeros(d + 1, d, dtype=torch.float64) for _ in range(k)]
    E = [0.0] * k
    out_row = [None] * k  # (||y||, φ, y)
    # проход 1: суммы и строка-выброс
    for i, x, _ in iterate_mlp_inputs(model, calib, k, 8, dev):
        mlp = model.model.layers[i].mlp
        xf = x[:, 1:].float().reshape(-1, d)
        y = mlp(x[:, 1:]).float().reshape(-1, d)
        phi = torch.cat([xf, torch.ones_like(xf[:, :1])], 1)
        G[i] += (phi.T @ phi).cpu().double()
        B[i] += (phi.T @ y).cpu().double()
        yn = y.norm(dim=-1)
        E[i] += (yn ** 2).sum().item()
        j = int(yn.argmax())
        if out_row[i] is None or yn[j].item() > out_row[i][0]:
            out_row[i] = (yn[j].item(), phi[j].cpu().double(), y[j].cpu().double())
    res = {"model": args.model, "lam": args.lam, "layers": {}}
    Ainv, Wr, r_out, hh = [None] * k, [None] * k, [None] * k, [0.0] * k
    for i in range(k):
        yn_out, phi_o, y_o = out_row[i]
        Gr, Br = G[i] - torch.outer(phi_o, phi_o), B[i] - torch.outer(phi_o, y_o)
        lam = args.lam * Gr.diagonal().mean()
        A = Gr + lam * torch.eye(d + 1, dtype=torch.float64)
        Ainv[i] = torch.linalg.inv(A)
        Wr[i] = Ainv[i] @ Br
        Wn = _ridge(G[i], B[i], args.lam)
        hh[i] = float(phi_o @ Ainv[i] @ phi_o)
        r_out[i] = y_o - phi_o @ Wr[i]
        dW_pred = torch.outer(Ainv[i] @ phi_o, r_out[i]) / (1 + hh[i])
        dW = Wn - Wr[i]
        res["layers"][i] = {"y_out_norm": yn_out, "energy_share": (yn_out ** 2) / E[i], "leverage_h": hh[i],
                            "r_out_norm": float(r_out[i].norm()),
                            "sherman_morrison_rel_err": float((dW - dW_pred).norm() / dW.norm().clamp_min(1e-12))}
    # проход 2: утечка в обычные токены
    spill = [[] for _ in range(k)]
    coup = [[] for _ in range(k)]
    for i, x, _ in iterate_mlp_inputs(model, calib[:64], k, 8, dev):  # 32k токенов достаточно
        mlp = model.model.layers[i].mlp
        xf = x[:, 1:].float().reshape(-1, d)
        y = mlp(x[:, 1:]).float().reshape(-1, d)
        phi = torch.cat([xf, torch.ones_like(xf[:, :1])], 1).cpu().double()
        c = phi @ (Ainv[i] @ out_row[i][1])  # (n,)
        dy = torch.abs(c) * r_out[i].norm() / (1 + hh[i])
        spill[i].append((dy / y.norm(dim=-1).cpu().double().clamp_min(1e-6)))
        coup[i].append(c.abs())
    for i in range(k):
        s, c = torch.cat(spill[i]), torch.cat(coup[i])
        res["layers"][i].update({"mean_rel_spill": float(s.mean()), "p90_rel_spill": float(s.quantile(0.9)),
                                 "mean_abs_coupling": float(c.mean())})
        L = res["layers"][i]
        print(f"слой {i}: ||y_out|| {L['y_out_norm']:.0f} доля энергии {100 * L['energy_share']:.1f}% рычаг h {L['leverage_h']:.3f} "
              f"||r_out|| {L['r_out_norm']:.0f} связь |c| {L['mean_abs_coupling']:.2e} утечка отн. {100 * L['mean_rel_spill']:.1f}% "
              f"(p90 {100 * L['p90_rel_spill']:.1f}%) проверка ШМ {L['sherman_morrison_rel_err']:.1e}", flush=True)
    save_json(res, ROOT / "results" / args.model / "trap" / "leverage.json")


if __name__ == "__main__":
    main()
