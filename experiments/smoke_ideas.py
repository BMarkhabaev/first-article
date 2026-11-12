#!/usr/bin/env python
"""Быстрая проверка новых путей кода (ранг линейной части, низкоранговая поправка M-O4, CATS) на 8k токенов."""
import sys, time
from pathlib import Path
import argparse
sys.path.insert(0, str(Path(__file__).resolve().parent))
from detlayers.common import load_model, pick_device, seed_all, calibration_windows, wikitext_windows
from detlayers.methods import build
from detlayers.metrics import compare_lm
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="0.6b"); ap.add_argument("--dtype", default="fp32")
args = ap.parse_args()
seed_all(0); dev = pick_device(None)
model, tok = load_model(args.model, dev, args.dtype)
calib = calibration_windows(tok, 131_072, 512, seed=0)[:16]
wins = wikitext_windows(tok, "wikitext-2-raw-v1", "test", 1024, max_windows=2)
tests = [("ridge_mlp", {"robust_tau": 10, "rank": 64}), ("ridge_mlp", {"robust_tau": 10}),
         ("routed_mlp", {"active_ratio": 0.25, "predictor": "linear", "robust_tau": 10, "lin_rank": 64}),
         ("routed_mlp", {"active_ratio": 0.25, "predictor": "linear", "robust_tau": 10, "lam_lin": None}),
         ("cats_mlp", {"ratio": 0.25})]
for method, params in tests:
    t0 = time.time()
    rep, info = build(model, method, 2, calib, dev, bs=8, **params)
    r = compare_lm(model, rep, wins, dev, bs=1)
    rep.disable()
    assert r["kl"] == r["kl"] and r["kl"] < 5, f"подозрительный KL {r['kl']} для {method} {params}"
    costs = {k: getattr(m, "extra_cost", lambda: {})() for k, m in rep.new.items()}
    print(f"OK {method} {params}: KL {r['kl']:.4f} dPPL {100*(r['ppl']/r['ppl_teacher']-1):.1f}% {time.time()-t0:.0f}s cost {list(costs.values())[0]}", flush=True)
print("smoke_ideas: все пути кода работают")
