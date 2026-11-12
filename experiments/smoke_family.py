"""Смоук-тест методов на модели другого семейства: python smoke_family.py --model smol1.7b"""
import argparse, time, sys
sys.path.insert(0, ".")
from detlayers.common import load_model, pick_device, seed_all, calibration_windows, wikitext_windows
from detlayers.methods import build
from detlayers.metrics import compare_lm
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="smol1.7b"); a = ap.parse_args()
seed_all(0); dev = pick_device(None)
model, tok = load_model(a.model, dev, "fp32")
att = model.model.layers[0].self_attn
print("attn attrs:", [x for x in ("q_norm", "k_norm", "scaling", "head_dim", "num_key_value_groups", "layer_idx") if hasattr(att, x)], flush=True)
calib = calibration_windows(tok, 131_072, 512, seed=0)[:64]  # 8 окон при d=2048 переобучают регрессию
wins = wikitext_windows(tok, "wikitext-2-raw-v1", "test", 1024, max_windows=2)
TESTS = [("sparse", {"window": 64}), ("routed_mlp", {"active_ratio": 0.25, "predictor": "linear", "robust_tau": 10}),
         ("svdllm_mlp", {"ratio": 0.5}), ("a3", {}), ("relpos", {}), ("head_typed", {"frac": 0.5})]
for method, params in TESTS:
    t0 = time.time()
    try:
        rep, info = build(model, method, 2, calib, dev, bs=8, **params)
        r = compare_lm(model, rep, wins, dev, bs=1); rep.disable()
        print(f"OK {method}: KL {r['kl']:.4f} dPPL {100*(r['ppl']/r['ppl_teacher']-1):.1f}% ({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        print(f"FAIL {method}: {type(e).__name__}: {str(e)[:200]}", flush=True)
