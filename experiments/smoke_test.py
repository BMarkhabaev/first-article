import sys, time, json, torch
sys.path.insert(0, __import__("os").path.dirname(__file__))
from detlayers.common import *
from detlayers.methods import build
from detlayers.metrics import compare_lm
dev = pick_device(); seed_all(0)
m, tok = load_model("0.6b", dev, "fp32")
calib = calibration_windows(tok, 128*256, 256)
wins = wikitext_windows(tok, "wikitext-2-raw-v1", "test", 512, max_windows=2)
cases = [
  ("sanity: sparse w=T", "sparse", 2, dict(window=10**6)),
  ("sanity: head_typed frac=0", "head_typed", 2, dict(frac=0.0)),
  ("sanity: a3 ratio=1", "a3", 2, dict(ratio_qk=1.0, ratio_vo=1.0, ratio_ffn=1.0)),
  ("drop", "drop", 2, {}),
  ("sparse w64", "sparse", 2, dict(window=64)),
  ("strided", "strided", 2, dict(window=32, stride=32)),
  ("performer", "performer", 2, dict(features=256)),
  ("linear_elu", "linear_elu", 2, {}),
  ("relpos", "relpos", 2, dict(max_offset=64)),
  ("head_typed .5", "head_typed", 2, dict(frac=0.5)),
  ("a3 .5", "a3", 2, {}),
  ("ridge_conv", "ridge_conv", 2, dict(window=8, pca_dim=128)),
  ("ngram", "ngram", 2, dict(m=3, sweeps=1)),
  ("salience it0", "salience", 2, dict(iters=0)),
  ("salience it3", "salience", 2, dict(iters=3)),
  ("salience it3 ex25%", "salience", 2, dict(iters=3, exact_frac=0.25)),
  ("sanity: a3_mlp ratio=1", "a3_mlp", 2, dict(ratio=1.0)),
  ("sanity: topk ratio=1", "topk_mlp", 2, dict(ratio=1.0)),
  ("a3_mlp .5", "a3_mlp", 2, dict(ratio=0.5)),
  ("topk_mlp .25 (DejaVu oracle)", "topk_mlp", 2, dict(ratio=0.25)),
  ("asvd_mlp .5", "asvd_mlp", 2, dict(ratio=0.5)),
  ("svdllm_mlp .5", "svdllm_mlp", 2, dict(ratio=0.5)),
  ("moe_mlp 32/25%", "moe_mlp", 2, dict(experts=32, active_frac=0.25)),
  ("ridge_mlp", "ridge_mlp", 2, {}),
  ("ridge_mlp keep25%", "ridge_mlp", 2, dict(keep_ratio=0.25)),
  ("table_mlp", "table_mlp", 2, {}),
  ("cluster_mlp 16", "cluster_mlp", 2, dict(clusters=16)),
  ("sparse+ridge_mlp", "sparse+ridge_mlp", 2, dict(attn=dict(window=64), mlp=dict(keep_ratio=0.25))),
]
only = sys.argv[1:] 
for name, meth, k, p in cases:
    if only and meth not in only: continue
    t=time.time()
    rep, info = build(m, meth, k, calib, dev, bs=4, **p)
    r = compare_lm(m, rep, wins, dev, probe_layer=k)
    rep.disable()
    if hasattr(rep, "new"):
        for mod in rep.new.values():
            if hasattr(mod, "tap"): mod.tap.handle.remove()
    print(f"{name:28s} ppl={r['ppl']:9.3f} (T {r['ppl_teacher']:.3f}) kl={r['kl']:.4f} top1={r['top1_agree']:.3f} relk={r.get('rel_err_layer_k',0):.3f} {time.time()-t:.0f}s", flush=True)
