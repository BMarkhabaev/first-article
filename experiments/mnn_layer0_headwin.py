"""Окно 64 только на «локальных» головах слоя 0 MNN-Pocket-0.3B (продолжение mnn_layer0_approx.py).

Голова считается локальной, если у исходного слоя доля массы внимания дальше 64 токенов меньше порога
(измерено в layer0_approx.json). Остальные головы — полный контекст. Варианты: таблицы fp32 и int8.

  python mnn_layer0_headwin.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from detlayers.common import pick_device, wikitext_windows  # noqa: E402
from detlayers.methods import Replacement  # noqa: E402
from detlayers.metrics import compare_lm  # noqa: E402
from mnn_layer0_approx import quantize_tables  # noqa: E402
from mnn_layer0_explicit import MODEL_DIR, OUT, ExplicitAttn0, IdTap, build_tables  # noqa: E402


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = pick_device(None)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32).to(dev).eval()
    theta = model.model.rotary_emb.inv_freq.float().cpu()
    tabs = {k: v.cpu() for k, v in build_tables(model).items()}
    tap = IdTap()
    model.model.embed_tokens.register_forward_pre_hook(lambda m, inp: setattr(tap, "ids", inp[0]))
    wins = wikitext_windows(tok, "wikitext-2-raw-v1", "test", 1024, max_windows=32)
    orig = model.model.layers[0].self_attn
    res = json.loads((OUT / "layer0_approx.json").read_text())
    m64 = res["mass_outside_window_by_head"]["64"]
    q8, _ = quantize_tables(tabs, 8)
    out = {}
    for thr in (0.25, 0.4):
        ws = [64 if m < thr else None for m in m64]
        local = [h for h, w in enumerate(ws) if w]
        for name, tb in ((f"headwin64<{thr}", tabs), (f"headwin64<{thr}+q8", q8)):
            mod = ExplicitAttn0(orig, tb, tap, theta, window=ws, n_sink=1)
            rep = Replacement(model)
            rep.add(0, "self_attn", mod)
            r = compare_lm(model, rep, wins, dev, bs=1)
            rep.disable()
            out[name] = {"kl": r["kl"], "top1": r["top1_agree"], "local_heads": local, "n_local": len(local)}
            print(f"{name:20s} локальных голов {len(local)}/14 {local}  KL {r['kl']:.2e}  top1 {r['top1_agree']:.4f}",
                  flush=True)
            del mod
    res["variants_headwin"] = out
    (OUT / "layer0_approx.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("сохранено", OUT / "layer0_approx.json")


if __name__ == "__main__":
    main()
