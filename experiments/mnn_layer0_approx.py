"""Аппроксимация явного внимания слоя 0 MNN-Pocket-0.3B (продолжение mnn_layer0_explicit.py).

Явная форма: таблицы на токен (амплитуды и фазы 32 частот RoPE для q и k, вектор v) + W_o.
Варианты удешевления, все без обучения:
  win W        — ключи только на расстоянии < W (+ позиция 0 как сток): вычисления O(T·W) вместо O(T²);
  q8 / q4      — таблицы квантованы: амплитуды и v — 8/4 бит с масштабом на (голову, частоту/канал),
                 фазы — 8/4 бит равномерно по окружности;
  amp1         — амплитуда частоты m у токена x ≈ s_h(x) · профиль_h(m): одно число на токен и голову
                 вместо 32 (профиль головы общий для всех токенов, s — проекция по наименьшим квадратам);
  комбинации   — окно + квантование.
Метрики — KL и совпадение top-1 с исходной моделью на 32 окнах WikiText-2 по 1024 токена; доля массы
внимания исходного слоя 0 вне окна W по головам; размер таблиц.

  python mnn_layer0_approx.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from detlayers.common import pick_device, wikitext_windows  # noqa: E402
from detlayers.methods import Replacement  # noqa: E402
from detlayers.metrics import compare_lm  # noqa: E402
from mnn_layer0_explicit import MODEL_DIR, OUT, ExplicitAttn0, IdTap, build_tables  # noqa: E402


def quantize_tables(t, bits):
    """Симуляция квантования: возвращает деквантованные таблицы и их размер в байтах."""
    L = 2 ** bits - 1
    out, nbytes = {}, 0
    for name in ("aq", "ak"):  # амплитуды ≥ 0: масштаб на (голову, частоту)
        a = t[name]
        sc = a.amax(dim=0, keepdim=True).clamp_min(1e-8)
        out[name] = torch.round(a / sc * L) / L * sc
        nbytes += a.numel() * bits / 8 + sc.numel() * 2
    for name in ("pq", "pk"):  # фазы: равномерно по окружности
        ph = t[name]
        out[name] = torch.round((ph + math.pi) / (2 * math.pi) * (L + 1)) % (L + 1) / (L + 1) * 2 * math.pi - math.pi
        nbytes += ph.numel() * bits / 8
    v = t["v"]  # симметрично, масштаб на (голову, канал)
    sc = v.abs().amax(dim=0, keepdim=True).clamp_min(1e-8)
    h = (L + 1) // 2 - 1
    out["v"] = torch.round(v / sc * h) / h * sc
    nbytes += v.numel() * bits / 8 + sc.numel() * 2
    return out, int(nbytes)


def amp_rank1(t):
    """Амплитуды: s_h(x) · prof_h(m); фазы и v без изменений."""
    out = dict(t)
    nbytes = 0
    for name in ("aq", "ak"):
        a = t[name]  # (V, heads, M)
        prof = a.mean(dim=0)  # (heads, M)
        prof = prof / prof.norm(dim=-1, keepdim=True)
        s = (a * prof[None]).sum(-1, keepdim=True)  # проекция на профиль
        out[name] = s * prof[None]
        nbytes += s.numel() * 2 + prof.numel() * 2
    nbytes += (t["pq"].numel() + t["pk"].numel() + t["v"].numel()) * 2
    return out, int(nbytes)


@torch.no_grad()
def mass_outside(model, wins, device, windows=(16, 32, 64, 128, 256)):
    """Доля массы внимания исходного слоя 0 на ключах дальше W (без позиции 0), по головам."""
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    layer, att, cfg = model.model.layers[0], model.model.layers[0].self_attn, model.config
    H, G, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    acc = {W: torch.zeros(H) for W in windows}
    pos0 = torch.zeros(H)
    n = 0
    for s in range(wins.shape[0]):
        ids = wins[s:s + 1].to(device)
        x = model.model.embed_tokens(ids)
        cos, sin = model.model.rotary_emb(x, torch.arange(ids.shape[1], device=device)[None])
        xn = layer.input_layernorm(x)
        q = att.q_norm(att.q_proj(xn).view(1, -1, H, D)).transpose(1, 2)
        k = att.k_norm(att.k_proj(xn).view(1, -1, G, D)).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k = k.repeat_interleave(H // G, dim=1)
        T = ids.shape[1]
        sc = (q @ k.transpose(-1, -2))[0].float() * att.scaling
        i = torch.arange(T, device=device)
        causal = i[:, None] >= i[None, :]
        p = torch.softmax(sc.masked_fill(~causal, float("-inf")), -1)
        dist = i[:, None] - i[None, :]
        pos0 += p[:, 1:, 0].mean(-1).cpu()
        for W in windows:
            far = (dist >= W) & (i[None, :] > 0)
            acc[W] += (p * far).sum(-1)[:, W:].mean(-1).cpu()  # только запросы, у которых есть дальние ключи
        n += 1
    return {W: (acc[W] / n).tolist() for W in windows}, (pos0 / n).tolist()


def main():
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = pick_device(None)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32).to(device).eval()
    cfg = model.config
    theta = model.model.rotary_emb.inv_freq.float().cpu()
    tabs = {k: v.cpu() for k, v in build_tables(model).items()}
    full_bytes = int(sum(v.numel() for v in tabs.values()) * 2)
    tap = IdTap()
    model.model.embed_tokens.register_forward_pre_hook(lambda m, inp: setattr(tap, "ids", inp[0]))
    wins = wikitext_windows(tok, "wikitext-2-raw-v1", "test", 1024, max_windows=32)
    orig = model.model.layers[0].self_attn

    mass, pos0 = mass_outside(model, wins[:8], device)
    for W, m in mass.items():
        print(f"масса внимания дальше {W:3d}: средняя {sum(m) / len(m):.3f}, макс. по головам {max(m):.3f}", flush=True)

    q8, b8 = quantize_tables(tabs, 8)
    q4, b4 = quantize_tables(tabs, 4)
    a1, ba1 = amp_rank1(tabs)
    variants = {
        "exact": (tabs, {}, full_bytes),
        "win16": (tabs, {"window": 16, "n_sink": 1}, full_bytes),
        "win32": (tabs, {"window": 32, "n_sink": 1}, full_bytes),
        "win64": (tabs, {"window": 64, "n_sink": 1}, full_bytes),
        "win128": (tabs, {"window": 128, "n_sink": 1}, full_bytes),
        "q8": (q8, {}, b8),
        "q4": (q4, {}, b4),
        "amp1": (a1, {}, ba1),
        "win64+q8": (q8, {"window": 64, "n_sink": 1}, b8),
        "win64+q4": (q4, {"window": 64, "n_sink": 1}, b4),
    }
    res = {"model": "taobao-mnn/MNN-Pocket-0.3B", "layer": 0, "eval": {"windows": int(wins.shape[0]), "len": 1024},
           "orig_attn_weights_bytes_fp16": int(sum(p.numel() for p in orig.parameters()) * 2),
           "mass_outside_window_by_head": mass, "attn_to_pos0_by_head": pos0, "variants": {}}
    for name, (tb, kw, nbytes) in variants.items():
        mod = ExplicitAttn0(orig, tb, tap, theta, **kw)
        rep = Replacement(model)
        rep.add(0, "self_attn", mod)
        r = compare_lm(model, rep, wins, device, bs=1)
        rep.disable()
        row = {"kl": r["kl"], "top1": r["top1_agree"], "dppl_pct": 100 * (r["ppl"] / r["ppl_teacher"] - 1),
               "table_mb": round(nbytes / 2 ** 20, 1), "keys_per_query": kw.get("window", "все")}
        res["variants"][name] = row
        print(f"{name:10s} KL {row['kl']:.2e}  top1 {row['top1']:.4f}  таблицы {row['table_mb']:6.1f} МБ  "
              f"ключей на запрос: {row['keys_per_query']}", flush=True)
        del mod
    res["seconds"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "layer0_approx.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"исходные веса внимания слоя 0: {res['orig_attn_weights_bytes_fp16'] / 2**20:.1f} МБ; "
          f"{res['seconds']}s; сохранено {OUT / 'layer0_approx.json'}")


if __name__ == "__main__":
    main()
