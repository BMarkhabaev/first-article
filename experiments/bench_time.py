#!/usr/bin/env python
"""Замер времени: окно 64 + сток (ленточная реализация на чистом PyTorch, O(T·w)) против полного внимания
(SDPA, O(T²)). (1) один слой внимания при T = 1k…8k (16k — только лента); (2) prefill всей модели
с заменой внимания первых k слоёв при T ≤ 8k. Медиана 5 запусков после прогрева.
Результат: results/<model>/timing.json. Числа зависят от фоновой нагрузки — см. поле "note"."""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detlayers.attention import DetAttention, SparseAttention  # noqa: E402
from detlayers.common import load_model, pick_device, save_json, seed_all  # noqa: E402
from detlayers.methods import Replacement  # noqa: E402

ROOT = Path(__file__).resolve().parent


def banded_attention(q, k, v, w, scale, n_sink=1, chunk=1024):
    """Каузальное внимание на ленте t-w+1..t (w ключей, как в SparseAttention) плюс n_sink первых ключей
    (стоки). q,k,v: (B,H,T,D).
    Память O(T·w), как у специализированного ядра скользящего окна."""
    B, H, T, D = q.shape
    out = torch.empty_like(q)
    kp, vp = F.pad(k, (0, 0, w, 0)), F.pad(v, (0, 0, w, 0))
    ar = torch.arange(1, w + 1, device=q.device)
    ks, vs = k[:, :, :n_sink], v[:, :, :n_sink]  # (B,H,S,D)
    sidx = torch.arange(n_sink, device=q.device)
    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        idx = torch.arange(s, e, device=q.device)[:, None] + ar[None, :]  # индексы в kp: (n, w)
        kb, vb = kp[:, :, idx], vp[:, :, idx]  # (B,H,n,w,D)
        qc = q[:, :, s:e]
        sc = torch.einsum("bhnd,bhnwd->bhnw", qc, kb) * scale
        sc = sc.masked_fill(~((idx - w) >= 0)[None, None], float("-inf"))
        scs = torch.einsum("bhnd,bhsd->bhns", qc, ks) * scale  # стоки
        t = torch.arange(s, e, device=q.device)[:, None]
        outside = sidx[None, :] <= (t - w)  # сток вне ленты (иначе он уже учтён в ленте)
        scs = scs.masked_fill(~outside[None, None], float("-inf"))
        p = torch.softmax(torch.cat([sc, scs], -1).float(), -1).to(q.dtype)
        out[:, :, s:e] = (torch.einsum("bhnw,bhnwd->bhnd", p[..., :w], vb)
                          + torch.einsum("bhns,bhsd->bhnd", p[..., w:], vs))
    return out


class BandedAttention(DetAttention):
    def __init__(self, orig, window=64, n_sink=1):
        super().__init__(orig)
        self.window, self.n_sink = window, n_sink

    def mix(self, q, k, v):
        return banded_attention(q, k, v, self.window, self.orig.scaling, self.n_sink)


def sync(dev):
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


def timeit(fn, dev, reps=5):
    fn()
    sync(dev)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sync(dev)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="0.6b")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--lengths", default="1024,2048,4096,8192")
    args = ap.parse_args()
    seed_all(0)
    dev = pick_device(None)
    model, tok = load_model(args.model, dev, "bf16")
    cfg = model.config
    H, D = cfg.num_attention_heads, cfg.head_dim
    lengths = [int(x) for x in args.lengths.split(",")]
    res = {"model": args.model, "device": dev.type, "dtype": "bf16", "window": args.window, "k": args.k,
           "note": "ленточная реализация на чистом PyTorch (без специализированных ядер); медиана 5 запусков; "
                   "возможна фоновая нагрузка", "layer": {}, "model_prefill": {}}
    # --- корректность ленты против плотной маски (SparseAttention) на коротком T
    L0 = model.model.layers[0].self_attn
    x = torch.randn(1, 512, cfg.hidden_size, device=dev, dtype=torch.bfloat16)
    pe = model.model.rotary_emb(x, torch.arange(512, device=dev)[None])
    a_band, _ = BandedAttention(L0, args.window)(x, pe)
    a_dense, _ = SparseAttention(L0, window=args.window, n_sink=1)(x, pe)
    res["band_vs_dense_max_abs_diff"] = float((a_band.float() - a_dense.float()).abs().max())
    # --- один слой внимания: полное SDPA против ленты
    scale = D ** -0.5
    for T in lengths + [16384]:
        q, k, v = (torch.randn(1, H, T, D, device=dev, dtype=torch.bfloat16) for _ in range(3))
        r = {}
        try:
            r["band_ms"] = 1000 * timeit(lambda: banded_attention(q, k, v, args.window, scale), dev)
        except RuntimeError as e:
            r["band_ms"] = None
            r["band_error"] = str(e)[:120]
        if T <= max(lengths):
            try:
                r["full_ms"] = 1000 * timeit(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True), dev)
            except RuntimeError as e:
                r["full_ms"] = None
                r["full_error"] = str(e)[:120]
        res["layer"][T] = r
        print("слой", T, r, flush=True)
        del q, k, v
    # --- prefill всей модели: исходная против замены внимания первых k слоёв
    rep = Replacement(model)
    for i in range(args.k):
        rep.add(i, "self_attn", BandedAttention(model.model.layers[i].self_attn, args.window))
    for T in lengths:
        ids = torch.randint(0, cfg.vocab_size, (1, T), device=dev)
        r = {}
        try:
            rep.disable()
            r["full_ms"] = 1000 * timeit(lambda: model(ids).logits, dev)
            rep.enable()
            r["band_ms"] = 1000 * timeit(lambda: model(ids).logits, dev)
            rep.disable()
            r["speedup"] = r["full_ms"] / r["band_ms"]
        except RuntimeError as e:
            r["error"] = str(e)[:120]
            rep.disable()
        res["model_prefill"][T] = r
        print("модель", T, r, flush=True)
    out = ROOT / "results" / args.model / "timing.json"
    save_json(res, out)
    print("saved", out)


if __name__ == "__main__":
    main()
