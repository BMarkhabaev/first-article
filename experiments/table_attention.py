"""Табличное внимание (формула с доски) для моделей архитектуры Qwen3: MNN-Pocket-0.3B, Qwen3-0.6B, Qwen3-1.7B.

Логит головы h между запросом x на позиции t и ключом y на позиции t−Δ после RoPE:
    A_h(x, y, Δ) = s · Σ_m |z^q_m|·|z^k_m| · cos(θ_m Δ + arg z^q_m − arg z^k_m),   z_m = c[m] + i·c[m + D/2].
В слое 0 q, k, v зависят только от токена → таблицы на токен воспроизводят слой точно. В слое ℓ > 0 вход зависит от
контекста; «таблица» слоя ℓ — средние q, k, v токена по калибровке (позиция 0 всегда считается точно).

Режимы:
  layer0    — проверка формулы, частоты, локальность голов, варианты сжатия (fp16, int8, int4, медленные частоты без Δ,
              урезанные частоты, окна на все/локальные головы)        → results/<m>/table_attn/layer0.json
  alllayers — для каждого слоя: доля дисперсии q, k, v, объяснённая токеном (R²); замена одного слоя таблицей
              (q,k,v или только q,k); замена первых k слоёв подряд        → results/<m>/table_attn/alllayers.json
  bench     — время внимания слоя 0: исходное против табличного (сбор q,k,v из таблиц + RoPE + SDPA), и всей модели
                                                                            → results/<m>/table_attn/bench.json

  python table_attention.py layer0 --model pocket0.3b
  python table_attention.py alllayers --model pocket0.3b
  python table_attention.py bench --model pocket0.3b
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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from detlayers.common import calibration_windows, load_model, pick_device, seed_all, wikitext_windows  # noqa: E402
from detlayers.methods import Replacement  # noqa: E402
from detlayers.metrics import compare_lm  # noqa: E402


class IdTap:
    ids = None


def attach_tap(model):
    tap = IdTap()
    model.model.embed_tokens.register_forward_pre_hook(lambda m, inp: setattr(tap, "ids", inp[0]))
    return tap


def dims(model):
    c = model.config
    return c.num_attention_heads, c.num_key_value_heads, c.head_dim


@torch.no_grad()
def project(att, h):
    """q, k (после QK-norm, до RoPE) и v из нормализованного входа слоя h (..., d)."""
    H, G, D = att.config.num_attention_heads, att.config.num_key_value_heads, att.head_dim
    q = att.q_norm(att.q_proj(h).view(*h.shape[:-1], H, D))
    k = att.k_norm(att.k_proj(h).view(*h.shape[:-1], G, D))
    v = att.v_proj(h).view(*h.shape[:-1], G, D)
    return q, k, v


@torch.no_grad()
def layer0_tables(model, chunk=8192):
    """Точные таблицы слоя 0 для всего словаря: q (V,H,D), k (V,G,D), v (V,G,D) — fp32 на CPU."""
    layer = model.model.layers[0]
    E = model.model.embed_tokens.weight
    qs, ks, vs = [], [], []
    for s in range(0, E.shape[0], chunk):
        q, k, v = project(layer.self_attn, layer.input_layernorm(E[s:s + chunk]))  # в dtype модели
        qs.append(q.float().cpu()), ks.append(k.float().cpu()), vs.append(v.float().cpu())
    return torch.cat(qs), torch.cat(ks), torch.cat(vs)


def to_polar(t):
    D = t.shape[-1]
    z = torch.complex(t[..., : D // 2].float(), t[..., D // 2:].float())
    return z.abs(), z.angle()


def from_polar(a, p):
    return torch.cat([a * torch.cos(p), a * torch.sin(p)], -1)


def quantize_polar(t, bits):
    """Квантование таблицы q или k в полярной форме: амплитуды — масштаб на (голову, частоту), фазы — по окружности."""
    a, p = to_polar(t)
    L = 2 ** bits - 1
    sc = a.amax(dim=0, keepdim=True).clamp_min(1e-8)
    a = torch.round(a / sc * L) / L * sc
    p = torch.round((p + math.pi) / (2 * math.pi) * (L + 1)) % (L + 1) / (L + 1) * 2 * math.pi - math.pi
    return from_polar(a, p)


def quantize_sym(t, bits):
    sc = t.abs().amax(dim=0, keepdim=True).clamp_min(1e-8)
    h = 2 ** (bits - 1) - 1
    return torch.round(t / sc * h) / h * sc


class TableAttn(nn.Module):
    """Внимание из таблиц на токен. tabs: q (R,H,D), k (R,G,D), v (R,G,D); vocab_map: id → строка (None — id = строка).
    Позиции t = 0..T−1. slow — частоты без поворота (не зависят от Δ); keep (H, D/2) — оставленные частоты;
    window — int или список на голову (None — без окна); exact_pos0 — позиция 0 из исходных проекций;
    v_exact — v из исходной проекции (только картина внимания из таблиц)."""

    def __init__(self, orig, q, k, v, tap, theta, vocab_map=None, slow=None, keep=None, window=None, n_sink=1,
                 exact_pos0=False, v_exact=False, fallback_exact=False):
        super().__init__()
        self.fallback_exact = fallback_exact  # токены вне таблицы (последняя строка) считаются точно
        self.orig, self.tap = orig, tap
        dev = orig.o_proj.weight.device
        self.H, self.G, self.D = q.shape[1], k.shape[1], q.shape[2]
        for name, t in (("q", q), ("k", k), ("v", v)):
            self.register_buffer(name, t.float().to(dev))
        self.register_buffer("vmap", vocab_map.to(dev) if vocab_map is not None else torch.zeros(0, dtype=torch.long))
        self.register_buffer("theta", theta.float().to(dev))
        M = self.D // 2
        self.register_buffer("slow", (slow if slow is not None else torch.zeros(M, dtype=torch.bool)).to(dev))
        self.register_buffer("keep", (keep if keep is not None else torch.ones(self.H, M, dtype=torch.bool)).to(dev))
        self.window, self.n_sink, self.exact_pos0, self.v_exact = window, n_sink, exact_pos0, v_exact

    def rot(self, t, T):
        """RoPE в полярной форме: поворот частоты m на θ_m·t, кроме медленных."""
        M = self.D // 2
        ang = torch.arange(T, device=t.device, dtype=torch.float32)[:, None] * self.theta[None]
        ang = torch.where(self.slow[None], torch.zeros_like(ang), ang)  # (T,M)
        c, s = torch.cos(ang)[None, :, None], torch.sin(ang)[None, :, None]
        a, b = t[..., :M], t[..., M:]
        return torch.cat([a * c - b * s, a * s + b * c], -1)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, **kw):
        ids = self.tap.ids.to(hidden_states.device)
        rows = self.vmap[ids] if self.vmap.numel() else ids
        B, T = ids.shape
        rep = self.H // self.G
        q, k, v = self.q[rows], self.k[rows], self.v[rows]  # (B,T,·,D)
        miss = (rows == self.q.shape[0] - 1) if (self.fallback_exact and self.vmap.numel()) else None
        if self.exact_pos0 or self.v_exact or miss is not None:
            qe, ke, ve = project(self.orig, hidden_states.float())
            if miss is not None:
                mq = miss[..., None, None]
                q, k, v = torch.where(mq, qe.float(), q), torch.where(mq, ke.float(), k), torch.where(mq, ve.float(), v)
            if self.exact_pos0:
                q, k, v = q.clone(), k.clone(), v.clone()
                q[:, 0], k[:, 0], v[:, 0] = qe[:, 0].float(), ke[:, 0].float(), ve[:, 0].float()
            if self.v_exact:
                v = ve.float()
        q = q * self.keep.repeat(1, 2)[None, None]  # выброшенные частоты обнуляются в запросе
        q = self.rot(q, T).transpose(1, 2)
        k = self.rot(k, T).repeat_interleave(rep, dim=2).transpose(1, 2)
        v = v.repeat_interleave(rep, dim=2).transpose(1, 2)
        s = (q @ k.transpose(-1, -2)) * self.orig.scaling
        i = torch.arange(T, device=s.device)
        mask = (i[:, None] >= i[None, :])[None]
        if self.window is not None:
            ws = self.window if isinstance(self.window, (list, tuple)) else [self.window] * self.H
            wt = torch.tensor([w if w else 10 ** 9 for w in ws], device=s.device)
            mask = mask & (((i[:, None] - i[None, :])[None] < wt[:, None, None]) | (i[None, None, :] < self.n_sink))
        p = torch.softmax(s.masked_fill(~mask[None], float("-inf")), -1)
        o = (p @ v).transpose(1, 2).reshape(B, T, -1)
        return self.orig.o_proj(o.to(hidden_states.dtype)), None


def evaluate(model, layer_mods, wins, device):
    rep = Replacement(model)
    for i, m in layer_mods.items():
        rep.add(i, "self_attn", m)
    r = compare_lm(model, rep, wins, device, bs=1)
    rep.disable()
    return {"kl": r["kl"], "top1": r["top1_agree"], "dppl_pct": 100 * (r["ppl"] / r["ppl_teacher"] - 1)}


# ================================================================ layer0

@torch.no_grad()
def layer0(args):
    device = pick_device(None)
    model, tok = load_model(args.model, device, "fp32")
    H, G, D = dims(model)
    M = D // 2
    theta = model.model.rotary_emb.inv_freq.float().cpu()
    period = (2 * math.pi / theta)
    ctx = model.config.max_position_embeddings
    t0 = time.time()
    q, k, v = layer0_tables(model)
    V = q.shape[0]
    res = {"model": args.model, "vocab": V, "heads_q": H, "heads_kv": G, "head_dim": D, "rope_base": float(1 / theta[1] ** (D / 2)),
           "context": ctx, "table_mb_fp16": round((q.numel() + k.numel() + v.numel()) * 2 / 2 ** 20, 1),
           "attn_weights_mb_fp16": round(sum(p.numel() for p in model.model.layers[0].self_attn.parameters()) * 2 / 2 ** 20, 2),
           "period_tokens": period.tolist()}
    # --- проверка формулы против проекций модели на тексте
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    wins = wikitext_windows(tok, "wikitext-2-raw-v1", "test", 1024, max_windows=args.eval_windows)
    layer = model.model.layers[0]
    err = 0.0
    mass = {W: torch.zeros(H) for W in (16, 32, 64, 128, 256)}
    pos0 = torch.zeros(H)
    nwin = min(8, wins.shape[0])
    tap = attach_tap(model)
    tmod = TableAttn(layer.self_attn, q, k, v, tap, theta)
    for s in range(nwin):
        ids = wins[s:s + 1].to(device)
        x = model.model.embed_tokens(ids)
        cos, sin = model.model.rotary_emb(x, torch.arange(ids.shape[1], device=device)[None])
        qq, kk, _ = project(layer.self_attn, layer.input_layernorm(x))
        qq, kk = apply_rotary_pos_emb(qq.transpose(1, 2), kk.transpose(1, 2), cos, sin)
        L = (qq @ kk.repeat_interleave(H // G, dim=1).transpose(-1, -2))[0] * layer.self_attn.scaling
        T = ids.shape[1]
        qt = tmod.rot(q[ids.cpu()].to(device), T).transpose(1, 2)
        kt = tmod.rot(k[ids.cpu()].to(device), T).repeat_interleave(H // G, dim=2).transpose(1, 2)
        A = (qt @ kt.transpose(-1, -2))[0] * layer.self_attn.scaling
        i = torch.arange(T, device=device)
        causal = i[:, None] >= i[None, :]
        err = max(err, (A - L)[:, causal].abs().max().item())
        p = torch.softmax(L.masked_fill(~causal, float("-inf")), -1)
        dist = i[:, None] - i[None, :]
        pos0 += p[:, 1:, 0].mean(-1).cpu()
        for W in mass:
            mass[W] += (p * ((dist >= W) & (i[None, :] > 0))).sum(-1)[:, W:].mean(-1).cpu()
    res["check_max_abs_logit_err"] = err
    res["mass_outside_window_by_head"] = {W: (m / nwin).tolist() for W, m in mass.items()}
    res["attn_to_pos0_by_head"] = (pos0 / nwin).tolist()
    # --- частоты
    qa, _ = to_polar(q)
    ka, _ = to_polar(k)
    share = qa.mean(0) * ka.mean(0).repeat_interleave(H // G, dim=0)  # (H,M)
    share = share / share.sum(-1, keepdim=True)
    order = share.argsort(-1, descending=True)
    res["freq_share_by_head"] = share.tolist()
    res["top_freq_by_head"] = order[:, 0].tolist()
    res["top_freq_share_by_head"] = share.gather(1, order[:, :1]).squeeze(1).tolist()
    res["n_freq_90pct_by_head"] = [int((share[h][order[h]].cumsum(0) < 0.9).sum()) + 1 for h in range(H)]
    slow = period > ctx
    res["slow_freqs"] = slow.nonzero().flatten().tolist()
    print(f"[{args.model}] формула: макс. ошибка логита {err:.1e}; главная частота по головам {res['top_freq_by_head']}"
          f" доли {[round(x, 2) for x in res['top_freq_share_by_head']]}", flush=True)
    m64 = res["mass_outside_window_by_head"][64]
    print(f"[{args.model}] масса дальше 64 по головам: {[round(x, 2) for x in m64]}; на поз.0: "
          f"{[round(x, 3) for x in res['attn_to_pos0_by_head']]}", flush=True)

    def keepK(K):
        m = torch.zeros(H, M, dtype=torch.bool)
        m.scatter_(1, order[:, :K], True)
        return m

    local = [64 if m < 0.25 else None for m in m64]
    q8, k8, v8 = quantize_polar(q, 8), quantize_polar(k, 8), quantize_sym(v, 8)
    q4, k4, v4 = quantize_polar(q, 4), quantize_polar(k, 4), quantize_sym(v, 4)
    var = {
        "exact": (q, k, v, {}),
        "fp16": (q.half().float(), k.half().float(), v.half().float(), {}),
        "int8": (q8, k8, v8, {}),
        "int4": (q4, k4, v4, {}),
        "slow_no_delta": (q, k, v, {"slow": slow}),
        f"top{int(M * 0.75)}freq": (q, k, v, {"keep": keepK(int(M * 0.75))}),
        f"top{M // 2}freq": (q, k, v, {"keep": keepK(M // 2)}),
        "win64_all": (q, k, v, {"window": 64}),
        "win64_local": (q, k, v, {"window": local}),
        "win64_local+int8": (q8, k8, v8, {"window": local}),
    }
    res["local_heads"] = [h for h, w in enumerate(local) if w]
    res["variants"] = {}
    for name, (qq_, kk_, vv_, kw) in var.items():
        mod = TableAttn(layer.self_attn, qq_, kk_, vv_, tap, theta, **kw)
        r = evaluate(model, {0: mod}, wins, device)
        res["variants"][name] = r
        print(f"[{args.model}] {name:18s} KL {r['kl']:.2e} top1 {r['top1']:.4f}", flush=True)
        del mod
    res["seconds"] = round(time.time() - t0, 1)
    out = ROOT / "results" / args.model / "table_attn"
    out.mkdir(parents=True, exist_ok=True)
    (out / "layer0.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("сохранено", out / "layer0.json")


# ================================================================ alllayers

@torch.no_grad()
def collect_means(model, calib, layers, device, bs, min_count):
    """Средние q, k, v токена (позиции ≥ 1) по калибровке для заданных слоёв + R² токена по q, k, v."""
    H, G, D = dims(model)
    ids_all = calib[:, 1:].reshape(-1)
    uniq, cnt = ids_all.unique(return_counts=True)
    seen = uniq[cnt >= min_count]
    V = model.config.vocab_size
    vmap = torch.full((V,), len(seen), dtype=torch.long)  # последняя строка — общее среднее (редкие токены)
    vmap[seen] = torch.arange(len(seen))
    R = len(seen) + 1
    acc = {l: {"q": torch.zeros(R, H, D), "k": torch.zeros(R, G, D), "v": torch.zeros(R, G, D),
               "q2": torch.zeros(R), "k2": torch.zeros(R), "v2": torch.zeros(R), "n": torch.zeros(R)} for l in layers}
    store = {}

    def make_hook(l):
        def hook(mod, args, kwargs):
            store[l] = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        return hook

    hooks = [model.model.layers[l].self_attn.register_forward_pre_hook(make_hook(l), with_kwargs=True) for l in layers]
    try:
        for s in range(0, calib.shape[0], bs):
            ids = calib[s:s + bs].to(device)
            store.clear()
            model(ids)
            rows = vmap[ids[:, 1:].reshape(-1).cpu()]
            for l in layers:
                q, k, v = project(model.model.layers[l].self_attn, store[l][:, 1:].float())
                a = acc[l]
                for name, t in (("q", q), ("k", k), ("v", v)):
                    t = t.reshape(-1, *t.shape[-2:]).float().cpu()
                    a[name].index_add_(0, rows, t)
                    a[name + "2"].index_add_(0, rows, t.pow(2).sum((-1, -2)))
                a["n"].index_add_(0, rows, torch.ones_like(rows, dtype=torch.float))
    finally:
        for h in hooks:
            h.remove()
    tabs, r2 = {}, {}
    for l in layers:
        a = acc[l]
        n = a["n"].clamp_min(1)
        r2[l] = {}
        for name in ("q", "k", "v"):
            S, S2 = a[name], a[name + "2"]
            ok = torch.arange(R) < R - 1  # только токены с ≥ min_count вхождений
            N = a["n"][ok].sum()
            tot = S2[ok].sum() - (S[ok].sum(0).pow(2).sum()) / N
            within = S2[ok].sum() - (S[ok].pow(2).sum((-1, -2)) / n[ok]).sum()
            r2[l][name] = float(1 - within / tot)
        mean = {name: a[name] / n[:, None, None] for name in ("q", "k", "v")}
        allm = {name: a[name][:-1].sum(0) / a["n"][:-1].sum() for name in ("q", "k", "v")}
        for name in ("q", "k", "v"):
            mean[name][-1] = allm[name]
        tabs[l] = mean
    return tabs, vmap, r2, len(seen)


@torch.no_grad()
def alllayers(args):
    seed_all(0)
    device = pick_device(None)
    model, tok = load_model(args.model, device, "fp32")
    L = model.config.num_hidden_layers
    theta = model.model.rotary_emb.inv_freq.float().cpu()
    tap = attach_tap(model)
    calib = calibration_windows(tok, args.calib_tokens, 512, seed=0)
    wins = wikitext_windows(tok, "wikitext-2-raw-v1", "test", 1024, max_windows=args.eval_windows)
    t0 = time.time()
    groups = [list(range(s, min(s + args.group, L))) for s in range(0, L, args.group)]
    res = {"model": args.model, "calib_tokens": int(calib.numel()), "eval_windows": int(wins.shape[0]), "layers": {},
           "prefix": {}}
    prefix_tabs = {}
    for gi, grp in enumerate(groups):
        tabs, vmap, r2, nseen = collect_means(model, calib, grp, device, args.bs, args.min_count)
        if gi == 0:
            ev = wins.reshape(-1)
            res["vocab_rows"] = nseen
            res["eval_token_coverage"] = float((vmap[ev] < nseen).float().mean())
            print(f"[{args.model}] токенов в таблице {nseen}; покрытие оценочных токенов {res['eval_token_coverage']:.3f}",
                  flush=True)
        if 0 in grp:  # слой 0: таблица точная для всего словаря
            q0, k0, v0 = layer0_tables(model)
            tabs[0] = {"q": q0, "k": k0, "v": v0, "full_vocab": True}
        for l in grp:
            att = model.model.layers[l].self_attn
            row = {"r2": r2[l]}
            vm = None if tabs[l].get("full_vocab") else vmap
            for mode in ("qkv", "qk"):
                mod = TableAttn(att, tabs[l]["q"], tabs[l]["k"], tabs[l]["v"], tap, theta, vocab_map=vm,
                                exact_pos0=True, v_exact=(mode == "qk"), fallback_exact=vm is not None)
                row[mode] = evaluate(model, {l: mod}, wins, device)
                del mod
            res["layers"][l] = row
            print(f"[{args.model}] слой {l:2d}: R² q {r2[l]['q']:.3f} k {r2[l]['k']:.3f} v {r2[l]['v']:.3f} | "
                  f"таблица q,k,v KL {row['qkv']['kl']:.4f} top1 {row['qkv']['top1']:.3f} | "
                  f"таблица q,k KL {row['qk']['kl']:.4f}", flush=True)
            if l < args.prefix_max:
                prefix_tabs[l] = (tabs[l], vm)
        if gi == 0:  # префиксы: первые k слоёв таблицами одновременно
            for kk in range(1, min(args.prefix_max, len(grp)) + 1):
                mods = {l: TableAttn(model.model.layers[l].self_attn, prefix_tabs[l][0]["q"], prefix_tabs[l][0]["k"],
                                     prefix_tabs[l][0]["v"], tap, theta, vocab_map=prefix_tabs[l][1], exact_pos0=True,
                                     fallback_exact=prefix_tabs[l][1] is not None)
                        for l in range(kk)}
                res["prefix"][kk] = evaluate(model, mods, wins, device)
                print(f"[{args.model}] первые {kk} слоёв таблицами: KL {res['prefix'][kk]['kl']:.4f} "
                      f"top1 {res['prefix'][kk]['top1']:.3f}", flush=True)
                del mods
        del tabs
        prefix_tabs = {l: v for l, v in prefix_tabs.items() if gi == 0}
    res["seconds"] = round(time.time() - t0, 1)
    out = ROOT / "results" / args.model / "table_attn"
    out.mkdir(parents=True, exist_ok=True)
    (out / "alllayers.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("сохранено", out / "alllayers.json")


# ================================================================ bench

class GatherAttn0(nn.Module):
    """Эффективная табличная версия слоя 0: q,k,v (до RoPE) берутся из таблиц, дальше обычные RoPE + SDPA + W_o."""

    def __init__(self, orig, q, k, v, tap, dtype):
        super().__init__()
        self.orig, self.tap = orig, tap
        dev = orig.o_proj.weight.device
        self.register_buffer("q", q.to(dtype).to(dev))
        self.register_buffer("k", k.to(dtype).to(dev))
        self.register_buffer("v", v.to(dtype).to(dev))

    def forward(self, hidden_states, position_embeddings, attention_mask=None, **kw):
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
        ids = self.tap.ids
        q = self.q[ids].transpose(1, 2).to(hidden_states.dtype)
        k = self.k[ids].transpose(1, 2).to(hidden_states.dtype)
        v = self.v[ids].transpose(1, 2).to(hidden_states.dtype)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True, scale=self.orig.scaling)
        o = o.transpose(1, 2).reshape(*hidden_states.shape[:-1], -1)
        return self.orig.o_proj(o), None


@torch.no_grad()
def bench(args):
    import statistics
    res = {"model": args.model, "runs": {}}
    for dev_name in args.devices.split(","):
        device = pick_device(dev_name)
        for dt_name, dt in (("fp32", torch.float32), ("fp16", torch.float16 if dev_name == "mps" else torch.bfloat16)):
            model, tok = load_model(args.model, device, dt)
            q, k, v = layer0_tables(model)
            tap = attach_tap(model)
            layer = model.model.layers[0]
            orig = layer.self_attn
            gat = GatherAttn0(orig, q, k, v, tap, dt)

            def sync():
                if device.type == "mps":
                    torch.mps.synchronize()

            for T in (128, 512, 2048):
                ids = torch.randint(0, model.config.vocab_size, (1, T), device=device)
                x = layer.input_layernorm(model.model.embed_tokens(ids))
                pe = model.model.rotary_emb(x, torch.arange(T, device=device)[None])
                tap.ids = ids

                def run_orig():
                    return orig(hidden_states=x, position_embeddings=pe, attention_mask=None)

                def run_tab():
                    return gat(x, pe)

                def full(mod_swap):
                    if mod_swap:
                        layer.self_attn = gat
                    out = model(ids)
                    layer.self_attn = orig
                    return out

                times = {}
                for name, fn in (("attn0_orig", run_orig), ("attn0_table", run_tab),
                                 ("model_orig", lambda: full(False)), ("model_table", lambda: full(True))):
                    for _ in range(3):
                        fn()
                    sync()
                    ts = []
                    for _ in range(args.reps):
                        t = time.perf_counter()
                        fn()
                        sync()
                        ts.append(time.perf_counter() - t)
                    times[name] = statistics.median(ts) * 1e3
                key = f"{dev_name}_{dt_name}_T{T}"
                res["runs"][key] = times
                print(f"[{args.model}] {key}: слой 0 {times['attn0_orig']:.2f} → {times['attn0_table']:.2f} мс "
                      f"(x{times['attn0_orig'] / times['attn0_table']:.2f}); модель {times['model_orig']:.1f} → "
                      f"{times['model_table']:.1f} мс (x{times['model_orig'] / times['model_table']:.3f})", flush=True)
            del model, gat
            if device.type == "mps":
                torch.mps.empty_cache()
    out = ROOT / "results" / args.model / "table_attn"
    out.mkdir(parents=True, exist_ok=True)
    (out / "bench.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("сохранено", out / "bench.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["layer0", "alllayers", "bench"])
    ap.add_argument("--model", default="pocket0.3b")
    ap.add_argument("--eval-windows", type=int, default=16)
    ap.add_argument("--calib-tokens", type=int, default=262_144)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--group", type=int, default=7)
    ap.add_argument("--prefix-max", type=int, default=7)
    ap.add_argument("--devices", default="mps,cpu")
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()
    {"layer0": layer0, "alllayers": alllayers, "bench": bench}[args.mode](args)


if __name__ == "__main__":
    main()
