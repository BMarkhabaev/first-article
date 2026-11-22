"""Формула с доски для слоя 0 MNN-Pocket-0.3B (Qwen3: d=896, 14 голов Q, 2 головы KV, head_dim 64, RoPE θ=1e6).

В слое 0 вход внимания — эмбеддинг токена, поэтому q и k зависят только от id токена:
    q_h(x) = q_norm(W_q^h · RMSNorm(E[x])),   k_g(y) = k_norm(W_k^g · RMSNorm(E[y])).
Пары RoPE в реализации HF — (m, m+32): z_m = c[m] + i·c[m+32], частота θ_m = θ^(-2m/64).
Логит головы h между запросом x на позиции t и ключом y на позиции t-Δ:
    A_h(x, y, Δ) = s · Σ_{m=0}^{31} R_m cos(θ_m Δ + φ_m),
    R_m = |z^q_m(x)|·|z^k_m(y)|,  φ_m = arg z^q_m(x) − arg z^k_m(y),  s = 1/√64.
Скрипт считает |z| и arg z для всех 65 536 токенов (q: 14 голов, k: 2 головы) и v = W_v·RMSNorm(E[x]),
проверяет формулу против логитов и вероятностей внимания самой модели на WikiText-2 и сохраняет таблицы.

  python mnn_layer0_rope.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from detlayers.common import wikitext_windows  # noqa: E402

MODEL_DIR = ROOT / "data" / "models" / "MNN-Pocket-0.3B"
OUT = ROOT / "results" / "mnn_pocket"


@torch.no_grad()
def token_tables(model):
    """|z|, arg z для q (V,H,32) и k (V,G,32); v (V,G,64) — для всех токенов словаря."""
    layer = model.model.layers[0]
    att = layer.self_attn
    cfg = model.config
    H, G, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    E = model.model.embed_tokens.weight.float()
    V = E.shape[0]
    zq_abs = torch.empty(V, H, D // 2)
    zq_arg = torch.empty(V, H, D // 2)
    zk_abs = torch.empty(V, G, D // 2)
    zk_arg = torch.empty(V, G, D // 2)
    v_all = torch.empty(V, G, D)
    for s in range(0, V, 8192):
        x = layer.input_layernorm(E[s:s + 8192])
        q = att.q_norm(att.q_proj(x).view(-1, H, D))
        k = att.k_norm(att.k_proj(x).view(-1, G, D))
        v = att.v_proj(x).view(-1, G, D)
        zq = torch.complex(q[..., : D // 2], q[..., D // 2:])
        zk = torch.complex(k[..., : D // 2], k[..., D // 2:])
        zq_abs[s:s + 8192], zq_arg[s:s + 8192] = zq.abs(), zq.angle()
        zk_abs[s:s + 8192], zk_arg[s:s + 8192] = zk.abs(), zk.angle()
        v_all[s:s + 8192] = v
    return zq_abs, zq_arg, zk_abs, zk_arg, v_all


def formula_logits(ids, tabs, theta, scaling, n_rep):
    """Логиты слоя 0 по формуле с доски для последовательности ids: (H, T, T), без маски."""
    zq_abs, zq_arg, zk_abs, zk_arg, _ = tabs
    T = ids.numel()
    Rq, Pq = zq_abs[ids], zq_arg[ids]  # (T,H,32)
    Rk = zk_abs[ids].repeat_interleave(n_rep, dim=1)  # (T,H,32)
    Pk = zk_arg[ids].repeat_interleave(n_rep, dim=1)
    pos = torch.arange(T, dtype=torch.float64)
    delta = (pos[:, None] - pos[None, :])  # (T,T): Δ = t_q − t_k
    out = torch.zeros(Rq.shape[1], T, T, dtype=torch.float64)
    for m in range(theta.numel()):
        R = Rq[:, :, m].T.double()[:, :, None] * Rk[:, :, m].T.double()[:, None, :]  # (H,T,T)
        phi = Pq[:, :, m].T.double()[:, :, None] - Pk[:, :, m].T.double()[:, None, :]
        out += R * torch.cos(theta[m].double() * delta[None] + phi)
    return out * scaling


@torch.no_grad()
def model_logits_and_probs(model, ids):
    """Логиты (до маски) и вероятности внимания слоя 0 из самой модели (eager)."""
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    layer = model.model.layers[0]
    att = layer.self_attn
    cfg = model.config
    H, G, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    x = model.model.embed_tokens(ids[None])
    pos_ids = torch.arange(ids.numel())[None]
    cos, sin = model.model.rotary_emb(x, pos_ids)
    xn = layer.input_layernorm(x)
    q = att.q_norm(att.q_proj(xn).view(1, -1, H, D)).transpose(1, 2)
    k = att.k_norm(att.k_proj(xn).view(1, -1, G, D)).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    k = k.repeat_interleave(H // G, dim=1)
    logits = (q @ k.transpose(-1, -2))[0].double() * att.scaling
    out = model(ids[None], output_attentions=True)
    probs = out.attentions[0][0].double()  # (H,T,T)
    return logits, probs


def main():
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32, attn_implementation="eager").eval()
    cfg = model.config
    H, G, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    inv_freq = model.model.rotary_emb.inv_freq.float()  # θ_m = base^(-2m/D)
    theta = inv_freq
    scaling = model.model.layers[0].self_attn.scaling
    assert theta.numel() == D // 2
    tabs = token_tables(model)
    zq_abs, zq_arg, zk_abs, zk_arg, v_all = tabs
    print(f"таблицы для {zq_abs.shape[0]} токенов: {time.time() - t0:.0f}s", flush=True)

    # --- проверка формулы на тексте
    wins = wikitext_windows(tok, "wikitext-2-raw-v1", "test", 512, max_windows=4)
    err_logit, err_prob, sink0 = [], [], []
    for w in wins:
        ids = w.clone()
        A = formula_logits(ids, tabs, theta, scaling, H // G)
        L, P = model_logits_and_probs(model, ids)
        T = ids.numel()
        causal = torch.ones(T, T, dtype=torch.bool).tril()
        err_logit.append((A - L)[:, causal].abs().max().item())
        Pf = torch.softmax(A.masked_fill(~causal, float("-inf")), dim=-1)
        err_prob.append((Pf - P).abs().max().item())
        sink0.append(P[:, 1:, 0].mean(dim=-1))  # средняя доля внимания на позицию 0, по головам
    sink0 = torch.stack(sink0).mean(0)
    print(f"макс. |логит формулы − логит модели| = {max(err_logit):.2e}; "
          f"макс. |p формулы − p модели| = {max(err_prob):.2e}", flush=True)

    # --- сводка по частотам
    period = (2 * math.pi / theta).tolist()
    ctx = cfg.max_position_embeddings
    q_amp = zq_abs.mean(dim=(0, 1))  # (32,) средняя |z^q_m| по токенам и головам
    k_amp = zk_abs.mean(dim=(0, 1))
    contrib = (q_amp * k_amp)
    contrib = (contrib / contrib.sum()).tolist()
    slow = [m for m in range(D // 2) if period[m] > ctx]  # почти не зависят от Δ в пределах контекста
    # --- нормы значений v (на доске: v_1 = W_v x_1 ≈ 0 для стока)
    vn = v_all.norm(dim=-1)  # (V,G)
    special = {t: tok.convert_tokens_to_ids(t) for t in tok.all_special_tokens}
    special_v = {t: [round(float(vn[i, g]), 3) for g in range(G)] for t, i in special.items() if i is not None and i < vn.shape[0]}
    small = vn.mean(1).argsort()[:15].tolist()
    res = {
        "model": "taobao-mnn/MNN-Pocket-0.3B", "layer": 0,
        "config": {"d": cfg.hidden_size, "heads_q": H, "heads_kv": G, "head_dim": D, "d_mlp": cfg.intermediate_size,
                   "vocab": cfg.vocab_size, "rope_theta": cfg.rope_parameters.get("rope_theta") if hasattr(cfg, "rope_parameters") and cfg.rope_parameters else getattr(cfg, "rope_theta", None),
                   "max_pos": ctx, "scaling": scaling},
        "rope_pairing": "HF rotate_half: (m, m+32)",
        "theta": theta.tolist(), "period_tokens": period,
        "slow_freqs_period_gt_context": slow,
        "share_of_mean_amplitude_product_by_freq": contrib,
        "check": {"windows": int(wins.shape[0]), "len": int(wins.shape[1]),
                  "max_abs_logit_err": max(err_logit), "max_abs_prob_err": max(err_prob)},
        "attn_to_pos0_by_head": sink0.tolist(),
        "v_norm": {"median": float(vn.median()), "p01": float(vn.quantile(0.01)), "special_tokens": special_v,
                   "smallest_tokens": [(tok.convert_ids_to_tokens(i), round(float(vn[i].mean()), 3)) for i in small]},
        "seconds": round(time.time() - t0, 1),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "layer0_summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    np.savez_compressed(OUT / "layer0_rope_tables.npz",
                        zq_abs=zq_abs.half().numpy(), zq_arg=zq_arg.half().numpy(),
                        zk_abs=zk_abs.half().numpy(), zk_arg=zk_arg.half().numpy(),
                        v=v_all.half().numpy(), theta=theta.numpy(), scaling=np.float32(scaling))
    print(json.dumps({k: res[k] for k in ("config", "check", "slow_freqs_period_gt_context")}, ensure_ascii=False))
    print("периоды (токены): m=0 →", round(period[0], 2), "; m=31 →", f"{period[-1]:.3g}")
    print("доля средней амплитуды по частотам (топ-5):",
          sorted(range(D // 2), key=lambda m: -contrib[m])[:5], [round(c, 3) for c in sorted(contrib, reverse=True)[:5]])
    print("внимание на позицию 0 по головам:", [round(x, 2) for x in sink0.tolist()])
    print("‖v‖: медиана", round(res["v_norm"]["median"], 3), "; спецтокены", special_v)
    print("наименьшие ‖v‖:", res["v_norm"]["smallest_tokens"][:8])
    print("сохранено", OUT)


if __name__ == "__main__":
    main()
