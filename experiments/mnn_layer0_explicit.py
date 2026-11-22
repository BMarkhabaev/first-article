"""Явная замена внимания слоя 0 MNN-Pocket-0.3B по формуле с доски.

Во время работы не используются W_q, W_k, W_v, RMSNorm и QK-norm: только таблицы на токен словаря
(амплитуды и фазы 32 частот RoPE для 14 голов запросов и 2 голов ключей, вектор v для 2 голов) и матрица W_o.
    A_h(x_i, x_j) = s · Σ_m R^q_{h,m}(x_i)·R^k_{g,m}(x_j)·cos(θ_m (i−j) + φ^q_{h,m}(x_i) − φ^k_{g,m}(x_j))
    p_h = softmax_j≤i(A_h),   out = W_o · concat_h Σ_j p_h(i,j) · v_g(x_j).
Через тождество cos(a−b) = cos a cos b + sin a sin b сумма по m считается как скалярное произведение двух
64-мерных векторов, построенных из таблиц, — это та же формула, без приближений.

Варианты (сравниваются с исходной моделью по KL/PPL на WikiText-2):
  exact      — таблицы fp32;
  fp16       — таблицы fp16 (как в layer0_rope_tables.npz);
  topK       — в каждой голове только K частот с наибольшей средней амплитудой (K = 24, 16, 8);
  slowconst  — частоты с периодом больше контекста модели (m ≥ 21) считаются без сдвига: cos(φ) вместо
               cos(θΔ + φ), то есть они дают поправку, зависящую только от пары токенов.

  python mnn_layer0_explicit.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from detlayers.common import pick_device, wikitext_windows  # noqa: E402
from detlayers.methods import Replacement  # noqa: E402
from detlayers.metrics import compare_lm  # noqa: E402

MODEL_DIR = ROOT / "data" / "models" / "MNN-Pocket-0.3B"
OUT = ROOT / "results" / "mnn_pocket"


class IdTap:
    ids = None


@torch.no_grad()
def build_tables(model):
    layer = model.model.layers[0]
    att = layer.self_attn
    cfg = model.config
    H, G, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    E = model.model.embed_tokens.weight.float()
    x = layer.input_layernorm(E)
    q = att.q_norm(att.q_proj(x).view(-1, H, D)).float()
    k = att.k_norm(att.k_proj(x).view(-1, G, D)).float()
    v = att.v_proj(x).view(-1, G, D).float()
    zq = torch.complex(q[..., : D // 2], q[..., D // 2:])
    zk = torch.complex(k[..., : D // 2], k[..., D // 2:])
    return {"aq": zq.abs(), "pq": zq.angle(), "ak": zk.abs(), "pk": zk.angle(), "v": v}


class ExplicitAttn0(nn.Module):
    """Внимание слоя 0 из таблиц на токен (см. докстринг модуля)."""

    def __init__(self, orig, tabs, tap, theta, keep=None, slow=None, dtype=torch.float32, window=None, n_sink=0):
        super().__init__()
        self.window, self.n_sink = window, n_sink  # окно по расстоянию (None — полный контекст) и число стоков
        self.tap, self.o_proj, self.scaling = tap, orig.o_proj, orig.scaling
        self.H, self.G = tabs["aq"].shape[1], tabs["ak"].shape[1]
        dev = orig.o_proj.weight.device
        for name in ("aq", "pq", "ak", "pk", "v"):  # таблицы хранятся в dtype (fp32/fp16), считаются в fp32
            self.register_buffer(name, tabs[name].to(dtype).to(dev))
        self.register_buffer("theta", theta.float().to(dev))
        M = theta.numel()
        keep = keep if keep is not None else torch.ones(self.H, M, dtype=torch.bool)
        self.register_buffer("keep", keep.to(dev))  # (H, M) какие частоты оставить в голове
        self.register_buffer("slow", (slow if slow is not None else torch.zeros(M, dtype=torch.bool)).to(dev))

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, **kw):
        ids = self.tap.ids.to(hidden_states.device)
        B, T = ids.shape
        rep = self.H // self.G
        pos = torch.arange(T, device=ids.device, dtype=torch.float32)
        ang = pos[:, None] * self.theta[None, :]  # (T, M): θ_m · t
        ang = torch.where(self.slow[None, :], torch.zeros_like(ang), ang)  # медленные частоты — без сдвига
        aq = self.aq[ids].float() * self.keep[None, None].float()  # (B,T,H,M)
        pq = self.pq[ids].float() + ang[None, :, None, :]
        ak = self.ak[ids].float().repeat_interleave(rep, dim=2)  # (B,T,H,M)
        pk = self.pk[ids].float().repeat_interleave(rep, dim=2) + ang[None, :, None, :]
        qv = torch.cat([aq * torch.cos(pq), aq * torch.sin(pq)], -1).transpose(1, 2)  # (B,H,T,2M)
        kv = torch.cat([ak * torch.cos(pk), ak * torch.sin(pk)], -1).transpose(1, 2)
        s = (qv @ kv.transpose(-1, -2)) * self.scaling  # Σ_m R cos(θΔ + φ)
        mask = torch.ones(T, T, dtype=torch.bool, device=s.device).tril_()
        if self.window is not None:  # число — одно окно на все головы; список — окно на голову (None — без окна)
            i = torch.arange(T, device=s.device)
            dist = (i[:, None] - i[None, :])
            ws = self.window if isinstance(self.window, (list, tuple)) else [self.window] * self.H
            wt = torch.tensor([w if w else 10 ** 9 for w in ws], device=s.device)
            mask = mask & (((dist[None] < wt[:, None, None]) | (i[None, None, :] < self.n_sink)))  # (H,T,T)
        p = torch.softmax(s.masked_fill(~mask, float("-inf")), dim=-1)
        v = self.v[ids].float().repeat_interleave(rep, dim=2).transpose(1, 2)  # (B,H,T,D)
        o = (p @ v).transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(o.to(hidden_states.dtype)), None


def main():
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = pick_device(None)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32).to(device).eval()
    cfg = model.config
    H, D = cfg.num_attention_heads, cfg.head_dim
    M = D // 2
    theta = model.model.rotary_emb.inv_freq.float().cpu()
    period = 2 * math.pi / theta
    tabs = {k: v.cpu() for k, v in build_tables(model).items()}
    tap = IdTap()
    model.model.embed_tokens.register_forward_pre_hook(lambda m, inp: setattr(tap, "ids", inp[0]))
    wins = wikitext_windows(tok, "wikitext-2-raw-v1", "test", 1024, max_windows=32)
    orig = model.model.layers[0].self_attn
    # средняя амплитуда частоты в голове — для отбора K частот
    share = tabs["aq"].mean(0) * tabs["ak"].mean(0).repeat_interleave(H // cfg.num_key_value_heads, dim=0)  # (H,M)
    order = share.argsort(dim=1, descending=True)

    def topk_mask(K):
        m = torch.zeros(H, M, dtype=torch.bool)
        m.scatter_(1, order[:, :K], True)
        return m

    slow = period > cfg.max_position_embeddings
    variants = {
        "exact": dict(),
        "fp16": dict(dtype=torch.float16),
        "top24": dict(keep=topk_mask(24)),
        "top16": dict(keep=topk_mask(16)),
        "top8": dict(keep=topk_mask(8)),
        "slowconst": dict(slow=slow),
    }
    res = {"model": "taobao-mnn/MNN-Pocket-0.3B", "layer": 0, "eval": {"windows": int(wins.shape[0]), "len": 1024},
           "slow_freqs": slow.nonzero().flatten().tolist(),
           "table_bytes_fp16": int(sum(v.numel() for v in tabs.values()) * 2), "variants": {}}
    for name, kw in variants.items():
        mod = ExplicitAttn0(orig, tabs, tap, theta, **kw)
        rep = Replacement(model)
        rep.add(0, "self_attn", mod)
        r = compare_lm(model, rep, wins, device, bs=1)
        rep.disable()
        row = {"kl": r["kl"], "dppl_pct": 100 * (r["ppl"] / r["ppl_teacher"] - 1), "top1": r["top1_agree"],
               "ppl": r["ppl"], "ppl_teacher": r["ppl_teacher"]}
        res["variants"][name] = row
        print(f"{name:10s} KL {row['kl']:.2e}  dPPL {row['dppl_pct']:+.4f}%  top1 {row['top1']:.4f}", flush=True)
        del mod
    res["seconds"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "layer0_explicit.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"PPL исходной {res['variants']['exact']['ppl_teacher']:.3f}; таблицы fp16: "
          f"{res['table_bytes_fp16'] / 2**20:.0f} МБ; {res['seconds']}s; сохранено {OUT / 'layer0_explicit.json'}")


if __name__ == "__main__":
    main()
