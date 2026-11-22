"""Какие частоты RoPE несут выбор головы: преимущество цели по частотам.

Логит — сумма по частотам m, поэтому преимущество цели A (см. head_circuits.py) точно делится на вклады частот:
A = Σ_m A_m. Для каждой головы и каждого вида цели (сам токен, предыдущий, тот же токен, индукция, сток)
сохраняются три главные частоты и доля частоты, на которую приходится наибольший вес k_norm слоя.
Так видно, что делает большой вес нормализации: кодирует расстояние, сток или совпадение токенов.

  python head_freq_adv.py --model 0.6b
Результат: results/<m>/heads/freq_adv.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from detlayers.common import load_model, wikitext_windows  # noqa: E402
from head_circuits import TYPES  # noqa: E402
from head_mechanisms import cplx  # noqa: E402
from table_attention import dims, project  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pocket0.3b")
    ap.add_argument("--windows", type=int, default=6)
    args = ap.parse_args()
    model, tok = load_model(args.model, torch.device("cpu"), "fp32")
    H, G, D = dims(model)
    M, rep = D // 2, H // G
    layers = model.model.layers
    NL = len(layers)
    theta = model.model.rotary_emb.inv_freq.double().cpu()
    period = (2 * math.pi / theta).tolist()
    wins = wikitext_windows(tok, "wikitext-2-raw-v1", "test", 512, max_windows=args.windows)
    nW, T = wins.shape
    it = torch.arange(T)
    dist = it[:, None] - it[None, :]
    causal = dist >= 0
    rot = torch.exp(1j * it.double()[:, None] * theta[None, :])
    g_of_h = torch.arange(H) // rep
    cap = {}
    hooks = [layers[l].self_attn.register_forward_pre_hook(
        lambda m_, a, kw, l=l: cap.__setitem__(l, (kw.get("hidden_states") if "hidden_states" in kw else a[0])[0].detach()), with_kwargs=True)
        for l in range(NL)]
    A = torch.zeros(NL, len(TYPES), H, M, dtype=torch.float64)
    for w in wins:
        model(w[None])
        same = (w[:, None] == w[None, :]) & (dist > 0)
        ind = torch.zeros(T, T, dtype=torch.bool)
        ind[:, 1:] = (w[:, None] == w[None, :-1]) & (dist[:, 1:] > 0)
        tg = {"self": dist == 0, "prev": dist == 1, "same": same, "ind": ind, "sink": (it[None, :] == 0) & (dist > 0)}
        lo = {"self": 1, "prev": 2, "same": 1, "ind": 1, "sink": 64}
        Wt = torch.zeros(len(TYPES), T, T, dtype=torch.float64)
        nv = torch.zeros(len(TYPES), dtype=torch.float64)
        for ti, name in enumerate(TYPES):
            tgt = tg[name] & causal
            oth = causal & ~tgt
            nt, no = tgt.sum(1), oth.sum(1)
            ok = (nt > 0) & (no > 0) & (it >= lo[name])
            Wt[ti] = (tgt.double() / nt.clamp_min(1)[:, None] - oth.double() / no.clamp_min(1)[:, None]) * ok[:, None]
            nv[ti] = ok.sum().clamp_min(1)
        Wc = Wt.to(torch.complex128)
        for l in range(NL):
            att = layers[l].self_attn
            q, k, _ = project(att, cap[l])
            zq, zk = cplx(q) * rot[:, None, :], cplx(k) * rot[:, None, :]
            kap = torch.einsum("tij,jgm->tigm", Wc, zk)[:, :, g_of_h, :]  # (nT,T,H,M)
            A[l] += float(att.scaling) * (zq[None] * kap.conj()).real.sum(1) / nv[:, None, None] / nW
    for h_ in hooks:
        h_.remove()
    out = {"model": args.model, "H": H, "M": M, "types": TYPES, "period_tokens": period, "heads": []}
    for l in range(NL):
        gk = layers[l].self_attn.k_norm.weight.detach().abs()
        mk = int(gk.argmax()) % M
        for h in range(H):
            row = {"layer": l, "head": h, "k_norm_freq": mk, "adv": {}}
            for ti, name in enumerate(TYPES):
                a = A[l, ti, h]
                tot = float(a.sum())
                top = a.abs().topk(3).indices.tolist()
                slow = sum(float(a[m]) for m in range(M) if period[m] > 20_000)
                row["adv"][name] = {"logits": tot, "top": [(m, float(a[m])) for m in top], "k_norm_freq_logits": float(a[mk]),
                                    "slow_logits": slow, "fast_logits": sum(float(a[m]) for m in range(M) if period[m] < 20)}
            out["heads"].append(row)
    dst = ROOT / "results" / args.model / "heads"
    (dst / "freq_adv.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("сохранено", dst / "freq_adv.json")


if __name__ == "__main__":
    main()
