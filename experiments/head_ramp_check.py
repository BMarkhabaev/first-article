"""Проверка механизма «спад с расстоянием» слоя 0 заменой веса нормализации (без обучения).

Главная компонента k_norm (и парная ей компонента q_norm той же частоты RoPE) заменяется медианой, после чего
внимание слоя 0 пересчитывается на тексте по точной формуле. Сравнивается доля внимания в пределах 8 токенов,
дальше 256 токенов и среднее расстояние (запросы с контекстом ≥ 1024 токенов).

  python head_ramp_check.py --model pocket0.3b
Результат: results/<m>/heads/ramp_check.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from detlayers.common import load_model, wikitext_windows  # noqa: E402
from head_mechanisms import cplx, realvec  # noqa: E402
from table_attention import dims, project  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pocket0.3b")
    ap.add_argument("--windows", type=int, default=4)
    args = ap.parse_args()
    model, tok = load_model(args.model, torch.device("cpu"), "fp32")
    H, G, D = dims(model)
    M, rep = D // 2, H // G
    layer = model.model.layers[0]
    att = layer.self_attn
    scale = float(att.scaling)
    theta = model.model.rotary_emb.inv_freq.double()
    E = model.model.embed_tokens.weight
    wins = wikitext_windows(tok, "wikitext-2-raw-v1", "test", 2048, max_windows=args.windows)
    T = wins.shape[1]
    it = torch.arange(T)
    dist = it[:, None] - it[None, :]
    causal = dist >= 0
    rot = torch.exp(1j * it.double()[:, None] * theta[None, :])

    def measure():
        near, far, mean_d = torch.zeros(H), torch.zeros(H), torch.zeros(H)
        for w in wins:
            q, k, _ = project(att, layer.input_layernorm(E[w]))
            zq, zk = cplx(q) * rot[:, None, :], cplx(k) * rot[:, None, :]
            for h in range(H):
                s = (realvec(zq[:, h]) @ realvec(zk[:, h // rep]).T) * scale
                p = torch.softmax(s.masked_fill(~causal, float("-inf")), -1)[1024:]
                dd = dist[1024:].double()
                near[h] += float((p * (dd <= 8)).sum(-1).mean())
                far[h] += float((p * (dd > 256)).sum(-1).mean())
                mean_d[h] += float((p * dd).sum(-1).mean())
        n = wins.shape[0]
        return {"near8": (near / n).tolist(), "far256": (far / n).tolist(), "mean_distance": (mean_d / n).tolist()}

    gk, gq = att.k_norm.weight, att.q_norm.weight
    k0, q0 = gk.clone(), gq.clone()
    ck = int(gk.abs().argmax())
    m = ck % M
    pair_q = [m, m + M][int(gq[m + M].abs() > gq[m].abs())]  # большая из двух компонент q_norm той же частоты
    res = {"model": args.model, "k_dim": ck, "k_gamma": float(gk[ck]), "k_median": float(gk.abs().median()), "freq": m,
           "q_dim": pair_q, "q_gamma": float(gq[pair_q]), "q_median": float(gq.abs().median()),
           "period_tokens": float(2 * torch.pi / theta[m]), "original": measure()}
    gk[ck] = gk.abs().median()
    res["k_to_median"] = measure()
    gq[pair_q] = gq.abs().median()
    res["k_and_q_to_median"] = measure()
    gk.copy_(k0), gq.copy_(q0)
    for dd in (m, m + M):
        gk[dd] = 0
        gq[dd] = 0
    res["freq_off"] = measure()
    gk.copy_(k0), gq.copy_(q0)
    out = ROOT / "results" / args.model / "heads"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ramp_check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    for key in ("original", "k_to_median", "k_and_q_to_median", "freq_off"):
        print(f"{key:20s} >256: " + " ".join(f"{x:.2f}" for x in res[key]["far256"]))
    print("сохранено", out / "ramp_check.json")


if __name__ == "__main__":
    main()
