#!/usr/bin/env python
"""Пункт 3: природа ошибок. Для одиночных замен слоя ℓ (внимание → окно 64 + сток; MLP → 25% нейронов по
активациям) берём ошибку на выходе модели δ_ℓ = h_final(S_ℓ) − h_final(T) по токенам и считаем:
  эффективный ранг ковариации δ_ℓ (participation ratio) и долю энергии в 8 главных компонентах;
  средний по токенам косинус между δ_ℓ и δ_ℓ' для пар слоёв (выравнивание ошибок ⇒ суперaддитивность).
Результат: results/<model>/criterion/error_structure.json."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detlayers.attention import SparseAttention  # noqa: E402
from detlayers.common import load_model, pick_device, save_json, seed_all, wikitext_windows  # noqa: E402
from detlayers.methods import Replacement  # noqa: E402
from detlayers.mlp import ExactPos0, TopKMLP  # noqa: E402

ROOT = Path(__file__).resolve().parent


@torch.no_grad()
def final_hidden(model, ids):
    return model.model(ids).last_hidden_state[0, 1:].float()  # (T-1, d), после финальной нормы


def eff_rank(d):
    c = torch.cov(d.T.double())
    ev = torch.linalg.eigvalsh(c).clamp_min(0)
    return float(ev.sum() ** 2 / (ev ** 2).sum()), float(ev[-8:].sum() / ev.sum())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="0.6b")
    ap.add_argument("--layers", default="0,1,2,3,4,5,7,10,16,22")
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--ratio", type=float, default=0.25)
    args = ap.parse_args()
    seed_all(0)
    dev = pick_device(None)
    model, tok = load_model(args.model, dev, "fp32")
    layers = model.model.layers
    kap = int(args.ratio * model.config.intermediate_size)
    L = [int(x) for x in args.layers.split(",")]
    wins = wikitext_windows(tok, "wikitext-2-raw-v1", "test", 1024, max_windows=args.windows)
    deltas = {("attn", l): [] for l in L} | {("mlp", l): [] for l in L}
    for s in range(wins.shape[0]):
        ids = wins[s:s + 1].to(dev)
        hT = final_hidden(model, ids)
        for l in L:
            for kind, mod in (("attn", SparseAttention(layers[l].self_attn, window=64, n_sink=1)),
                              ("mlp", ExactPos0(layers[l].mlp, TopKMLP(layers[l].mlp, kap)))):
                rep = Replacement(model)
                rep.add(l, "self_attn" if kind == "attn" else "mlp", mod)
                rep.enable()
                deltas[(kind, l)].append((final_hidden(model, ids) - hT).cpu())
                rep.disable()
        print(f"окно {s + 1}/{wins.shape[0]}", flush=True)
    res = {"model": args.model, "layers": L, "per_layer": {}, "pairwise_cos": {}}
    D = {key: torch.cat(v) for key, v in deltas.items()}  # (N, d)
    for kind in ("attn", "mlp"):
        for l in L:
            er, top8 = eff_rank(D[(kind, l)])
            res["per_layer"][f"{kind}{l}"] = {"eff_rank": er, "top8_energy": top8, "rms": float(D[(kind, l)].norm(dim=-1).mean())}
        cos = torch.zeros(len(L), len(L))
        for i, a in enumerate(L):
            for j, b in enumerate(L):
                cos[i, j] = F.cosine_similarity(D[(kind, a)], D[(kind, b)], dim=-1).mean()
        off = cos[~torch.eye(len(L), dtype=torch.bool)]
        res["pairwise_cos"][kind] = {"matrix": [[round(float(x), 4) for x in row] for row in cos],
                                     "mean_offdiag": float(off.mean()), "mean_offdiag_first6": float(cos[:6, :6][~torch.eye(6, dtype=torch.bool)].mean())}
    for kind in ("attn", "mlp"):
        print(kind, "эфф. ранг по слоям:", [round(res["per_layer"][f"{kind}{l}"]["eff_rank"], 1) for l in L],
              "| ср. косинус ошибок разных слоёв:", round(res["pairwise_cos"][kind]["mean_offdiag"], 3),
              "(первые 6:", round(res["pairwise_cos"][kind]["mean_offdiag_first6"], 3), ")", flush=True)
    save_json(res, ROOT / "results" / args.model / "criterion" / "error_structure.json")


if __name__ == "__main__":
    main()
