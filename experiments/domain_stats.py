#!/usr/bin/env python
"""Априорный предиктор доменного разрыва: статистики слоёв 0..5 на трёх корпусах (WikiText-2 test, pile-10k,
самогенерированный текст) без запуска замены: доля вклада топ-25% нейронов, локальная ошибка отбора 25% (отн. остаточного
потока), локальная ошибка окна 64. Результат: results/<model>/criterion/domain_stats.json."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detlayers.attention import SparseAttention  # noqa: E402
from detlayers.common import embed, load_model, pick_device, run_layers, save_json, seed_all, wikitext_windows  # noqa: E402
from detlayers.mlp import TopKMLP, _z  # noqa: E402
from eval_corpus import corpus_windows  # noqa: E402

ROOT = Path(__file__).resolve().parent


def rel_to(a, b, ref):
    return ((a[:, 1:].float() - b[:, 1:].float()).norm(dim=-1) / ref[:, 1:].float().norm(dim=-1).clamp_min(1e-6)).mean().item()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="0.6b")
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--windows", type=int, default=16)
    args = ap.parse_args()
    seed_all(0)
    dev = pick_device(None)
    model, tok = load_model(args.model, dev, "fp32")
    layers = model.model.layers
    kap = int(0.25 * model.config.intermediate_size)
    corpora = {"wikitext": wikitext_windows(tok, "wikitext-2-raw-v1", "test", 1024, max_windows=args.windows),
               "pile10k": corpus_windows(tok, "pile10k", 1024, args.windows)}
    sg = ROOT / "data" / f"selfgen_{args.model}.pt"
    if sg.exists():
        corpora["selfgen"] = torch.load(sg)[: args.windows * 2].reshape(-1)[: args.windows * 1024].view(args.windows, 1024)
    res = {"model": args.model, "corpora": {}}
    for cname, wins in corpora.items():
        acc = {k: [0.0] * args.layers for k in ("top25_share", "inj_mlp", "e_mlp_local", "e_attn_local", "inj_attn")}
        for s in range(wins.shape[0]):
            ids = wins[s:s + 1].to(dev)
            h, pos, pe = embed(model, ids)
            for l in range(args.layers):
                L = layers[l]
                xn = L.input_layernorm(h)
                aT, _ = L.self_attn(xn, position_embeddings=pe, attention_mask=None)
                aW, _ = SparseAttention(L.self_attn, window=64, n_sink=1)(xn, pe)
                acc["e_attn_local"][l] += ((aW[:, 1:] - aT[:, 1:]).float().norm(dim=-1) / aT[:, 1:].float().norm(dim=-1).clamp_min(1e-6)).mean().item()
                acc["inj_attn"][l] += rel_to(aW, aT, h)
                mid = h + aT
                xm = L.post_attention_layernorm(mid)
                mT = L.mlp(xm)
                mK = TopKMLP(L.mlp, kap)(xm)
                acc["e_mlp_local"][l] += ((mK[:, 1:] - mT[:, 1:]).float().norm(dim=-1) / mT[:, 1:].float().norm(dim=-1).clamp_min(1e-6)).mean().item()
                acc["inj_mlp"][l] += rel_to(mK, mT, mid)
                wd = L.mlp.down_proj.weight.norm(dim=0)
                c = _z(L.mlp, xm)[0, 1:].float().abs() * wd
                acc["top25_share"][l] += (c.topk(kap, dim=-1).values.sum(-1) / c.sum(-1).clamp_min(1e-9)).mean().item()
                h = mid + mT
        res["corpora"][cname] = {k: [round(v / wins.shape[0], 5) for v in vals] for k, vals in acc.items()}
        print(cname, {k: [round(x, 3) for x in v[:4]] for k, v in res["corpora"][cname].items()}, flush=True)
    save_json(res, ROOT / "results" / args.model / "criterion" / "domain_stats.json")


if __name__ == "__main__":
    main()
