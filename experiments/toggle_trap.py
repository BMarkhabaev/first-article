#!/usr/bin/env python
"""Пункт 2: управляемый эксперимент — ловушку можно включить и выключить составом калибровки.
Сканируем пул окон, помечаем «отравленные» (на позиции ≥1 норма остаточного потока на входе слоя k превышает
tau·медиану); строим калибровки: чистая (0 отравленных) и с дозой n ∈ {1,4,16} отравленных окон; для каждой —
наивный M-O4 при k, KL/PPL на 32 окнах; для максимальной дозы — ещё робастная калибровка.
Результат: results/<model>/trap/toggle.json."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detlayers.common import calibration_windows, embed, load_model, pick_device, run_layers, save_json, seed_all, wikitext_windows  # noqa: E402
from detlayers.methods import build  # noqa: E402
from detlayers.metrics import compare_lm  # noqa: E402
from sweep import k_sink  # noqa: E402

ROOT = Path(__file__).resolve().parent


@torch.no_grad()
def scan(model, pool, k, dev, tau, bs=8):
    """Для каждого окна: макс. норма остаточного потока на входе слоя k по позициям ≥1 и медиана."""
    mx, med, pos, tokid = [], [], [], []
    for s in range(0, pool.shape[0], bs):
        ids = pool[s:s + bs].to(dev)
        h, p, pe = embed(model, ids)
        h = run_layers(model, model.model.layers, h, p, pe, 0, k)
        n = h[:, 1:].float().norm(dim=-1)  # (B,T-1)
        mx.append(n.max(dim=1).values.cpu()); med.append(n.median(dim=1).values.cpu())
        j = n.argmax(dim=1)
        pos.append((j + 1).cpu()); tokid.append(ids[torch.arange(ids.shape[0]), j + 1].cpu())
    mx, med, pos, tokid = map(torch.cat, (mx, med, pos, tokid))
    glob_med = med.median().item()
    poisoned = (mx > tau * glob_med).nonzero().flatten().tolist()
    return poisoned, mx, pos, tokid, glob_med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="0.6b")
    ap.add_argument("--k", type=int, default=0, help="0 = k_sink")
    ap.add_argument("--pool-windows", type=int, default=2048)
    ap.add_argument("--n-calib", type=int, default=256)
    ap.add_argument("--doses", default="0,1,4,16")
    ap.add_argument("--tau", type=float, default=10.0)
    args = ap.parse_args()
    seed_all(0)
    dev = pick_device(None)
    model, tok = load_model(args.model, dev, "fp32")
    k = args.k or k_sink(args.model)
    pool = calibration_windows(tok, args.pool_windows * 512, 512, seed=0)
    t0 = time.time()
    poisoned, mx, pos, tokid, gmed = scan(model, pool, k, dev, args.tau)
    print(f"пул {pool.shape[0]} окон; отравленных {len(poisoned)}; медиана нормы {gmed:.1f}; "
          f"примеры: {[(i, int(pos[i]), tok.decode([int(tokid[i])]), round(float(mx[i]))) for i in poisoned[:6]]}  {time.time()-t0:.0f}s", flush=True)
    clean_idx = [i for i in range(pool.shape[0]) if i not in set(poisoned)]
    wins = wikitext_windows(tok, "wikitext-2-raw-v1", "test", 1024, max_windows=32)
    res = {"model": args.model, "k": k, "tau": args.tau, "pool_windows": int(pool.shape[0]), "n_poisoned_in_pool": len(poisoned),
           "poisoned_examples": [{"window": i, "pos": int(pos[i]), "token": tok.decode([int(tokid[i])]), "norm": float(mx[i])} for i in poisoned[:20]],
           "median_norm": gmed, "runs": {}}
    doses = [int(x) for x in args.doses.split(",")]
    for n in doses:
        n_eff = min(n, len(poisoned))
        idx = clean_idx[: args.n_calib - n_eff] + poisoned[:n_eff]
        calib = pool[idx]
        for robust in ([None] if n != max(doses) else [None, args.tau]):
            t0 = time.time()
            params = {"active_ratio": 0.25, "predictor": "linear"} | ({"robust_tau": robust} if robust else {})
            rep, info = build(model, "routed_mlp", k, calib, dev, bs=8, **params)
            lm = compare_lm(model, rep, wins, dev, bs=1)
            rep.disable()
            key = f"dose{n_eff}" + ("_robust" if robust else "")
            res["runs"][key] = {"n_poisoned": n_eff, "robust_tau": robust, "dppl_pct": 100 * (lm["ppl"] / lm["ppl_teacher"] - 1), "kl": lm["kl"],
                                "top1_agree": lm["top1_agree"], "seconds": round(time.time() - t0)}
            print(key, json.dumps({a: round(b, 4) for a, b in res["runs"][key].items() if isinstance(b, float)}), flush=True)
            save_json(res, ROOT / "results" / args.model / "trap" / "toggle.json")
            del rep
    print("saved")


if __name__ == "__main__":
    main()
