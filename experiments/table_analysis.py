#!/usr/bin/env python
"""Пункт 6: что в первых слоях НЕ покрывается словарём. Для замен первых k слоёв (таблица по токену робастн.,
окно 64, M-O4 робастн.) считаем ошибку остаточного потока на входе слоя k по токенам и агрегируем:
по частоте токена (квартили частоты в калибровке), по типу (пунктуация / цифры / начало слова / продолжение слова /
перевод строки) и топ-20 токенов с наибольшей ошибкой. Результат: results/<model>/criterion/table_analysis.json."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detlayers.common import calibration_windows, embed, load_model, pick_device, run_layers, save_json, seed_all, wikitext_windows  # noqa: E402
from detlayers.methods import build  # noqa: E402
from sweep import CONFIGS, k_sink  # noqa: E402

ROOT = Path(__file__).resolve().parent


def tok_type(s: str) -> str:
    t = s.replace("Ġ", " ")
    if "\n" in t:
        return "перевод строки"
    st = t.strip()
    if not st:
        return "пробел"
    if all(ch.isdigit() for ch in st):
        return "цифры"
    if all(not ch.isalnum() for ch in st):
        return "пунктуация"
    return "начало слова" if t.startswith(" ") else "продолжение слова"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="0.6b")
    ap.add_argument("--configs", default="MO2-table-r,L1-window64,MO4-routed25r")
    ap.add_argument("--windows", type=int, default=16)
    args = ap.parse_args()
    seed_all(0)
    dev = pick_device(None)
    model, tok = load_model(args.model, dev, "fp32")
    k = k_sink(args.model)
    calib = calibration_windows(tok, 131_072, 512, seed=0)
    freq = torch.bincount(calib.flatten(), minlength=model.config.vocab_size).float()
    wins = wikitext_windows(tok, "wikitext-2-raw-v1", "test", 1024, max_windows=args.windows)
    res = {"model": args.model, "k": k, "configs": {}}
    for name in args.configs.split(","):
        method, params, _ = CONFIGS[name]
        rep, info = build(model, method, k, calib, dev, bs=8, **params)
        err = defaultdict(list)
        for s in range(wins.shape[0]):
            ids = wins[s:s + 1].to(dev)
            rep.disable()
            h, pos, pe = embed(model, ids)
            hT = run_layers(model, model.model.layers, h, pos, pe, 0, k)
            rep.enable()
            for mod in rep.new.values():  # таблицы по токенам читают ids через tap (ручной проход не вызывает хук)
                for sub in (mod, getattr(mod, "approx", None)):  # ExactPos0 оборачивает TableMLP
                    if sub is not None and getattr(sub, "tap", None) is not None:
                        sub.tap.ids = ids
            h, pos, pe = embed(model, ids)
            hS = run_layers(model, model.model.layers, h, pos, pe, 0, k)
            rep.disable()
            e = ((hS - hT)[0, 1:].float().norm(dim=-1) / hT[0, 1:].float().norm(dim=-1).clamp_min(1e-6)).cpu()
            for t, v in zip(ids[0, 1:].tolist(), e.tolist()):
                err[t].append(v)
        # агрегаты
        per_tok = {t: (sum(v) / len(v), len(v)) for t, v in err.items()}
        all_e = torch.tensor([v for vals in err.values() for v in vals])
        fr = torch.tensor([freq[t].item() for t in err for _ in err[t]])
        qs = fr.quantile(torch.tensor([0.25, 0.5, 0.75]))
        by_freq = {}
        for lab, lo, hi in (("редкие (Q1)", -1, qs[0]), ("Q2", qs[0], qs[1]), ("Q3", qs[1], qs[2]), ("частые (Q4)", qs[2], 1e18)):
            m = (fr > lo) & (fr <= hi)
            by_freq[lab] = float(all_e[m].mean()) if m.any() else None
        by_type = defaultdict(list)
        for t, vals in err.items():
            by_type[tok_type(tok.convert_ids_to_tokens(t))] += vals
        by_type = {a: (sum(v) / len(v), len(v)) for a, v in by_type.items()}
        worst = sorted(((m_, n_, t) for t, (m_, n_) in per_tok.items() if n_ >= 5), reverse=True)[:20]
        res["configs"][name] = {"mean_err": float(all_e.mean()), "by_freq_quartile": by_freq,
                                "by_type": {a: {"mean_err": round(v[0], 4), "count": v[1]} for a, v in by_type.items()},
                                "worst_tokens": [{"token": tok.decode([t]), "mean_err": round(m_, 3), "count": n_} for m_, n_, t in worst]}
        print(name, "ср. ошибка", round(float(all_e.mean()), 3), "| по частоте:", {a: round(b, 3) for a, b in by_freq.items() if b},
              "| по типу:", {a: round(v[0], 3) for a, v in by_type.items()}, flush=True)
        print("   худшие токены:", [(w["token"], w["mean_err"]) for w in res["configs"][name]["worst_tokens"][:10]], flush=True)
        del rep
    save_json(res, ROOT / "results" / args.model / "criterion" / "table_analysis.json")


if __name__ == "__main__":
    main()
