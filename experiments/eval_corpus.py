#!/usr/bin/env python
"""PPL/KL главных конфигураций на втором корпусе (не WikiText): проверка, что калибровка на WikiText-103
не переобучилась под домен. Корпус: NeelNanda/pile-10k (разнородные документы), запасной — LAMBADA.
Результат: results/<model>/corpus2/<name>_k<k>.json (поля как у run.py: lm = compare_lm)."""
from __future__ import annotations

import os

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")  # модель из кэша; датасет качаем (нужна сеть)
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detlayers.common import calibration_windows, load_model, pick_device, save_json, seed_all, wikitext_windows  # noqa: E402
from detlayers.methods import build  # noqa: E402
from detlayers.metrics import compare_lm  # noqa: E402
from sweep import CONFIGS, k_sink  # noqa: E402

ROOT = Path(__file__).resolve().parent


def corpus_windows(tok, name, seq_len, n, seed=0):
    from datasets import load_dataset
    local = ROOT / "data" / "pile-10k.jsonl"
    if name == "pile10k" and local.exists():  # офлайн-копия (tools/export_data.py)
        import json as _json
        texts = [t for t in (_json.loads(l)["text"] for l in open(local, encoding="utf-8")) if len(t) > 500]
    elif name == "pile10k":
        ds = load_dataset("NeelNanda/pile-10k", split="train")
        texts = [t for t in ds["text"] if len(t) > 500]
    elif name.startswith("selfgen"):  # текст, сгенерированный самой моделью (selfgen_calib.py)
        return torch.load(ROOT / "data" / f"{name}.pt")[: n].to(torch.long)
    elif name == "lambada":
        ds = load_dataset("EleutherAI/lambada_openai", "default", split="test")
        texts = list(ds["text"])
    else:
        raise ValueError(name)
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(texts), generator=g).tolist()
    if seed != 0:  # калибровочные документы — из второй половины перестановки seed 0, чтобы не пересекаться с оценкой
        base = torch.randperm(len(texts), generator=torch.Generator().manual_seed(0)).tolist()
        order = base[len(base) // 2:]
    sep = [tok.eos_token_id] if tok.eos_token_id is not None else []
    ids = []
    for i in order:
        ids += tok(texts[i])["input_ids"] + sep
        if len(ids) >= n * seq_len:
            break
    return torch.tensor(ids[: n * seq_len]).view(n, seq_len)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="0.6b")
    ap.add_argument("--corpus", default="pile10k")
    ap.add_argument("--configs", default="orig,L1-window64,MO4-routed25r,MO4-routed25,ML5-svdllm50,B-window+MO4-25r")
    ap.add_argument("--windows", type=int, default=32)
    ap.add_argument("--calib-tokens", type=int, default=131_072)
    ap.add_argument("--calib", default="wikitext", help="wikitext (как в статье) или pile10k — калибровка на втором корпусе")
    args = ap.parse_args()
    seed_all(0)
    device = pick_device(None)
    model, tok = load_model(args.model, device, "fp32")
    try:
        wins = corpus_windows(tok, args.corpus, 1024, args.windows)
        corpus = args.corpus
    except Exception as e:  # noqa: BLE001
        print("не удалось загрузить", args.corpus, "->", repr(e)[:200], "; беру lambada", flush=True)
        wins = corpus_windows(tok, "lambada", 1024, args.windows)
        corpus = "lambada"
    k = k_sink(args.model)
    calib = None
    sub = "corpus2" if args.calib == "wikitext" else f"corpus2_calib_{args.calib}"
    wt = wikitext_windows(tok, "wikitext-2-raw-v1", "test", 1024, max_windows=args.windows) if args.calib != "wikitext" else None
    for name in args.configs.split(","):
        method, params, group = CONFIGS[name]
        kk = 0 if name == "orig" else k
        out = ROOT / "results" / args.model / sub / (f"{name}.json" if name == "orig" else f"{name}_k{k}.json")
        if out.exists():
            print("[skip]", out.name, flush=True)
            continue
        if calib is None and method not in ("orig", "sparse"):
            if args.calib == "wikitext":
                calib = calibration_windows(tok, args.calib_tokens, 512, seed=0)
            else:  # калибровка из второго корпуса: другие документы (seed 1), чем в оценке (seed 0)
                calib = corpus_windows(tok, args.calib, 512, args.calib_tokens // 512, seed=1)
        t0 = time.time()
        rep, info = build(model, method, kk, calib, device, bs=8, **params)
        lm = compare_lm(model, rep, wins, device, bs=1)
        lm_wt = compare_lm(model, rep, wt, device, bs=1) if wt is not None else None
        rep.disable()
        res = {"model": args.model, "method": method, "k": kk, "params": params, "corpus": corpus, "calib": args.calib,
               "windows": int(wins.shape[0]), "seq_len": int(wins.shape[1]), "build": info, "lm": lm,
               "lm_wikitext": lm_wt, "seconds": round(time.time() - t0, 1)}
        save_json(res, out)
        print(name, corpus, json.dumps({a: round(b, 4) for a, b in lm.items() if isinstance(b, float)}), flush=True)
        del rep


if __name__ == "__main__":
    main()
