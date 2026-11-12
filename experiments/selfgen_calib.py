#!/usr/bin/env python
"""Самогенерированная калибровка: сэмплируем из самой модели N окон по 512 токенов (без внешних данных, домен-нейтрально).
Старт — токен конца текста; do_sample, top_p=0.95, T=1.0. Результат: data/selfgen_<model>.pt (N×512 int64)."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detlayers.common import load_model, pick_device, seed_all  # noqa: E402

ROOT = Path(__file__).resolve().parent


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="0.6b")
    ap.add_argument("--windows", type=int, default=256)
    ap.add_argument("--len", type=int, default=512)
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()
    seed_all(0)
    dev = pick_device(None)
    model, tok = load_model(args.model, dev, "bf16")
    model.config.use_cache = True
    start = tok.eos_token_id if tok.eos_token_id is not None else 0
    out = []
    t0 = time.time()
    for b in range(0, args.windows, args.bs):
        n = min(args.bs, args.windows - b)
        ids = torch.full((n, 1), start, dtype=torch.long, device=dev)
        gen = model.generate(ids, do_sample=True, top_p=0.95, temperature=1.0, max_new_tokens=args.len,
                             min_new_tokens=args.len, pad_token_id=start)
        out.append(gen[:, 1:1 + args.len].cpu())
        print(f"{b + n}/{args.windows}  {time.time() - t0:.0f}s", flush=True)
    data = torch.cat(out)
    (ROOT / "data").mkdir(exist_ok=True)
    torch.save(data, ROOT / "data" / f"selfgen_{args.model}.pt")
    print("пример:", tok.decode(data[0, :60].tolist()).replace("\n", " ")[:200])
    print("saved", data.shape)


if __name__ == "__main__":
    main()
