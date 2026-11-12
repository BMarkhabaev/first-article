#!/usr/bin/env python
"""Экспорт текстов для офлайн-машины (запускать там, где есть интернет): data/wikitext-2-raw-v1.test.txt,
data/wikitext-103-raw-v1.train.txt (первые 64M символов), data/pile-10k.jsonl."""
import json
from pathlib import Path

from datasets import load_dataset

out = Path(__file__).resolve().parent.parent / "data"
out.mkdir(exist_ok=True)
for cfg, split, lim in (("wikitext-2-raw-v1", "test", None), ("wikitext-103-raw-v1", "train", 64_000_000)):
    t = "\n\n".join(load_dataset("Salesforce/wikitext", cfg, split=split)["text"])
    (out / f"{cfg}.{split}.txt").write_text(t[:lim] if lim else t, encoding="utf-8")
with open(out / "pile-10k.jsonl", "w", encoding="utf-8") as f:
    for x in load_dataset("NeelNanda/pile-10k", split="train"):
        f.write(json.dumps({"text": x["text"]}, ensure_ascii=False) + "\n")
print("готово:", sorted(p.name for p in out.iterdir()))
