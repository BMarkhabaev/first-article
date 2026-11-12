"""Общие утилиты: устройство, загрузка модели/данных, ручной прямой проход по первым слоям."""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

MODELS = {
    "0.6b": "Qwen/Qwen3-0.6B-Base",
    "1.7b": "Qwen/Qwen3-1.7B-Base",
    "4b": "Qwen/Qwen3-4B-Base",
    "8b": "Qwen/Qwen3-8B-Base",
    "14b": "Qwen/Qwen3-14B-Base",
    "32b": "Qwen/Qwen3-32B",  # Base-варианта 32B не выпущено — пост-обученная модель (та же архитектура)
    # второе семейство (архитектура Llama: без QK-norm, MHA/GQA, SwiGLU)
    "smol1.7b": "HuggingFaceTB/SmolLM2-1.7B",
    "llama1b": "meta-llama/Llama-3.2-1B",
}
# Офлайн-машина (H100): веса в MODELS_DIR/Qwen3-<размер>[-Base]/, тексты в DATA_DIR (см. tools/export_data.py)
MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")
DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parent.parent / "data"))


def resolve_model(model: str) -> str:
    """Ключ MODELS → локальная папка (если есть) или HF id; путь/HF id возвращается как есть."""
    key = model.lower()
    if key not in MODELS:
        return model
    repo = MODELS[key].split("/")[1]
    for cand in {repo, repo.replace("-Base", ""), repo + "-Base"}:
        if (Path(MODELS_DIR) / cand / "config.json").exists():
            return str(Path(MODELS_DIR) / cand)
    return MODELS[key]


def pick_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def load_model(model: str, device: torch.device, dtype: torch.dtype | str = torch.bfloat16):
    """Загружает Qwen3 Base. `model` — ключ из MODELS или путь/HF id."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if isinstance(dtype, str):
        dtype = DTYPES[dtype]
    name = resolve_model(model)
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, attn_implementation="sdpa")
    m.to(device).eval()
    m.config.use_cache = False
    return m, tok


# ---------------------------------------------------------------- данные

def _wikitext_text(config: str, split: str) -> str:
    """Текст WikiText: из DATA_DIR/<config>.<split>.txt (офлайн), иначе из datasets."""
    local = DATA_DIR / f"{config}.{split}.txt"
    if local.exists():
        return local.read_text(encoding="utf-8")
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", config, split=split)
    return "\n\n".join(ds["text"])


def wikitext_windows(tok, config: str, split: str, seq_len: int, max_windows: int | None = None,
                     max_chars: int | None = None) -> torch.Tensor:
    """Неперекрывающиеся окна токенов длины seq_len (как в стандартном PPL-протоколе)."""
    text = _wikitext_text(config, split)
    if max_chars:
        text = text[:max_chars]
    ids = tok(text, return_tensors="pt").input_ids[0]
    n = ids.numel() // seq_len
    if max_windows:
        n = min(n, max_windows)
    return ids[: n * seq_len].view(n, seq_len)


def calibration_windows(tok, n_tokens: int, seq_len: int, seed: int = 0) -> torch.Tensor:
    """Калибровка: WikiText-103 train (не пересекается с WikiText-2 test).
    Берём случайные (seed) непрерывные окна из первых ~4*n_tokens токенов корпуса."""
    text = _wikitext_text("wikitext-103-raw-v1", "train")
    # ~4 символа на токен; берём с запасом x4, чтобы было из чего выбирать
    text = text[: max(n_tokens * 16, 2_000_000)]
    ids = tok(text, return_tensors="pt").input_ids[0]
    n_avail = ids.numel() // seq_len
    windows = ids[: n_avail * seq_len].view(n_avail, seq_len)
    g = torch.Generator().manual_seed(seed)
    n_need = min(n_avail, max(1, n_tokens // seq_len))
    idx = torch.randperm(n_avail, generator=g)[:n_need]
    return windows[idx]


# ------------------------------------------------- ручной проход по слоям

@dataclass
class Prefix:
    """Состояния остаточного потока после первых `upto` слоёв."""
    hidden: torch.Tensor  # (B, T, d)


def embed(model, input_ids: torch.Tensor):
    inner = model.model
    h = inner.embed_tokens(input_ids)
    pos = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    pe = inner.rotary_emb(h, pos)
    return h, pos, pe


def run_layers(model, layers, h, pos, pe, start: int, stop: int):
    """Прогон слоёв layers[start:stop]. attention_mask=None + sdpa => каузальная маска.
    Наши заменённые модули всегда каузальны сами по себе."""
    for i in range(start, stop):
        h = layers[i](h, attention_mask=None, position_embeddings=pe, position_ids=pos,
                      past_key_values=None, use_cache=False)
        if isinstance(h, tuple):
            h = h[0]
    return h


@torch.no_grad()
def prefix_hidden(model, input_ids: torch.Tensor, upto: int, layers=None) -> torch.Tensor:
    layers = layers if layers is not None else model.model.layers
    h, pos, pe = embed(model, input_ids)
    return run_layers(model, layers, h, pos, pe, 0, upto)


# ------------------------------------------------------------- I/O

def save_json(obj, path: os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def env_info(device: torch.device) -> dict:
    import transformers

    info = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": str(device),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.dt = time.perf_counter() - self.t0
