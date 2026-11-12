"""Метрики качества замены: PPL, KL(учитель‖ученик), совпадение top-1, ошибка остаточного
потока; запуск lm-eval на изменённой модели."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .common import prefix_hidden


@torch.no_grad()
def compare_lm(model, rep, windows: torch.Tensor, device, bs: int = 1, probe_layer: int | None = None):
    """Учитель и ученик в одном процессе (Replacement включается/выключается).
    Возвращает PPL обоих, средний KL(p_T‖p_S) на токен, долю совпадения argmax,
    относительную ошибку остаточного потока на слое probe_layer и на выходе."""
    nll_t = nll_s = kl = agree = 0.0
    n = 0
    rel_probe = rel_final = 0.0
    nb = 0
    per = {"nll_t": [], "nll_s": [], "kl": [], "tokens": []}  # по окнам — для бутстрэп-интервалов
    for s in range(0, windows.shape[0], bs):
        ids = windows[s:s + bs].to(device)
        with rep.teacher():
            out_t = model(ids, output_hidden_states=True)
        rep.enable()
        out_s = model(ids, output_hidden_states=True)
        lt = out_t.logits[:, :-1].float()
        ls = out_s.logits[:, :-1].float()
        tgt = ids[:, 1:]
        lpt = F.log_softmax(lt, -1)
        lps = F.log_softmax(ls, -1)
        w_nt = -lpt.gather(-1, tgt[..., None]).squeeze(-1).sum(-1)
        w_ns = -lps.gather(-1, tgt[..., None]).squeeze(-1).sum(-1)
        w_kl = (lpt.exp() * (lpt - lps)).sum(-1).sum(-1)
        per["nll_t"] += w_nt.tolist(); per["nll_s"] += w_ns.tolist(); per["kl"] += w_kl.tolist()
        per["tokens"] += [tgt.shape[1]] * tgt.shape[0]
        nll_t += w_nt.sum().item()
        nll_s += w_ns.sum().item()
        kl += w_kl.sum().item()
        agree += (lt.argmax(-1) == ls.argmax(-1)).sum().item()
        n += tgt.numel()
        # ошибка остаточного потока (без позиции 0 с массивной активацией)
        def rel(a, b):
            a, b = a[:, 1:].float(), b[:, 1:].float()
            return ((a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-6)).mean().item()
        if probe_layer is not None:
            rel_probe += rel(out_s.hidden_states[probe_layer], out_t.hidden_states[probe_layer])
        rel_final += rel(out_s.hidden_states[-1], out_t.hidden_states[-1])
        nb += 1
    res = {
        "ppl_teacher": math.exp(nll_t / n),
        "ppl": math.exp(nll_s / n),
        "kl": kl / n,
        "top1_agree": agree / n,
        "rel_err_final": rel_final / nb,
        "tokens": n,
        "per_window": {k: [round(v, 4) for v in vals] for k, vals in per.items()},
    }
    if probe_layer is not None:
        res["rel_err_layer_k"] = rel_probe / nb
    return res


def run_lm_eval(model, tok, tasks: list[str], limit: int | float | None, batch_size: int, device,
                num_fewshot: dict[str, int] | None = None, max_length: int = 2048,
                limits: dict[str, int | None] | None = None,
                batch_sizes: dict[str, int] | None = None) -> dict:
    """lm-eval на уже изменённой (Replacement включён) модели. MMLU — 5-shot (как в отчёте
    Qwen3 для Base-моделей), остальное — 0-shot."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=batch_size, max_length=max_length,
              device=str(device))
    num_fewshot = num_fewshot or {}
    results = {}
    groups = {}
    limits = limits or {}
    keyf = lambda t: (num_fewshot.get(t, 0), limits.get(t, limit) or 0)  # noqa: E731
    for nf, lim in sorted(set(keyf(t) for t in tasks)):
        ts = [t for t in tasks if keyf(t) == (nf, lim)]
        bs_override = [batch_sizes[t] for t in ts if batch_sizes and t in batch_sizes]
        if bs_override and str(batch_size).isdigit():  # напр. MMLU 5-shot — длинные промпты
            lm.batch_size_per_gpu = min(bs_override)
        elif str(batch_size).isdigit():
            lm.batch_size_per_gpu = int(batch_size)
        while True:  # MPS: при нехватке памяти уменьшаем пакет вдвое (результат от пакета не зависит)
            try:
                r = lm_eval.simple_evaluate(model=lm, tasks=ts, num_fewshot=nf, limit=lim or None,
                                            random_seed=0, numpy_random_seed=0, torch_random_seed=0,
                                            fewshot_random_seed=0, log_samples=False, bootstrap_iters=0)
                break
            except torch.OutOfMemoryError:
                bs_now = int(lm.batch_size_per_gpu) if str(lm.batch_size_per_gpu).isdigit() else 0
                if bs_now <= 1:
                    raise
                import gc; gc.collect()
                if hasattr(torch, "mps"):
                    torch.mps.empty_cache()
                lm.batch_size_per_gpu = max(1, bs_now // 2)
                print(f"[lm-eval] OOM, пакет {bs_now} -> {lm.batch_size_per_gpu}", flush=True)
        for name, v in r["results"].items():
            results[name] = {m: x for m, x in v.items() if isinstance(x, (int, float)) and "stderr" not in m}
        groups.update(r.get("groups", {}) and {g: {m: x for m, x in v.items() if isinstance(x, (int, float))}
                                                 for g, v in r["groups"].items()})
    return {"tasks": results, "groups": groups}


def summarize_lm_eval(res: dict, tasks: list[str]) -> dict:
    """Одна основная метрика на задачу (acc_norm, если есть, иначе acc; для lambada — acc и ppl)."""
    out = {}
    tr = res["tasks"]
    for t in tasks:
        src = tr.get(t) or res.get("groups", {}).get(t)
        if src is None:
            continue
        if t.startswith("lambada"):
            out[t] = {"acc": src.get("acc,none"), "ppl": src.get("perplexity,none")}
        elif "acc_norm,none" in src and t in ("hellaswag", "arc_easy", "arc_challenge", "piqa"):
            out[t] = {"acc_norm": src.get("acc_norm,none"), "acc": src.get("acc,none")}
        else:
            out[t] = {"acc": src.get("acc,none")}
    return out
