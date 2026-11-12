#!/usr/bin/env python
"""Доверительные интервалы: бенчмарки — stderr из lm-eval (lm_eval_raw), PPL/KL — бутстрэп по окнам (per_window,
есть у прогонов после 2026-11-10). Таблица tables/ci_<model>.tex и печать."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
TASKS = [("lambada_openai", "acc"), ("hellaswag", "acc_norm"), ("arc_easy", "acc_norm"), ("arc_challenge", "acc_norm"),
         ("piqa", "acc_norm"), ("winogrande", "acc"), ("mmlu", "acc")]


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def bootstrap(per, B=2000, seed=0):
    g = torch.Generator().manual_seed(seed)
    nt, ns, kl, tok = (torch.tensor(per[k], dtype=torch.float64) for k in ("nll_t", "nll_s", "kl", "tokens"))
    n = nt.numel()
    idx = torch.randint(n, (B, n), generator=g)
    T = tok[idx].sum(1)
    dppl = 100 * (torch.exp((ns[idx].sum(1) - nt[idx].sum(1)) / T) - 1)
    klb = kl[idx].sum(1) / T
    q = lambda x: (x.quantile(0.025).item(), x.quantile(0.975).item())  # noqa: E731
    return q(dppl), q(klb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="0.6b")
    ap.add_argument("--configs", default="orig,L1-window64,MO4-routed25r,ML5-svdllm50,B-window+MO4-25r")
    ap.add_argument("--dirs", default="ci,bench1000,final")
    args = ap.parse_args()
    rows = []
    for name in args.configs.split(","):
        r = None
        for d in args.dirs.split(","):
            fn = "orig.json" if name == "orig" else f"{name}_k3.json"
            r = load(ROOT / "results" / args.model / d / fn)
            if r:
                src = d
                break
        if not r:
            rows.append((name, "—", "—", "—", "—"))
            continue
        lm = r["lm"]
        ci = ""
        if "per_window" in lm:
            (a, b), (c, d_) = bootstrap(lm["per_window"])
            ci = f"[{a:.1f}; {b:.1f}]"
            kl_ci = f"[{c:.3f}; {d_:.3f}]"
        else:
            kl_ci = ""
        dppl = 100 * (lm["ppl"] / lm["ppl_teacher"] - 1)
        raw = (r.get("lm_eval_raw") or {})
        if not raw:  # у прогонов для ДИ бенчей нет — берём их из bench1000 / final
            for d2 in ("bench1000", "final"):
                r2 = load(ROOT / "results" / args.model / d2 / ("orig.json" if name == "orig" else f"{name}_k3.json"))
                if r2 and r2.get("lm_eval_raw"):
                    raw = r2["lm_eval_raw"]
                    src = f"{src}+{d2}"
                    break
        tr = raw.get("tasks", {})
        gr = raw.get("groups", {})
        accs, ses = [], []
        for t, m in TASKS:
            src_ = tr.get(t) or gr.get(t)
            if not src_:
                continue
            acc = src_.get(f"{m},none", float("nan"))
            n_ = src_.get("sample_len") or 0
            accs.append(100 * acc)
            ses.append(100 * math.sqrt(max(acc * (1 - acc), 0.0) / n_) if n_ else 0.0)  # биномиальная SE по размеру подвыборки
        mean = sum(accs) / len(accs) if accs else float("nan")
        se_mean = math.sqrt(sum(s * s for s in ses)) / len(ses) if ses else float("nan")
        rows.append((name, f"{dppl:.1f} {ci}", f"{lm['kl']:.3f} {kl_ci}", f"{mean:.1f} ± {1.96 * se_mean:.1f}" if accs else "—", src))
        print(f"{name:20s} dPPL {rows[-1][1]:22s} KL {rows[-1][2]:22s} бенчи ср. {rows[-1][3]:14s} ({src})")
    lines = [r"\begin{tabular}{lllll}", r"\toprule", r"Конфигурация & рост PPL, \% [95\% ДИ] & KL [95\% ДИ] & среднее по бенчам ± 1.96 SE & источник \\", r"\midrule"]
    for row in rows:
        lines.append(" & ".join(str(x).replace("%", r"\%") for x in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out = ROOT.parent / "tables" / f"ci_{args.model}.tex"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
