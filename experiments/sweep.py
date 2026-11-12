#!/usr/bin/env python
"""Все эксперименты статьи по этапам. Готовые JSON пропускаются (можно прерывать и продолжать).

  python sweep.py --stage profile            # профиль слоёв, все модели
  python sweep.py --stage ksweep             # 0.6B: KL/PPL по k для всех методов
  python sweep.py --stage pareto             # 0.6B: качество vs бюджет (MLP и внимание)
  python sweep.py --stage final              # все модели: таблица с бенчами
  python sweep.py --stage howmany            # 0.6B: «сколько слоёв», 7 методов, k до 12
  python sweep.py --stage howmany_other      # 1.7B (и крупнее): сокращённый howmany, 3 метода, k до 8
  python sweep.py --stage bench1000          # все модели: главные столбцы с бенчами на 1000 примерах
  python sweep.py --stage all                # всё по порядку
  python sweep.py --stage final --models 0.6b,1.7b --dry   # показать команды без запуска

Каждый запуск — отдельный процесс run.py (память освобождается между запусками).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MODELS = ["0.6b", "1.7b", "4b", "8b", "14b", "32b"]
BF16_MODELS = {"14b", "32b"}
SMALL_BATCH_MODELS = {"smol1.7b", "llama1b"}  # MHA/GQA-модели второго семейства на Mac: calib-bs 4, lm-bs 4
RUN_TIMEOUT_S = 150 * 60  # зависания MPS при k=8 (1.7B-класс): 0% CPU без прогресса  # fp32-веса не помещаются на одну карту: модель в bf16, суммы калибровки — fp32/fp64
# H100: полные бенчи, полный WikiText-2, калибровка 262k; для 8B+ — только столбцы таблицы (FINAL_MAIN + робастные)
H100_PRESET = {"limit": 0, "mmlu_limit": 0, "lm_bs": "auto:8", "distill_models": "0.6b,1.7b,4b"}
FULL_CONFIG_MODELS = {"0.6b", "1.7b", "4b"}

# ---------------------------------------------------------------- конфигурации
# имя -> (метод, параметры, группа). Группы: lit-attn, our-attn, lit-mlp, our-mlp, block, base
CONFIGS = {
    "orig": ("orig", {}, "base"),
    "drop": ("drop", {}, "base"),
    # --- внимание: литература
    "L1-window64": ("sparse", {"window": 64}, "lit-attn"),
    "L2-strided": ("strided", {"window": 32, "stride": 32}, "lit-attn"),
    "L3-performer-s0": ("performer", {"features": 256, "seed": 0}, "lit-attn"),
    "L3-performer-s1": ("performer", {"features": 256, "seed": 1}, "lit-attn"),
    "L3-performer-s2": ("performer", {"features": 256, "seed": 2}, "lit-attn"),
    "L4-linear-elu": ("linear_elu", {}, "lit-attn"),
    "L5-relpos": ("relpos", {"max_offset": 64}, "lit-attn"),
    # окно + стоки по массивной активации (не только позиция 0): у Llama-семейства сток живёт на токенах « the»
    "L1-window64s": ("sparse", {"window": 64, "sink_ratio": 2.0}, "our-attn"),
    "B-window64s+routed25r": ("sparse+routed_mlp", {"attn": {"window": 64, "sink_ratio": 2.0},
                              "mlp": {"active_ratio": 0.25, "predictor": "linear", "robust_tau": 10}}, "block"),
    "L6-a3-attn": ("a3", {"mlp": False}, "lit-attn"),
    # --- внимание: наши
    "O1-headtyped50": ("head_typed", {"frac": 0.5}, "our-attn"),
    "O2-ridgeconv": ("ridge_conv", {"window": 8, "pca_dim": 256}, "our-attn"),
    "O4-salience": ("salience", {"iters": 3}, "our-attn"),
    "O4-salience-ex25": ("salience", {"iters": 3, "exact_frac": 0.25}, "our-attn"),
    "O4-salience-ex25-w64": ("salience", {"iters": 3, "exact_frac": 0.25, "window": 64}, "our-attn"),
    # --- MLP: литература
    "ML1-a3mlp50": ("a3_mlp", {"ratio": 0.5}, "lit-mlp"),
    "ML2-topk25-dejavu-oracle": ("topk_mlp", {"ratio": 0.25}, "lit-mlp"),
    "ML4-asvd50": ("asvd_mlp", {"ratio": 0.5}, "lit-mlp"),
    "ML5-svdllm50": ("svdllm_mlp", {"ratio": 0.5}, "lit-mlp"),
    "ML7-moe32x25": ("moe_mlp", {"experts": 32, "active_frac": 0.25}, "lit-mlp"),
    # --- MLP: наши
    "MO1-ridge": ("ridge_mlp", {}, "our-mlp"),
    "MO1-ridge-keep25": ("ridge_mlp", {"keep_ratio": 0.25}, "our-mlp"),
    "MO2-table": ("table_mlp", {}, "our-mlp"),
    "MO3-cluster16": ("cluster_mlp", {"clusters": 16}, "our-mlp"),
    "MO4-routed25": ("routed_mlp", {"active_ratio": 0.25, "predictor": "linear"}, "our-mlp"),
    "MO4-routed10": ("routed_mlp", {"active_ratio": 0.10, "predictor": "linear"}, "our-mlp"),
    "O5-windowtail16": ("window_tail", {"window": 16}, "our-attn"),
    # робастная калибровка (строки с массивной активацией исключены из наименьших квадратов, см. mlp.robust_pairs)
    "MO4-routed25r": ("routed_mlp", {"active_ratio": 0.25, "predictor": "linear", "robust_tau": 10}, "our-mlp"),
    "MO4-gate25r": ("routed_mlp", {"active_ratio": 0.25, "predictor": "gate", "robust_tau": 10}, "our-mlp"),  # 2 ступени
    "MO4-quad25r": ("routed_mlp", {"active_ratio": 0.25, "predictor": "quad", "quad_rank": 256, "robust_tau": 10}, "our-mlp"),
    "MO1-ridge-keep25r": ("ridge_mlp", {"keep_ratio": 0.25, "robust_tau": 10}, "our-mlp"),
    "ML1-a3mlp50r": ("a3_mlp", {"ratio": 0.5, "robust_tau": 10}, "lit-mlp"),
    # идеи 2026-11-08: регрессия пониженного ранга, низкоранговая поправка M-O4, CATS
    "MO1-ridge-r": ("ridge_mlp", {"robust_tau": 10}, "our-mlp"),
    "MO1-ridge-r64": ("ridge_mlp", {"robust_tau": 10, "rank": 64}, "our-mlp"),
    "MO1-ridge-r128": ("ridge_mlp", {"robust_tau": 10, "rank": 128}, "our-mlp"),
    "MO1-ridge-r256": ("ridge_mlp", {"robust_tau": 10, "rank": 256}, "our-mlp"),
    "MO4-routed25r-nolin": ("routed_mlp", {"active_ratio": 0.25, "predictor": "linear", "robust_tau": 10, "lam_lin": None}, "our-mlp"),
    "MO4-routed25r-lr64": ("routed_mlp", {"active_ratio": 0.25, "predictor": "linear", "robust_tau": 10, "lin_rank": 64}, "our-mlp"),
    "MO4-routed25r-lr128": ("routed_mlp", {"active_ratio": 0.25, "predictor": "linear", "robust_tau": 10, "lin_rank": 128}, "our-mlp"),
    "MO4-routed25r-lr256": ("routed_mlp", {"active_ratio": 0.25, "predictor": "linear", "robust_tau": 10, "lin_rank": 256}, "our-mlp"),
    "ML8-cats25": ("cats_mlp", {"ratio": 0.25}, "lit-mlp"),
    # отравление калибровки у опубликованных методов: робастные SVD-LLM / ASVD (без позиции 0 и выбросов)
    "ML5-svdllm50r": ("svdllm_mlp", {"ratio": 0.5, "robust_tau": 10}, "lit-mlp"),
    "MO2-table-r": ("table_mlp", {"robust_tau": 10}, "our-mlp"),  # пункт 6: робастная таблица по токену
    "ML4-asvd50r": ("asvd_mlp", {"ratio": 0.5, "robust_tau": 10}, "lit-mlp"),
    "ML8-cats50": ("cats_mlp", {"ratio": 0.5}, "lit-mlp"),
    "L6-a3-full-r": ("a3", {"robust_tau": 10}, "block"),
    # --- блок целиком
    "L6-a3-full": ("a3", {}, "block"),
    "B-window+svdllm": ("sparse+svdllm_mlp", {"attn": {"window": 64}, "mlp": {"ratio": 0.5}}, "block"),
    "B-window+ridge-keep25": ("sparse+ridge_mlp", {"attn": {"window": 64}, "mlp": {"keep_ratio": 0.25}}, "block"),
    "B-salience+ridge-keep25": ("salience+ridge_mlp", {"attn": {"iters": 3, "exact_frac": 0.25},
                                                       "mlp": {"keep_ratio": 0.25}}, "block"),
    "O3-ngram": ("ngram", {"m": 3, "sweeps": 2}, "block"),
    "B-window+MO4-25": ("sparse+routed_mlp", {"attn": {"window": 64},
                                              "mlp": {"active_ratio": 0.25, "predictor": "linear"}}, "block"),
    "B-O1+MO4-25": ("head_typed+routed_mlp", {"attn": {"frac": 0.5},
                                              "mlp": {"active_ratio": 0.25, "predictor": "linear"}}, "block"),
    "B-window+MO4-25r": ("sparse+routed_mlp", {"attn": {"window": 64},
                                               "mlp": {"active_ratio": 0.25, "predictor": "linear", "robust_tau": 10}}, "block"),
    "B-O1+MO4-25r": ("head_typed+routed_mlp", {"attn": {"frac": 0.5},
                                               "mlp": {"active_ratio": 0.25, "predictor": "linear", "robust_tau": 10}}, "block"),
}

# калибровка больше для табличных методов
CALIB_TOKENS = {"ngram": 4_000_000, "table_mlp": 2_000_000}

# Mac (M4, 16 ГБ): 4B/8B не помещаются в fp32; оценка укорочена, бенчи — подвыборки.
# Значения подобраны по замеру скорости MPS (см. PLAN.md, журнал).
MAC_PRESET = {
    "models": "0.6b,1.7b", "calib_tokens": 131_072, "eval_windows": 32, "eval_len": 1024,
    "limit": 200, "mmlu_limit": 5, "profile_tokens": 65_536, "distill_steps": 300,
    "distill_models": "0.6b", "lm_only_main": True, "lm_bs": "8", "mmlu_bs": 2,
}

# «Сколько слоёв можно заменить»: лучшие методы каждого вида, k до ~половины глубины, KL/PPL
HOWMANY_K = [1, 2, 3, 4, 6, 8, 12]
HOWMANY_K_DEEP = [1, 2, 3, 4, 6, 8, 12, 18]  # 36-слойные модели (4B, 8B)
HOWMANY_K_64 = [1, 2, 3, 4, 6, 8, 12, 18, 24, 32]  # 64 слоя (32B); для последовательной калибровки — до 12
SEQ_K_CAP_BIG = 12  # у моделей вне FULL_CONFIG_MODELS методы с последовательной калибровкой (routed/ridge) — до k=12


def howmany_ks(model: str) -> list[int]:
    if model not in ("0.6b", "1.7b", "4b", "8b", "14b", "32b"):  # второе семейство на Mac — до k=8
        return [1, 2, 3, 4, 6, 8]
    p = RESULTS / model / "profile.json"
    L = len(json.loads(p.read_text()).get("resid_norm_pos0", [])) if p.exists() else 28
    return HOWMANY_K_64 if L >= 64 else (HOWMANY_K_DEEP if L >= 36 else HOWMANY_K)


def skip_big(model: str, method: str, k: int) -> bool:
    return model not in FULL_CONFIG_MODELS and k > SEQ_K_CAP_BIG and ("routed" in method or "ridge" in method)
HOWMANY = {
    "L1-window64": ("sparse", {"window": 64}),
    "L1-window64s": ("sparse", {"window": 64, "sink_ratio": 2.0}),
    "O1-headtyped50": ("head_typed", {"frac": 0.5}),
    "ML2-topk25-dejavu-oracle": ("topk_mlp", {"ratio": 0.25}),
    "MO4-routed25": ("routed_mlp", {"active_ratio": 0.25, "predictor": "linear"}),
    "ML5-svdllm50": ("svdllm_mlp", {"ratio": 0.5}),
    "B-window64+routed25": ("sparse+routed_mlp", {"attn": {"window": 64},
                                                  "mlp": {"active_ratio": 0.25, "predictor": "linear"}}),
    "B-window64+svdllm": ("sparse+svdllm_mlp", {"attn": {"window": 64}, "mlp": {"ratio": 0.5}}),
    "MO4-routed25r": ("routed_mlp", {"active_ratio": 0.25, "predictor": "linear", "robust_tau": 10}),
    "ML8-cats25": ("cats_mlp", {"ratio": 0.25}),
    "ML5-svdllm50r": ("svdllm_mlp", {"ratio": 0.5, "robust_tau": 10}),
    "B-window64+routed25r": ("sparse+routed_mlp", {"attn": {"window": 64},
                                                   "mlp": {"active_ratio": 0.25, "predictor": "linear", "robust_tau": 10}}),
    "B-window64s+routed25r": ("sparse+routed_mlp", {"attn": {"window": 64, "sink_ratio": 2.0},
                                                    "mlp": {"active_ratio": 0.25, "predictor": "linear", "robust_tau": 10}}),
    "B-window64+topk25": ("sparse+topk_mlp", {"attn": {"window": 64},
                                             "mlp": {"ratio": 0.25, "protect_sinks": False}}),  # диагностика: окно + оракул
}
ROBUST_HOWMANY = ["MO4-routed25r", "B-window64+routed25r"]
ROBUST_FINAL = ["MO4-routed25r", "B-window+MO4-25r", "MO1-ridge-keep25r"]
EXTRAS = [("ML1-a3mlp50r", False), ("L6-a3-full-r", True)]
IDEAS = [("MO1-ridge-r", False), ("MO1-ridge-r64", False), ("MO1-ridge-r128", False), ("MO1-ridge-r256", False),
         ("MO4-routed25r-nolin", False), ("MO4-routed25r-lr64", False), ("MO4-routed25r-lr128", False),
         ("MO4-routed25r-lr256", False), ("ML8-cats25", True), ("ML8-cats50", False)]
IDEAS_HOWMANY = ["ML8-cats25"]
NOVELTY = [("ML5-svdllm50r", True), ("ML4-asvd50r", False), ("MO2-table-r", False)]  # ловушка у опубликованных методов; таблица робастн.
NOVELTY_HOWMANY = ["ML5-svdllm50r"]  # (конфигурация, бенчи?) — проверка A3 на ту же уязвимость

# сокращённый howmany для моделей кроме 0.6B (решение пользователя 2026-11-07): 3 метода, k до 8
HOWMANY_OTHER = ["L1-window64", "MO4-routed25", "MO4-routed25r", "ML2-topk25-dejavu-oracle",
                 "O1-headtyped50", "ML5-svdllm50", "B-window64+routed25r", "B-window64+svdllm",
                 "L1-window64s", "B-window64s+routed25r", "B-window64+topk25"]  # полный набор записки
HOWMANY_OTHER_K = [1, 2, 3, 4, 6, 8]
# бенчмарки с большой подвыборкой для главных столбцов: 1000 примеров на задачу, MMLU 20 на подзадачу
BENCH_MAIN = ["orig", "L1-window64", "MO4-routed25", "ML5-svdllm50", "B-window+MO4-25", "MO4-routed25r", "B-window+MO4-25r"]
BENCH_LIMIT, BENCH_MMLU_LIMIT = 1000, 20

# основные столбцы итоговой таблицы (для 4B/8B — только они)
FINAL_MAIN = ["orig", "L1-window64", "L1-window64s", "L5-relpos", "L6-a3-full", "ML5-svdllm50", "B-window+svdllm",
              "O1-headtyped50", "MO4-routed25", "B-window+MO4-25", "B-O1+MO4-25"]

KSWEEP = [1, 2, 3, 4, 6]

# дистилляционный трек (отдельные строки таблицы, помечены «+дист.»)
DISTILL = ["L4-linear-elu", "O2-ridgeconv", "ML5-svdllm50", "MO1-ridge", "O4-salience-ex25-w64",
           "B-window+ridge-keep25"]
DISTILL_STEPS = 1000

PARETO = {  # (база конфигурации, параметр, значения)
    "a3_mlp": ("a3_mlp", "ratio", [0.1, 0.25, 0.5, 0.75]),
    "asvd_mlp": ("asvd_mlp", "ratio", [0.1, 0.25, 0.5, 0.75]),
    "svdllm_mlp": ("svdllm_mlp", "ratio", [0.1, 0.25, 0.5, 0.75]),
    "topk_mlp": ("topk_mlp", "ratio", [0.05, 0.1, 0.25, 0.5]),
    "moe_mlp": ("moe_mlp", "active_frac", [0.125, 0.25, 0.5]),
    "ridge_mlp": ("ridge_mlp", "keep_ratio", [0.0, 0.05, 0.1, 0.25, 0.5]),
    "sparse": ("sparse", "window", [8, 16, 32, 64, 128]),
    "salience": ("salience", "exact_frac", [0.0, 0.25, 0.5, 0.75]),
    "head_typed": ("head_typed", "frac", [0.25, 0.5, 0.75]),
    "routed_mlp": ("routed_mlp", "active_ratio", [0.05, 0.1, 0.25, 0.5], {"predictor": "linear"}),
    # робастная калибровка (наивные варианты выше отравлены строкой с массивной активацией на 0.6B)
    "routed_mlp_r": ("routed_mlp", "active_ratio", [0.05, 0.1, 0.25, 0.5], {"predictor": "linear", "robust_tau": 10}),
    "ridge_mlp_r": ("ridge_mlp", "keep_ratio", [0.0, 0.05, 0.1, 0.25, 0.5], {"robust_tau": 10}),
}


# ---------------------------------------------------------------- запуск

def k_sink(model: str) -> int:
    """Число первых слоёв до появления массивной активации на позиции 0 (из профиля)."""
    p = RESULTS / model / "profile.json"
    if not p.exists():
        return 3
    prof = json.loads(p.read_text())
    h0, rest = prof["resid_norm_pos0"], prof["resid_norm_other"]
    for i, (a, b) in enumerate(zip(h0, rest)):
        if a > 100 * max(b, 1e-6):
            return max(1, i)
    return 3


def run(model, name, method, params, k, out, lm: bool, args, extra=()):
    if out.exists() and not args.force:
        print(f"[skip] {out.relative_to(ROOT)}")
        return
    cmd = [sys.executable, str(ROOT / "run.py"), "--model", model, "--method", method, "--k", str(k),
           "--params", json.dumps(params), "--out", str(out), "--tag", name,
           *(["--dtype", "bf16", "--calib-bs", "4"] if model in BF16_MODELS else []),
           *(["--calib-bs", "4"] if model in SMALL_BATCH_MODELS else []),  # MHA-модели: память MPS при k=8
           "--calib-tokens", str(int(CALIB_TOKENS.get(method.split("+")[-1], args.calib_tokens)
                                     * (args.calib_tokens / 262_144 if method.split("+")[-1] in CALIB_TOKENS else 1))),
           "--eval-windows", str(args.eval_windows), "--eval-len", str(args.eval_len),
           "--lm-bs", str(args.lm_bs), "--mmlu-bs", str(args.mmlu_bs)]
    if lm:
        cmd += ["--tasks", "default", "--limit", str(args.limit), "--mmlu-limit", str(args.mmlu_limit)]
    else:
        cmd += ["--tasks", "none"]
    if model in SMALL_BATCH_MODELS:
        cmd = [c if c != str(args.lm_bs) or cmd[i - 1] != "--lm-bs" else "4" for i, c in enumerate(cmd)]
        if "--mmlu-bs" in cmd:
            cmd[cmd.index("--mmlu-bs") + 1] = "1"
    cmd += list(extra)
    print(f"[run] {' '.join(cmd)}", flush=True)
    if args.dry:
        return
    log = out.with_suffix(".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    for attempt, extra_args in enumerate(([], ["--calib-bs", "2"])):  # сторож: зависание на MPS → повтор с меньшим батчем
        t0 = time.time()
        with open(log, "w" if attempt == 0 else "a") as f:
            proc = subprocess.Popen(cmd + extra_args, stdout=f, stderr=subprocess.STDOUT)
            try:
                rc = proc.wait(timeout=RUN_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait(); rc = "TIMEOUT"
        print(f"      rc={rc}  {time.time() - t0:.0f}s  log={log.relative_to(ROOT)}" + (f"  (попытка {attempt + 1})" if attempt else ""), flush=True)
        if rc != "TIMEOUT":
            break


def stage_profile(args):
    for m in args.models:
        out = RESULTS / m / "profile.json"
        if out.exists() and not args.force:
            print(f"[skip] {out.relative_to(ROOT)}")
            continue
        cmd = [sys.executable, str(ROOT / "profile_layers.py"), "--model", m, "--tokens", str(args.profile_tokens)]
        print("[run]", " ".join(cmd), flush=True)
        if not args.dry:
            subprocess.call(cmd)


def stage_ksweep(args):
    m = "0.6b"
    for name, (method, params, group) in CONFIGS.items():
        if name == "orig" or name in ("L3-performer-s1", "L3-performer-s2"):
            continue
        for k in KSWEEP:
            run(m, name, method, params, k, RESULTS / m / "ksweep" / f"{name}_k{k}.json", False, args)


def stage_pareto(args):
    for m in (["0.6b"] if args.preset == "mac" else args.models):
        k = k_sink(m)
        for fam, (method, key, values, *base) in PARETO.items():
            for v in values:
                params = {**(base[0] if base else {}), key: v}
                name = f"{fam}-{key}{v}"
                run(m, name, method, params, k, RESULTS / m / "pareto" / f"{name}_k{k}.json", False, args)


def stage_final(args):
    for m in args.models:
        full = (m == "0.6b") if args.preset == "mac" else (m in FULL_CONFIG_MODELS)
        names = list(CONFIGS) if full else FINAL_MAIN + ROBUST_FINAL
        ks = [k_sink(m)]  # зависимость от k — в этапе ksweep
        for name in names:
            method, params, group = CONFIGS[name]
            if name == "orig":
                run(m, name, method, params, 0, RESULTS / m / "final" / "orig.json", True, args)
                continue
            lm = (name in FINAL_MAIN or name in ROBUST_FINAL) or not args.lm_only_main
            for k in ks:
                run(m, name, method, params, k, RESULTS / m / "final" / f"{name}_k{k}.json", lm, args)


def stage_howmany(args):
    """Полный набор методов howmany; на Mac — только 0.6B, на H100 — все модели из --models."""
    models = ["0.6b"] if args.preset == "mac" else args.models
    for m in models:
        for k in howmany_ks(m):  # сначала малые k для всех методов — полезный результат раньше
            for name, (method, params) in HOWMANY.items():
                if skip_big(m, method, k):
                    continue
                run(m, name, method, params, k, RESULTS / m / "howmany" / f"{name}_k{k}.json", False, args)


def stage_howmany_other(args):
    """Сокращённый «сколько слоёв» для моделей кроме 0.6B."""
    for m in args.models:
        if m == "0.6b":
            continue
        for k in HOWMANY_OTHER_K:
            for name in HOWMANY_OTHER:
                method, params = HOWMANY[name]
                run(m, name, method, params, k, RESULTS / m / "howmany" / f"{name}_k{k}.json", False, args)


def stage_bench1000(args):
    """Главные столбцы с бенчами на большой подвыборке (results/<model>/bench1000/)."""
    extra = ["--limit", str(BENCH_LIMIT), "--mmlu-limit", str(BENCH_MMLU_LIMIT)]
    for m in args.models:
        k = k_sink(m)
        for name in BENCH_MAIN:
            method, params, group = CONFIGS[name]
            kk = 0 if name == "orig" else k
            out = RESULTS / m / "bench1000" / (f"{name}.json" if name == "orig" else f"{name}_k{k}.json")
            run(m, name, method, params, kk, out, True, args, extra=extra)


def stage_robust(args):
    """Робастная калибровка: кривые howmany и столбцы таблицы с бенчами."""
    for m in (["0.6b"] if args.preset == "mac" else args.models):
        for k in howmany_ks(m):
            for name in ROBUST_HOWMANY:
                method, params = HOWMANY[name]
                if skip_big(m, method, k):
                    continue
                run(m, name, method, params, k, RESULTS / m / "howmany" / f"{name}_k{k}.json", False, args)
    for mm in args.models:
        k = k_sink(mm)
        for name in ROBUST_FINAL:
            method, params, group = CONFIGS[name]
            run(mm, name, method, params, k, RESULTS / mm / "final" / f"{name}_k{k}.json", True, args)


def stage_extras(args):
    """Дополнительные столбцы (робастный A3) при k_sink."""
    for m in args.models:
        k = k_sink(m)
        for name, lm in EXTRAS:
            method, params, group = CONFIGS[name]
            run(m, name, method, params, k, RESULTS / m / "final" / f"{name}_k{k}.json", lm, args)


def stage_ideas(args):
    """Идеи 2026-11-08: ранг линейной части, низкоранговая поправка M-O4, CATS (k_sink) + кривая CATS."""
    for m in (["0.6b"] if args.preset == "mac" else args.models):
        k = k_sink(m)
        for name, lm in IDEAS:
            method, params, group = CONFIGS[name]
            run(m, name, method, params, k, RESULTS / m / "final" / f"{name}_k{k}.json", lm, args)
        for kk in howmany_ks(m):
            for name in IDEAS_HOWMANY:
                method, params = HOWMANY[name]
                run(m, name, method, params, kk, RESULTS / m / "howmany" / f"{name}_k{kk}.json", False, args)


def stage_novelty(args):
    """Робастные SVD-LLM/ASVD при k_sink (+ кривая SVD-LLM робастн.)."""
    for m in (["0.6b"] if args.preset == "mac" else args.models):
        k = k_sink(m)
        for name, lm in NOVELTY:
            method, params, group = CONFIGS[name]
            run(m, name, method, params, k, RESULTS / m / "final" / f"{name}_k{k}.json", lm, args)
        for kk in howmany_ks(m):
            for name in NOVELTY_HOWMANY:
                method, params = HOWMANY[name]
                run(m, name, method, params, kk, RESULTS / m / "howmany" / f"{name}_k{kk}.json", False, args)


def stage_distill(args):
    for m in args.models:
        if m not in args.distill_models:
            continue
        k = k_sink(m)
        for name in DISTILL:
            method, params, group = CONFIGS[name]
            run(m, name + "+distill", method, params, k, RESULTS / m / "final" / f"{name}+distill_k{k}.json",
                True, args, extra=["--distill-steps", str(args.distill_steps)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["profile", "ksweep", "pareto", "final", "distill", "howmany",
                                                       "howmany_other", "bench1000", "robust", "extras", "ideas", "novelty", "all"])
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--calib-tokens", type=int, default=262_144)
    ap.add_argument("--eval-windows", type=int, default=0, help="0 = весь WikiText-2 test")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--mmlu-limit", type=int, default=50)
    ap.add_argument("--lm-bs", default="auto:4")
    ap.add_argument("--mmlu-bs", type=int, default=0)
    ap.add_argument("--eval-len", type=int, default=2048)
    ap.add_argument("--profile-tokens", type=int, default=131_072)
    ap.add_argument("--distill-steps", type=int, default=DISTILL_STEPS)
    ap.add_argument("--distill-models", default="0.6b,1.7b")
    ap.add_argument("--lm-only-main", action="store_true",
                    help="бенчи только для столбцов итоговой таблицы, остальные методы — KL/PPL")
    ap.add_argument("--preset", choices=["h100", "mac"], default="h100",
                    help="mac: 0.6B+1.7B, укороченная оценка и подвыборки бенчей (M4, 16 ГБ)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    for key, val in (MAC_PRESET if args.preset == "mac" else H100_PRESET).items():
        if getattr(args, key) == ap.get_default(key):
            setattr(args, key, val)
    args.models = args.models.split(",")
    args.distill_models = args.distill_models.split(",")
    # порядок «all»: сначала то, что нужно для таблицы и записки, потом дополнительные треки
    # ksweep исключён из «all» 2026-11-08 (дублирует howmany; 165 прогонов) — запускается только явно
    if args.stage != "all":
        stages = [args.stage]
    elif args.preset == "h100":
        stages = ["profile", "final", "robust", "extras", "ideas", "novelty", "howmany", "distill", "pareto"]
    else:
        stages = ["profile", "final", "robust", "extras", "ideas", "howmany_other", "bench1000", "distill", "pareto"]
    for st in stages:
        print(f"===== stage {st}", flush=True)
        globals()[f"stage_{st}"](args)


if __name__ == "__main__":
    main()
