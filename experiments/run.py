#!/usr/bin/env python
"""Один эксперимент: модель × метод × k → JSON в results/.

Примеры:
  python run.py --model 0.6b --method orig --tasks default
  python run.py --model 0.6b --method ridge_conv --k 2 --params '{"window": 8}'
  python run.py --model 0.6b --method performer --k 2 --params '{"seed": 1}' --no-lm-eval
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from detlayers.common import (RESULTS_DIR, Timer, calibration_windows, env_info, load_model, pick_device,
                              save_json, seed_all, wikitext_windows)
from detlayers.methods import build
from detlayers.metrics import compare_lm, run_lm_eval, summarize_lm_eval

DEFAULT_TASKS = ["lambada_openai", "hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande", "mmlu"]
FEWSHOT = {"mmlu": 5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="0.6b")
    ap.add_argument("--method", default="orig")
    ap.add_argument("--k", type=int, default=0, help="число заменяемых первых слоёв")
    ap.add_argument("--params", default="{}", help="JSON с гиперпараметрами метода")
    ap.add_argument("--calib-tokens", type=int, default=262_144)
    ap.add_argument("--calib-len", type=int, default=512)
    ap.add_argument("--calib-bs", type=int, default=8)
    ap.add_argument("--eval-len", type=int, default=2048)
    ap.add_argument("--eval-windows", type=int, default=0, help="0 = весь WikiText-2 test")
    ap.add_argument("--eval-bs", type=int, default=1)
    ap.add_argument("--tasks", default="default", help="'default', 'none' или список через запятую")
    ap.add_argument("--limit", type=float, default=0, help="lm-eval limit (0 = полный бенч)")
    ap.add_argument("--lm-bs", default="auto:4", help="размер пакета lm-eval: число или 'auto' (подбор по памяти)")
    ap.add_argument("--lm-max-len", type=int, default=4096, help="макс. длина контекста lm-eval (MMLU 5-shot > 2048)")
    ap.add_argument("--mmlu-bs", type=int, default=0, help="отдельный размер пакета для MMLU (0 = как --lm-bs)")
    ap.add_argument("--mmlu-limit", type=int, default=50, help="вопросов на подзадачу MMLU (0 = все)")
    ap.add_argument("--lm-dtype", default="bf16", choices=["fp32", "bf16"],
                    help="точность для lm-eval (KL/PPL всегда в --dtype)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp16"],
                    help="fp32 по умолчанию: KL-метрики без шума низкой точности")
    ap.add_argument("--tag", default="")
    ap.add_argument("--distill-steps", type=int, default=0, help=">0 — дистилляционный трек (не детерминированный)")
    ap.add_argument("--distill-lr", type=float, default=1e-4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seed_all(0)
    device = pick_device(args.device)
    params = json.loads(args.params)
    model, tok = load_model(args.model, device, args.dtype)

    tag = args.tag or (json.dumps(params, sort_keys=True).replace(" ", "").replace('"', "")
                       .replace("{", "").replace("}", "").replace(":", "=").replace(",", "_"))
    name = f"{args.model}_{args.method}_k{args.k}" + (f"_{tag}" if tag else "")
    out_path = Path(args.out) if args.out else RESULTS_DIR / args.model / f"{name}.json"

    result = {"model": args.model, "method": args.method, "k": args.k, "params": params,
              "env": env_info(device), "args": vars(args)}

    calib = None
    no_calib = {"orig", "drop", "sparse", "strided", "performer", "linear_elu", "topk_mlp", "moe_mlp"}
    if not set(args.method.split("+")) <= no_calib:
        calib = calibration_windows(tok, args.calib_tokens, args.calib_len, seed=0)
        result["calib"] = {"windows": calib.shape[0], "len": calib.shape[1]}
    with Timer() as t:
        rep, info = build(model, args.method, args.k, calib, device, bs=args.calib_bs, **params)
    result["build"] = {"seconds": round(t.dt, 1), **info}
    costs = {f"{i}.{a}": getattr(m, "extra_cost", lambda: {})() for (i, a), m in rep.new.items()}
    result["build"]["modules"] = {k: v for k, v in costs.items() if v}
    if args.distill_steps > 0:
        from detlayers.distill import distill
        if calib is None:
            calib = calibration_windows(tok, args.calib_tokens, args.calib_len, seed=0)
        with Timer() as t:
            result["distill"] = distill(model, rep, calib, args.k, device, args.distill_steps,
                                        args.distill_lr, bs=args.calib_bs)
        result["distill"]["seconds"] = round(t.dt, 1)

    wins = wikitext_windows(tok, "wikitext-2-raw-v1", "test", args.eval_len,
                            max_windows=args.eval_windows or None)
    if device.type == "mps":
        import gc; gc.collect(); torch.mps.empty_cache()
    with Timer() as t:
        result["lm"] = compare_lm(model, rep, wins, device, bs=args.eval_bs,
                                  probe_layer=args.k if args.k > 0 else None)
    result["lm"]["seconds"] = round(t.dt, 1)
    print(json.dumps(result["lm"], ensure_ascii=False))

    tasks = DEFAULT_TASKS if args.tasks == "default" else ([] if args.tasks == "none" else args.tasks.split(","))
    if tasks:
        save_json(result, out_path)  # PPL/KL сохраняются до бенчмарков: падение lm-eval не теряет метрики
        rep.enable()
        del wins
        if device.type == "mps":
            import gc; gc.collect()
            try:
                torch.mps.empty_cache()
            except torch.OutOfMemoryError:  # асинхронно отложенная ошибка MPS после compare_lm
                pass
        if args.lm_dtype == "bf16" and args.dtype != "bf16":
            def _to_bf16():
                model.to(torch.bfloat16)
                for m in rep.old.values():  # отключённые исходные модули (k слоёв fp32) тоже держат память MPS
                    m.to(torch.bfloat16)
            try:
                _to_bf16()
            except torch.OutOfMemoryError:  # конвертация на MPS не влезла: через CPU
                print("[run] bf16 через CPU (MPS OOM при конвертации)", flush=True)
                model.to("cpu")
                for m in rep.old.values():
                    m.to("cpu")
                import gc; gc.collect()
                try:
                    torch.mps.empty_cache()
                except torch.OutOfMemoryError:
                    pass
                _to_bf16()
                model.to(device)
                for m in rep.old.values():
                    m.to(device)
        if device.type == "mps":  # fp32-копия и калибровочные тензоры остаются в кэше MPS → OOM в lm-eval
            import gc; gc.collect()
            try:
                torch.mps.empty_cache()
            except torch.OutOfMemoryError:
                pass
        limit = None if not args.limit else (int(args.limit) if args.limit >= 1 else args.limit)
        limits = {"mmlu": args.mmlu_limit or None}
        with Timer() as t:
            lm_bs = int(args.lm_bs) if str(args.lm_bs).isdigit() else args.lm_bs
            # фиксированные матрицы внимания (B×H×T×T) на длинных промптах MMLU не влезают в MPS
            if device.type == "mps" and any(m in args.method for m in ("relpos", "head_typed", "salience")):
                args.mmlu_bs = 1
            raw = run_lm_eval(model, tok, tasks, limit, lm_bs, device, FEWSHOT, limits=limits,
                              max_length=args.lm_max_len,
                              batch_sizes={"mmlu": args.mmlu_bs} if args.mmlu_bs else None)
        result["lm_eval"] = summarize_lm_eval(raw, tasks)
        result["lm_eval_raw"] = raw
        result["lm_eval_seconds"] = round(t.dt, 1)
        result["lm_eval_limit"] = {"default": limit, **limits}
        result["lm_eval_dtype"] = args.lm_dtype
        print(json.dumps(result["lm_eval"], ensure_ascii=False))

    save_json(result, out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
