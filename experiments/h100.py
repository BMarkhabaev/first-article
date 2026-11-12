#!/usr/bin/env python
"""Все прогоны статьи на машине с несколькими GPU (офлайн). Задания независимы: один run.py — одна карта.

  python h100.py --models 0.6b,1.7b,4b,8b --gpus 8             # всё: профили → прогоны → анализы
  python h100.py --models 4b --stages final,howmany --plan      # только показать задания
Переменные окружения: MODELS_DIR (веса, по умолчанию /app/models), HF_HOME (кэш датасетов lm-eval), офлайн-флаги.
Готовые JSON пропускаются, можно перезапускать. Журналы: logs/h100/<задание>.log, сводка logs/h100/summary.log."""
from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = ["final", "robust", "extras", "ideas", "novelty", "howmany", "distill", "pareto"]
ANALYSES = ["analyze_cliff.py --model {m} --configs routed3,routedr3,routed2,routed4,window3,svdllm3 "
            "--correctors-for routed3,routedr3 --robust-corr-tau 10",
            "analyze_profile.py --model {m}", "ghost_blocks.py --model {m}", "eval_corpus.py --model {m}",
            "eval_long.py --model {m} --study sinks --k 3", "eval_long.py --model {m} --study sinks --k 6",
            "eval_long.py --model {m} --study length --k 3", "bench_time.py --model {m}",
            "hessian_outlier.py --model {m}", "toggle_trap.py --model {m}", "error_structure.py --model {m}",
            "leverage.py --model {m}", "sink_neurons.py --model {m}", "cats_depth.py --model {m}",
            "mlp_mechanism.py --model {m}"]
ENV = {"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}


def sweep_jobs(model: str, stage: str) -> list[str]:
    r = subprocess.run([sys.executable, "sweep.py", "--preset", "h100", "--models", model, "--stage", stage, "--dry"],
                       capture_output=True, text=True, cwd=ROOT, env={**os.environ, **ENV})
    return [l[6:] for l in r.stdout.splitlines() if l.startswith("[run] ")]


def run_pool(jobs: list[str], gpus: list[str], log_dir: Path, plan: bool):
    log_dir.mkdir(parents=True, exist_ok=True)
    if plan or not jobs:
        return
    q = queue.Queue()
    for i, j in enumerate(jobs):
        q.put((i, j))
    summary = open(log_dir / "summary.log", "a")
    lock = threading.Lock()

    def worker(gpu):
        while True:
            try:
                i, cmd = q.get_nowait()
            except queue.Empty:
                return
            tag = f"{i:04d}"
            t0 = time.time()
            with open(log_dir / f"{tag}.log", "w") as f:
                f.write(cmd + "\n")
                f.flush()
                rc = subprocess.call(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT,
                                     env={**os.environ, **ENV, "CUDA_VISIBLE_DEVICES": gpu})
            with lock:
                line = f"{time.strftime('%F %T')} gpu{gpu} rc={rc} {time.time() - t0:.0f}s {tag} {cmd[:160]}"
                summary.write(line + "\n")
                summary.flush()
                print(line, flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="4b,8b,32b")
    ap.add_argument("--gpus", default="8", help="число карт или список id через запятую")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--no-analyses", action="store_true")
    ap.add_argument("--plan", action="store_true")
    args = ap.parse_args()
    models = args.models.split(",")
    gpus = [str(i) for i in range(int(args.gpus))] if args.gpus.isdigit() else args.gpus.split(",")
    log_dir = ROOT / "logs" / "h100"
    # 1) профили (нужны для k_sink и глубины) — по одному заданию на модель
    prof = [j for m in models for j in sweep_jobs(m, "profile")]
    print(f"профили: {len(prof)} заданий", flush=True)
    run_pool(prof, gpus, log_dir, args.plan)
    # 2) прогоны всех этапов — общая очередь на все карты
    jobs = [j for m in models for st in args.stages.split(",") for j in sweep_jobs(m, st)]
    print(f"прогоны: {len(jobs)} заданий на {len(gpus)} карт", flush=True)
    if args.plan:
        for j in jobs[:5]:
            print("   ", j[:200])
    run_pool(jobs, gpus, log_dir, args.plan)
    # 3) анализы и дополнительные скрипты
    if not args.no_analyses:
        extra = [f"{sys.executable} -u {a.format(m=m)}" for m in models for a in ANALYSES]
        extra += [f"{sys.executable} -u planner.py --fit {models[0]} --test {m}" for m in models]  # план с меньшей модели
        print(f"анализы: {len(extra)} заданий", flush=True)
        run_pool(extra, gpus, log_dir, args.plan)
    if not args.plan:
        for s in ("make_report.py", "make_brief.py"):
            subprocess.call([sys.executable, s], cwd=ROOT)
    print("готово", flush=True)


if __name__ == "__main__":
    main()
