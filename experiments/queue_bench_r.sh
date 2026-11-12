#!/bin/bash
# Бенчи на 1000 для робастных столбцов 0.6B (BENCH_MAIN пополнен после старта основного прогона) — после fix17.
cd "$(dirname "$0")" || exit 1
until ! pgrep -f "queue_fix17.sh" >/dev/null; do sleep 120; done
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "$(date '+%F %T') bench_r: бенчи 1000 для робастных столбцов 0.6B" >> logs/novelty.log
caffeinate -i -s .venv/bin/python -u sweep.py --preset mac --stage bench1000 --models 0.6b >> logs/sweep_mac_bench_r.log 2>&1
.venv/bin/python make_report.py >> logs/novelty.log 2>&1
echo "$(date '+%F %T') bench_r: готово" >> logs/novelty.log
