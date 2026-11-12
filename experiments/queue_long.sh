#!/bin/bash
# Стоки и длина контекста (0.6B) — после кривых 1.7B, чтобы не множить процессы.
cd "$(dirname "$0")" || exit 1
until ! pgrep -f "queue_howmany_17b" >/dev/null; do sleep 120; done
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p results/0.6b/long
echo "$(date '+%F %T') старт eval_long (стоки, длина) 0.6B" >> logs/queue_after_robust.log
caffeinate -i -s .venv/bin/python -u eval_long.py --model 0.6b --study sinks --k 3 > results/0.6b/long/sinks_k3.log 2>&1
caffeinate -i -s .venv/bin/python -u eval_long.py --model 0.6b --study sinks --k 6 > results/0.6b/long/sinks_k6.log 2>&1
caffeinate -i -s .venv/bin/python -u eval_long.py --model 0.6b --study length --k 3 > results/0.6b/long/length_k3.log 2>&1
echo "$(date '+%F %T') eval_long завершён" >> logs/queue_after_robust.log
