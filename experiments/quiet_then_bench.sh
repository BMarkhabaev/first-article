#!/bin/bash
# Замер времени — когда Mac свободен: 5 минут подряд без прогонов/анализов/сторожа.
cd "$(dirname "$0")" || exit 1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
quiet=0
while [ "$quiet" -lt 5 ]; do
  if pgrep -f "sweep.py|run.py --model|analyze_|eval_corpus|pipeline_guard|make_brief" >/dev/null; then quiet=0; else quiet=$((quiet + 1)); fi
  sleep 60
done
echo "$(date '+%F %T') Mac свободен, замер времени" >> logs/guard.log
caffeinate -i -s .venv/bin/python -u bench_time.py --model 0.6b > logs/bench_time_0.6b.log 2>&1
caffeinate -i -s .venv/bin/python -u bench_time.py --model 1.7b > logs/bench_time_1.7b.log 2>&1
echo "$(date '+%F %T') замер времени завершён" >> logs/guard.log
