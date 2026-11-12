#!/bin/bash
# Очередь после робастного этапа: анализ обрыва и критерий на 1.7B, робастный A3, второй корпус.
cd "$(dirname "$0")" || exit 1
until ! pgrep -f "stage robust" >/dev/null; do sleep 120; done
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p results/1.7b/cliff results/1.7b/criterion
echo "$(date '+%F %T') старт анализов 1.7B" >> logs/queue_after_robust.log
caffeinate -i -s .venv/bin/python -u analyze_cliff.py --model 1.7b --configs routed3,routed2,routed4,window3 --correctors-for "" > results/1.7b/cliff/analysis.log 2>&1
caffeinate -i -s .venv/bin/python -u analyze_profile.py --model 1.7b > results/1.7b/criterion/profile.log 2>&1
echo "$(date '+%F %T') старт extras (робастный A3) 0.6B" >> logs/queue_after_robust.log
caffeinate -i -s .venv/bin/python -u sweep.py --preset mac --stage extras --models 0.6b >> logs/sweep_mac_extras.log 2>&1
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE
echo "$(date '+%F %T') старт второго корпуса 0.6B" >> logs/queue_after_robust.log
caffeinate -i -s .venv/bin/python -u eval_corpus.py --model 0.6b > logs/eval_corpus_0.6b.log 2>&1
echo "$(date '+%F %T') очередь завершена" >> logs/queue_after_robust.log
