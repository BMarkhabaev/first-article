#!/bin/bash
# Калибровка на втором корпусе (pile-10k) → оценка на pile-10k и WikiText: переобучение под домен — после pareto_r.
cd "$(dirname "$0")" || exit 1
until ! pgrep -f "queue_pareto_r.sh" >/dev/null; do sleep 120; done
export TRANSFORMERS_OFFLINE=1
echo "$(date '+%F %T') calibdom: калибровка на pile-10k (0.6B)" >> logs/novelty.log
caffeinate -i -s .venv/bin/python -u eval_corpus.py --model 0.6b --calib pile10k --configs MO4-routed25r,ML5-svdllm50,B-window+MO4-25r > logs/eval_corpus_calibpile_0.6b.log 2>&1
echo "$(date '+%F %T') calibdom: готово" >> logs/novelty.log
