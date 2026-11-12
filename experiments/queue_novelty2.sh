#!/bin/bash
# Пункты 1–3 (вторая волна): теория отравления через рычаг строки, нейроны стока, CATS по глубине — после a3fix.
cd "$(dirname "$0")" || exit 1
until ! pgrep -f "queue_a3fix.sh" >/dev/null; do sleep 120; done
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LOG=logs/novelty.log
echo "$(date '+%F %T') novelty2: старт" >> $LOG
mkdir -p results/0.6b/trap results/1.7b/trap results/0.6b/sink results/1.7b/sink
for m in 0.6b 1.7b; do
  caffeinate -i -s .venv/bin/python -u leverage.py --model $m > results/$m/trap/leverage.log 2>&1
  caffeinate -i -s .venv/bin/python -u sink_neurons.py --model $m > results/$m/sink/neurons.log 2>&1
done
caffeinate -i -s .venv/bin/python -u cats_depth.py --model 0.6b > results/0.6b/criterion/cats_depth.log 2>&1
[ -f results/1.7b/criterion/profile.json ] && caffeinate -i -s .venv/bin/python -u cats_depth.py --model 1.7b > results/1.7b/criterion/cats_depth.log 2>&1
echo "$(date '+%F %T') novelty2: завершено" >> $LOG
