#!/bin/bash
# Робастные столбцы 1.7B были посчитаны этапом final без бенчмарков (lm_only_main): пересчитать с бенчами
# после очереди по новизне (последовательно, чтобы не было свопа).
cd "$(dirname "$0")" || exit 1
until ! pgrep -f "queue_novelty.sh" >/dev/null; do sleep 120; done
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "$(date '+%F %T') fix17: пересчёт робастных столбцов 1.7B с бенчами" >> logs/novelty.log
for n in MO4-routed25r B-window+MO4-25r MO1-ridge-keep25r; do
  f="results/1.7b/final/${n}_k3.json"
  [ -f "$f" ] && ! grep -q '"lm_eval"' "$f" && rm -f "$f"
done
caffeinate -i -s .venv/bin/python -u sweep.py --preset mac --stage final --models 1.7b >> logs/sweep_mac_fix17.log 2>&1
.venv/bin/python make_brief.py >> logs/novelty.log 2>&1
.venv/bin/python make_report.py >> logs/novelty.log 2>&1
echo "$(date '+%F %T') fix17: готово" >> logs/novelty.log
