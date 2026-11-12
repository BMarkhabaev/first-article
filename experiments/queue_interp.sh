#!/bin/bash
# Пункт 6: робастная таблица (в этапе novelty уже добавлена; здесь — если не посчитана) и анализ покрытия — после rigor.
cd "$(dirname "$0")" || exit 1
until ! pgrep -f "queue_rigor.sh" >/dev/null; do sleep 120; done
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "$(date '+%F %T') interp: робастная таблица + анализ покрытия" >> logs/novelty.log
caffeinate -i -s .venv/bin/python -u sweep.py --preset mac --stage novelty >> logs/sweep_mac_novelty.log 2>&1
caffeinate -i -s .venv/bin/python -u table_analysis.py --model 0.6b > results/0.6b/criterion/table_analysis.log 2>&1
.venv/bin/python make_report.py >> logs/novelty.log 2>&1
echo "$(date '+%F %T') interp: готово" >> logs/novelty.log
