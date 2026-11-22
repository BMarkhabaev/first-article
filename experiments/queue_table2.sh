#!/bin/bash
# Табличное внимание, перезапуск после исправления: пропуски словаря считаются точно, слой 0 — точная таблица.
cd "$(dirname "$0")" || exit 1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LOG=logs/novelty.log
run() { caffeinate -i -s .venv/bin/python -u table_attention.py $1 --model $2 ${3:-} > results/$2/table_attn/$1.log 2>&1; echo "$(date '+%F %T') table: $2 $1 rc=$?" >> $LOG; }
mkdir -p results/1.7b/table_attn
run alllayers pocket0.3b
run alllayers 0.6b
run layer0 1.7b
run alllayers 1.7b
run bench pocket0.3b
run bench 0.6b "--devices mps"
echo "$(date '+%F %T') table: всё готово" >> $LOG
