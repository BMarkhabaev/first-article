#!/bin/bash
# Табличное внимание: MNN-Pocket (все режимы), затем Qwen3-0.6B и 1.7B (слой 0 и все слои). Идёт параллельно queue_pocket.sh.
cd "$(dirname "$0")" || exit 1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LOG=logs/novelty.log
for M in pocket0.3b 0.6b 1.7b; do
  mkdir -p results/$M/table_attn
  for MODE in layer0 alllayers; do
    caffeinate -i -s .venv/bin/python -u table_attention.py $MODE --model $M > results/$M/table_attn/$MODE.log 2>&1
    echo "$(date '+%F %T') table: $M $MODE rc=$?" >> $LOG
  done
done
caffeinate -i -s .venv/bin/python -u table_attention.py bench --model pocket0.3b > results/pocket0.3b/table_attn/bench.log 2>&1
echo "$(date '+%F %T') table: bench pocket rc=$?" >> $LOG
caffeinate -i -s .venv/bin/python -u table_attention.py bench --model 0.6b --devices mps > results/0.6b/table_attn/bench.log 2>&1
echo "$(date '+%F %T') table: всё готово" >> $LOG
