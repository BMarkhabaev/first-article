#!/bin/bash
# Больше алгоритмов на кривых слайда: HOWMANY_MORE у 0.6B → MNN-Pocket → 1.7B (медленные
# M-O1 из HOWMANY_MORE_SLOW сняты); после каждого шага — пересборка обоих слайдов (refresh_slides.sh).
# Ждёт очередь queue_topk17.sh (pid в $1). Запуск: nohup bash queue_more.sh <pid> > logs/queue_more.sh.out 2>&1 &
cd "$(dirname "$0")" || exit 1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LOG=logs/novelty.log
WAIT=${1:-}
[ -n "$WAIT" ] && while kill -0 "$WAIT" 2>/dev/null; do sleep 60; done
for ST in howmany_more; do  # M-O1 (howmany_more_slow) снят 2026-11-14: при k=3 вдвое–впятеро хуже M-O4
  for M in 0.6b pocket0.3b 1.7b; do
    echo "$(date '+%F %T') more: $ST $M старт" >> $LOG
    caffeinate -i -s .venv/bin/python -u sweep.py --preset mac --stage $ST --models $M >> logs/sweep_mac_more.log 2>&1
    echo "$(date '+%F %T') more: $ST $M rc=$?" >> $LOG
    bash refresh_slides.sh "$ST $M"
  done
done
echo "$(date '+%F %T') more: всё готово" >> $LOG
