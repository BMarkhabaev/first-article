#!/bin/bash
# После queue_final4: точки 0.6B для «окно 64 + массивные стоки + M-O4» (этап howmany, словарь HOWMANY).
cd "$(dirname "$0")" || exit 1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LOG=logs/novelty.log
while pgrep -f "queue_final4.sh" >/dev/null; do sleep 60; done
echo "$(date '+%F %T') final5: 0.6b howmany (добор B-window64s+routed25r)" >> $LOG
caffeinate -i -s .venv/bin/python -u sweep.py --preset mac --stage howmany --models 0.6b >> logs/sweep_mac_final_rest.log 2>&1
echo "$(date '+%F %T') final5: готово rc=$?" >> $LOG
