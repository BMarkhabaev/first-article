#!/bin/bash
# Повтор упавшей дистилляции O4-salience (tap.ids не был выставлен в цикле distill) — после novelty2.
cd "$(dirname "$0")" || exit 1
until ! pgrep -f "queue_novelty2.sh" >/dev/null; do sleep 120; done
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "$(date '+%F %T') fixdistill: повтор O4-salience+distill" >> logs/novelty.log
caffeinate -i -s .venv/bin/python -u sweep.py --preset mac --stage distill >> logs/sweep_mac_fixdistill.log 2>&1
.venv/bin/python make_report.py >> logs/novelty.log 2>&1
echo "$(date '+%F %T') fixdistill: готово" >> logs/novelty.log
