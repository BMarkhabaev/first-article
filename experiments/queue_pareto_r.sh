#!/bin/bash
# Парето для робастных M-O4 и M-O1 (9 прогонов, 0.6B) — после fixdistill.
cd "$(dirname "$0")" || exit 1
until ! pgrep -f "queue_fixdistill.sh" >/dev/null; do sleep 120; done
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "$(date '+%F %T') pareto_r: робастные семейства Парето" >> logs/novelty.log
caffeinate -i -s .venv/bin/python -u sweep.py --preset mac --stage pareto >> logs/sweep_mac_pareto_r.log 2>&1
.venv/bin/python make_report.py >> logs/novelty.log 2>&1
echo "$(date '+%F %T') pareto_r: готово" >> logs/novelty.log
