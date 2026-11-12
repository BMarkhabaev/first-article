#!/bin/bash
# Окно + стоки по массивности (L1-window64s, B-window64s+routed25r): добор точек после цепочки второго семейства.
cd "$(dirname "$0")" || exit 1
until ! pgrep -f "queue_smol.sh" >/dev/null; do sleep 120; done
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LOG=logs/novelty.log
echo "$(date '+%F %T') sinks: старт (окно + стоки по массивности)" >> $LOG
for M in smol1.7b llama1b 1.7b 0.6b; do
  if [ "$M" = "llama1b" ] && [ ! -f results/llama1b/profile.json ]; then continue; fi
  caffeinate -i -s .venv/bin/python -u sweep.py --preset mac --stage final --models $M >> logs/sweep_mac_sinks.log 2>&1
  if [ "$M" = "0.6b" ]; then
    caffeinate -i -s .venv/bin/python -u sweep.py --preset mac --stage howmany --models $M >> logs/sweep_mac_sinks.log 2>&1
  else
    caffeinate -i -s .venv/bin/python -u sweep.py --preset mac --stage howmany_other --models $M >> logs/sweep_mac_sinks.log 2>&1
  fi
  echo "$(date '+%F %T') sinks: $M готово" >> $LOG
done
.venv/bin/python make_report.py >> $LOG 2>&1
.venv/bin/python make_brief.py >> $LOG 2>&1
(cd ../brief && tectonic -X compile --bundle https://data1.fullyjustified.net/tlextras-2022.0r0.tar brief.tex > /dev/null 2>&1; tectonic -X compile --bundle https://data1.fullyjustified.net/tlextras-2022.0r0.tar slide.tex > /dev/null 2>&1)
echo "$(date '+%F %T') sinks: всё готово" >> $LOG
