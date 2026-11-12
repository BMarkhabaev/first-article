#!/bin/bash
# Полный A3 (L6-a3-full, L6-a3-full-r) пересчитать с точной позицией 0 — после bench_r.
cd "$(dirname "$0")" || exit 1
until ! pgrep -f "queue_bench_r.sh" >/dev/null; do sleep 120; done
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "$(date '+%F %T') a3fix: пересчёт полного A3 с ExactPos0" >> logs/novelty.log
mkdir -p results/old_a3 && for m in 0.6b 1.7b; do for n in L6-a3-full L6-a3-full-r; do
  [ -f results/$m/final/${n}_k3.json ] && mv results/$m/final/${n}_k3.json results/old_a3/${m}_${n}_k3.json; done; done
caffeinate -i -s .venv/bin/python -u sweep.py --preset mac --stage final --models 0.6b,1.7b >> logs/sweep_mac_a3fix.log 2>&1
caffeinate -i -s .venv/bin/python -u sweep.py --preset mac --stage extras --models 0.6b,1.7b >> logs/sweep_mac_a3fix.log 2>&1
.venv/bin/python make_report.py >> logs/novelty.log 2>&1
echo "$(date '+%F %T') a3fix: готово" >> logs/novelty.log
