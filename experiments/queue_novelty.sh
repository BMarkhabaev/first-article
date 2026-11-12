#!/bin/bash
# Пункты 1–4 по новизне (после resume.sh, последовательно): робастные SVD-LLM/ASVD, гессиан, переключатель ловушки,
# природа ошибок, планировщик 0.6B→1.7B.
cd "$(dirname "$0")" || exit 1
until ! pgrep -f "resume.sh" >/dev/null; do sleep 120; done
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LOG=logs/novelty.log
echo "$(date '+%F %T') novelty: старт" >> $LOG
caffeinate -i -s .venv/bin/python -u sweep.py --preset mac --stage novelty >> logs/sweep_mac_novelty.log 2>&1
mkdir -p results/0.6b/hessian results/0.6b/trap results/1.7b/trap
caffeinate -i -s .venv/bin/python -u hessian_outlier.py --model 0.6b > results/0.6b/hessian/outlier.log 2>&1
caffeinate -i -s .venv/bin/python -u toggle_trap.py --model 0.6b > results/0.6b/trap/toggle.log 2>&1
caffeinate -i -s .venv/bin/python -u toggle_trap.py --model 1.7b > results/1.7b/trap/toggle.log 2>&1
caffeinate -i -s .venv/bin/python -u error_structure.py --model 0.6b > results/0.6b/criterion/error_structure.log 2>&1
caffeinate -i -s .venv/bin/python -u planner.py --fit 0.6b --test 1.7b > results/1.7b/criterion/planner.log 2>&1
caffeinate -i -s .venv/bin/python -u planner.py --fit 0.6b --test 0.6b > results/0.6b/criterion/planner.log 2>&1
.venv/bin/python make_report.py >> $LOG 2>&1
echo "$(date '+%F %T') novelty: завершено" >> $LOG
