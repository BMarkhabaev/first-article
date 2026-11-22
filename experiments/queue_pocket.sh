#!/bin/bash
# MNN-Pocket-0.3B (архитектура Qwen3): полный протокол статьи. Запуск: nohup bash queue_pocket.sh > logs/queue_pocket.sh.out 2>&1 &
cd "$(dirname "$0")" || exit 1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LOG=logs/novelty.log
M=pocket0.3b
mkdir -p results/$M/criterion results/$M/cliff results/$M/trap results/$M/sink
echo "$(date '+%F %T') pocket: старт" >> $LOG
caffeinate -i -s .venv/bin/python -u smoke_family.py --model $M > logs/smoke_$M.log 2>&1
grep -E "^(OK|FAIL|attn)" logs/smoke_$M.log >> $LOG
caffeinate -i -s .venv/bin/python -u sweep.py --preset mac --stage profile --models $M >> logs/sweep_mac_pocket.log 2>&1
echo "$(date '+%F %T') pocket: профиль rc=$?" >> $LOG
caffeinate -i -s .venv/bin/python -u sweep.py --preset mac --stage final --models $M >> logs/sweep_mac_pocket.log 2>&1
echo "$(date '+%F %T') pocket: таблица rc=$?" >> $LOG
caffeinate -i -s .venv/bin/python -u sweep.py --preset mac --stage howmany_other --models $M >> logs/sweep_mac_pocket.log 2>&1
echo "$(date '+%F %T') pocket: кривые rc=$?" >> $LOG
caffeinate -i -s .venv/bin/python -u analyze_profile.py --model $M > results/$M/criterion/profile.log 2>&1
caffeinate -i -s .venv/bin/python -u planner.py --fit 0.6b --test $M > results/$M/criterion/planner.log 2>&1
caffeinate -i -s .venv/bin/python -u sink_neurons.py --model $M > results/$M/sink/neurons.log 2>&1
caffeinate -i -s .venv/bin/python -u toggle_trap.py --model $M > results/$M/trap/toggle.log 2>&1
caffeinate -i -s .venv/bin/python -u analyze_cliff.py --model $M --configs routed3,routed2,routed4,window3 --correctors-for "" > results/$M/cliff/analysis.log 2>&1
caffeinate -i -s .venv/bin/python -u mlp_mechanism.py --model $M > results/$M/criterion/mlp_mechanism.log 2>&1
caffeinate -i -s .venv/bin/python -u cats_depth.py --model $M > results/$M/criterion/cats_depth.log 2>&1
caffeinate -i -s .venv/bin/python -u leverage.py --model $M > results/$M/trap/leverage.log 2>&1
echo "$(date '+%F %T') pocket: всё готово" >> $LOG
