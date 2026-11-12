#!/bin/bash
# Единая очередь после анализов 1.7B: идеи (0.6B) → робастный корректор → призраки блоков → кривые 1.7B → стоки/длина.
cd "$(dirname "$0")" || exit 1
until ! pgrep -f "queue_after_robust" >/dev/null; do sleep 120; done
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LOG=logs/queue_after_robust.log
echo "$(date '+%F %T') старт идей (0.6B): ранг, M-O4 lr, CATS" >> $LOG
caffeinate -i -s .venv/bin/python -u sweep.py --preset mac --stage ideas >> logs/sweep_mac_ideas.log 2>&1
echo "$(date '+%F %T') робастный корректор (Ghosted-операторы) 0.6B" >> $LOG
caffeinate -i -s .venv/bin/python -u analyze_cliff.py --model 0.6b --configs routed3,routedr3 --correctors-for routed3,routedr3 --robust-corr-tau 10 --out-name correctors_robust.json > results/0.6b/cliff/correctors_robust.log 2>&1
echo "$(date '+%F %T') призраки блоков 0.6B" >> $LOG
mkdir -p results/0.6b/ghost
caffeinate -i -s .venv/bin/python -u ghost_blocks.py --model 0.6b > results/0.6b/ghost/blocks.log 2>&1
echo "$(date '+%F %T') кривые 1.7B (howmany_other)" >> $LOG
caffeinate -i -s .venv/bin/python -u sweep.py --preset mac --stage howmany_other --models 1.7b >> logs/sweep_mac_howmany17.log 2>&1
.venv/bin/python make_brief.py >> $LOG 2>&1
echo "$(date '+%F %T') стоки и длина контекста 0.6B" >> $LOG
mkdir -p results/0.6b/long
caffeinate -i -s .venv/bin/python -u eval_long.py --model 0.6b --study sinks --k 3 > results/0.6b/long/sinks_k3.log 2>&1
caffeinate -i -s .venv/bin/python -u eval_long.py --model 0.6b --study sinks --k 6 > results/0.6b/long/sinks_k6.log 2>&1
caffeinate -i -s .venv/bin/python -u eval_long.py --model 0.6b --study length --k 3 > results/0.6b/long/length_k3.log 2>&1
echo "$(date '+%F %T') очередь queue_rest завершена" >> $LOG
