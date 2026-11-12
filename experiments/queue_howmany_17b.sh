#!/bin/bash
# Кривые «сколько слоёв» на 1.7B (в т.ч. робастный M-O4) — после очереди анализов, чтобы не плодить процессы.
cd "$(dirname "$0")" || exit 1
until ! pgrep -f "queue_after_robust" >/dev/null; do sleep 120; done
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "$(date '+%F %T') старт howmany_other 1.7B" >> logs/queue_after_robust.log
caffeinate -i -s .venv/bin/python -u sweep.py --preset mac --stage howmany_other --models 1.7b >> logs/sweep_mac_howmany17.log 2>&1
.venv/bin/python make_brief.py >> logs/queue_after_robust.log 2>&1
echo "$(date '+%F %T') howmany_other 1.7B завершён, записка пересчитана" >> logs/queue_after_robust.log
