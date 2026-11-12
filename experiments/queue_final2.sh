#!/bin/bash
# После точек SmolLM2 window64s (ветка B queue_final.sh, pid 31095): Llama кривые → добор smol/1.7b → идеи 4/6.
cd "$(dirname "$0")" || exit 1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LOG=logs/novelty.log
while kill -0 31095 2>/dev/null || pgrep -f "run.py --model smol1.7b --method sparse" >/dev/null; do sleep 30; done
echo "$(date '+%F %T') final2: smol window64s готовы, старт Llama кривых" >> $LOG
caffeinate -i -s .venv/bin/python -u sweep.py --preset mac --stage howmany_other --models llama1b >> logs/sweep_mac_final_llama.log 2>&1
echo "$(date '+%F %T') final2: Llama howmany_other готово rc=$?" >> $LOG
caffeinate -i -s .venv/bin/python -u sweep.py --preset mac --stage howmany_other --models smol1.7b >> logs/sweep_mac_final_rest.log 2>&1
caffeinate -i -s .venv/bin/python -u sweep.py --preset mac --stage howmany_other --models 1.7b >> logs/sweep_mac_final_rest.log 2>&1
for M in 0.6b 1.7b; do
  caffeinate -i -s .venv/bin/python -u static_late_mlp.py --model $M > results/$M/criterion/static_late.log 2>&1
  echo "$(date '+%F %T') final2: static_late $M rc=$?" >> $LOG
  caffeinate -i -s .venv/bin/python -u activation_reuse.py --model $M > results/$M/criterion/activation_reuse.log 2>&1
  echo "$(date '+%F %T') final2: activation_reuse $M rc=$?" >> $LOG
done
echo "$(date '+%F %T') final2: всё готово" >> $LOG
