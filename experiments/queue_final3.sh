#!/bin/bash
# Приоритет пользователя (12.11 14:05): досчитать SmolLM2; Llama — в самом конце.
cd "$(dirname "$0")" || exit 1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LOG=logs/novelty.log
while kill -0 31095 2>/dev/null || pgrep -f "run.py --model smol1.7b --method sparse" >/dev/null; do sleep 30; done
echo "$(date '+%F %T') final3: smol window64s готовы; добор кривых SmolLM2" >> $LOG
caffeinate -i -s .venv/bin/python -u sweep.py --preset mac --stage howmany_other --models smol1.7b >> logs/sweep_mac_final_rest.log 2>&1
echo "$(date '+%F %T') final3: smol howmany_other rc=$?" >> $LOG
caffeinate -i -s .venv/bin/python -u sweep.py --preset mac --stage final --models smol1.7b >> logs/sweep_mac_final_rest.log 2>&1
echo "$(date '+%F %T') final3: smol final (недостающие конфигурации) rc=$?" >> $LOG
caffeinate -i -s .venv/bin/python -u sweep.py --preset mac --stage howmany_other --models 1.7b >> logs/sweep_mac_final_rest.log 2>&1
for M in 0.6b 1.7b; do
  caffeinate -i -s .venv/bin/python -u static_late_mlp.py --model $M > results/$M/criterion/static_late.log 2>&1
  echo "$(date '+%F %T') final3: static_late $M rc=$?" >> $LOG
  caffeinate -i -s .venv/bin/python -u activation_reuse.py --model $M > results/$M/criterion/activation_reuse.log 2>&1
  echo "$(date '+%F %T') final3: activation_reuse $M rc=$?" >> $LOG
done
caffeinate -i -s .venv/bin/python -u sweep.py --preset mac --stage howmany_other --models llama1b >> logs/sweep_mac_final_llama.log 2>&1
echo "$(date '+%F %T') final3: всё готово (Llama последней)" >> $LOG
