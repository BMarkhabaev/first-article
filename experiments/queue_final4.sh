#!/bin/bash
# Приоритет пользователя (12.11 14:05): 1) smol window64s все k; 2) 1.7b window64s все k; затем остальное, Llama последней.
cd "$(dirname "$0")" || exit 1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LOG=logs/novelty.log
while kill -0 31095 2>/dev/null || pgrep -f "run.py --model smol1.7b --method sparse" >/dev/null; do sleep 20; done
echo "$(date '+%F %T') final4: smol window64s готовы; 1.7b window64s" >> $LOG
for K in 3 6 2 4 8; do
  OUT=results/1.7b/howmany/L1-window64s_k$K.json; [ -f "$OUT" ] && continue
  caffeinate -i -s .venv/bin/python -u run.py --model 1.7b --method sparse --k $K --params '{"window": 64, "sink_ratio": 2.0}' \
    --dtype fp32 --calib-tokens 131072 --eval-windows 32 --eval-len 1024 --eval-bs 1 --tasks none \
    --out $OUT > ${OUT%.json}.log 2>&1
  echo "$(date '+%F %T') final4: 1.7b window64s k=$K rc=$?" >> $LOG
done
caffeinate -i -s .venv/bin/python -u sweep.py --preset mac --stage howmany_other --models smol1.7b >> logs/sweep_mac_final_rest.log 2>&1
echo "$(date '+%F %T') final4: smol howmany_other rc=$?" >> $LOG
caffeinate -i -s .venv/bin/python -u sweep.py --preset mac --stage final --models smol1.7b >> logs/sweep_mac_final_rest.log 2>&1
echo "$(date '+%F %T') final4: smol final (недостающие) rc=$?" >> $LOG
caffeinate -i -s .venv/bin/python -u sweep.py --preset mac --stage howmany_other --models 1.7b >> logs/sweep_mac_final_rest.log 2>&1
for M in 0.6b 1.7b; do
  caffeinate -i -s .venv/bin/python -u static_late_mlp.py --model $M > results/$M/criterion/static_late.log 2>&1
  echo "$(date '+%F %T') final4: static_late $M rc=$?" >> $LOG
  caffeinate -i -s .venv/bin/python -u activation_reuse.py --model $M > results/$M/criterion/activation_reuse.log 2>&1
  echo "$(date '+%F %T') final4: activation_reuse $M rc=$?" >> $LOG
done
caffeinate -i -s .venv/bin/python -u sweep.py --preset mac --stage howmany_other --models llama1b >> logs/sweep_mac_final_llama.log 2>&1
echo "$(date '+%F %T') final4: всё готово (Llama последней)" >> $LOG
