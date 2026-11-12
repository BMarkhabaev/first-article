#!/bin/bash
# К дедлайну: кривые Llama (A) параллельно с точками SmolLM2 «окно + стоки по массивности» (B); затем остальное.
cd "$(dirname "$0")" || exit 1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LOG=logs/novelty.log
echo "$(date '+%F %T') final: старт (Llama кривые + SmolLM2 window64s)" >> $LOG
( caffeinate -i -s .venv/bin/python -u sweep.py --preset mac --stage howmany_other --models llama1b >> logs/sweep_mac_final_llama.log 2>&1
  echo "$(date '+%F %T') final: Llama howmany_other готово rc=$?" >> $LOG ) &
PA=$!
( for K in 3 6 2 4 8; do
    OUT=results/smol1.7b/howmany/L1-window64s_k$K.json; [ -f "$OUT" ] && continue
    caffeinate -i -s .venv/bin/python -u run.py --model smol1.7b --method sparse --k $K --params '{"window": 64, "sink_ratio": 2.0}' \
      --dtype fp32 --calib-bs 4 --calib-tokens 131072 --eval-windows 32 --eval-len 1024 --eval-bs 1 --lm-bs 4 --mmlu-bs 1 --tasks none \
      --out $OUT > ${OUT%.json}.log 2>&1
    echo "$(date '+%F %T') final: smol window64s k=$K rc=$?" >> $LOG
  done ) &
PB=$!
wait $PA $PB
echo "$(date '+%F %T') final: обе ветки готовы; дальше добор smol/1.7b и идеи 4/6" >> $LOG
caffeinate -i -s .venv/bin/python -u sweep.py --preset mac --stage howmany_other --models smol1.7b >> logs/sweep_mac_final_rest.log 2>&1
caffeinate -i -s .venv/bin/python -u sweep.py --preset mac --stage howmany_other --models 1.7b >> logs/sweep_mac_final_rest.log 2>&1
for M in 0.6b 1.7b; do
  caffeinate -i -s .venv/bin/python -u static_late_mlp.py --model $M > results/$M/criterion/static_late.log 2>&1
  echo "$(date '+%F %T') final: static_late $M rc=$?" >> $LOG
  caffeinate -i -s .venv/bin/python -u activation_reuse.py --model $M > results/$M/criterion/activation_reuse.log 2>&1
  echo "$(date '+%F %T') final: activation_reuse $M rc=$?" >> $LOG
done
echo "$(date '+%F %T') final: всё готово" >> $LOG
