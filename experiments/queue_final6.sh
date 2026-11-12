#!/bin/bash
# 12.11 14:45. Порядок: (текущий добор smol) → smol final → MLP-идеи (gate/quad/table) на smol и 0.6b →
# 1.7b howmany_other → идеи 4/6 → 0.6b howmany → Llama (последней).
cd "$(dirname "$0")" || exit 1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LOG=logs/novelty.log
SM="--calib-bs 4 --calib-tokens 131072 --eval-windows 32 --eval-len 1024 --eval-bs 1 --lm-bs 4 --mmlu-bs 1 --tasks none"
Q6="--calib-tokens 131072 --eval-windows 32 --eval-len 1024 --eval-bs 1 --tasks none"
while pgrep -f "sweep.py --preset mac --stage howmany_other --models smol1.7b" >/dev/null; do sleep 60; done
echo "$(date '+%F %T') final6: добор smol готов; smol final (недостающие)" >> $LOG
caffeinate -i -s .venv/bin/python -u sweep.py --preset mac --stage final --models smol1.7b >> logs/sweep_mac_final_rest.log 2>&1
echo "$(date '+%F %T') final6: smol final rc=$?" >> $LOG
mkdir -p results/smol1.7b/mlp_ideas results/0.6b/mlp_ideas
for K in 1 2 3; do
  for PRED in gate quad; do
    OUT=results/smol1.7b/mlp_ideas/MO4-${PRED}25r_k$K.json; [ -f "$OUT" ] && continue
    P="{\"active_ratio\": 0.25, \"predictor\": \"$PRED\", \"robust_tau\": 10, \"quad_rank\": 256}"
    caffeinate -i -s .venv/bin/python -u run.py --model smol1.7b --method routed_mlp --k $K --params "$P" $SM --out $OUT > ${OUT%.json}.log 2>&1
    echo "$(date '+%F %T') final6: smol $PRED k=$K rc=$?" >> $LOG
  done
done
for K in 1 2; do
  OUT=results/smol1.7b/mlp_ideas/MO2-table-r_k$K.json; [ -f "$OUT" ] && continue
  caffeinate -i -s .venv/bin/python -u run.py --model smol1.7b --method table_mlp --k $K --params '{"robust_tau": 10}' \
    --calib-bs 4 --calib-tokens 2000000 --eval-windows 32 --eval-len 1024 --eval-bs 1 --tasks none --out $OUT > ${OUT%.json}.log 2>&1
  echo "$(date '+%F %T') final6: smol table-r k=$K rc=$?" >> $LOG
done
for PRED in gate quad; do
  OUT=results/0.6b/mlp_ideas/MO4-${PRED}25r_k3.json; [ -f "$OUT" ] && continue
  P="{\"active_ratio\": 0.25, \"predictor\": \"$PRED\", \"robust_tau\": 10, \"quad_rank\": 256}"
  caffeinate -i -s .venv/bin/python -u run.py --model 0.6b --method routed_mlp --k 3 --params "$P" $Q6 --out $OUT > ${OUT%.json}.log 2>&1
  echo "$(date '+%F %T') final6: 0.6b $PRED k=3 rc=$?" >> $LOG
done
caffeinate -i -s .venv/bin/python -u sweep.py --preset mac --stage howmany_other --models 1.7b >> logs/sweep_mac_final_rest.log 2>&1
for M in 0.6b 1.7b; do
  caffeinate -i -s .venv/bin/python -u static_late_mlp.py --model $M > results/$M/criterion/static_late.log 2>&1
  echo "$(date '+%F %T') final6: static_late $M rc=$?" >> $LOG
  caffeinate -i -s .venv/bin/python -u activation_reuse.py --model $M > results/$M/criterion/activation_reuse.log 2>&1
  echo "$(date '+%F %T') final6: activation_reuse $M rc=$?" >> $LOG
done
caffeinate -i -s .venv/bin/python -u sweep.py --preset mac --stage howmany --models 0.6b >> logs/sweep_mac_final_rest.log 2>&1
caffeinate -i -s .venv/bin/python -u sweep.py --preset mac --stage howmany_other --models llama1b >> logs/sweep_mac_final_llama.log 2>&1
echo "$(date '+%F %T') final6: всё готово (Llama последней)" >> $LOG
