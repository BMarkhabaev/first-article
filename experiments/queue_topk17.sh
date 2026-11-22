#!/bin/bash
# Qwen3-1.7B: блок «окно 64 + отбор 25% по активациям» (B-window64+topk25) для рисунка слайда, затем k=8 у
# двух других блоков. Ждёт окончания кривых MNN-Pocket (процесс sweep howmany_other), чтобы не делить память.
# Запуск: nohup bash queue_topk17.sh <pid> > logs/queue_topk17.sh.out 2>&1 &
cd "$(dirname "$0")" || exit 1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LOG=logs/novelty.log
WAIT=${1:-}
[ -n "$WAIT" ] && while kill -0 "$WAIT" 2>/dev/null; do sleep 60; done
echo "$(date '+%F %T') topk17: старт" >> $LOG
R=results/1.7b/howmany
COMMON="--calib-tokens 131072 --eval-windows 32 --eval-len 1024 --lm-bs 8 --mmlu-bs 2 --tasks none"
one() {  # имя метод параметры k
  local out=$R/$1_k$4.json
  [ -f "$out" ] && return
  caffeinate -i -s .venv/bin/python -u run.py --model 1.7b --method $2 --k $4 --params "$3" --out $out --tag $1 \
    $COMMON > ${out%.json}.log 2>&1
  echo "$(date '+%F %T') topk17: $1 k=$4 rc=$?" >> $LOG
}
for k in 1 2 3 4 6 8; do
  one B-window64+topk25 sparse+topk_mlp '{"attn": {"window": 64}, "mlp": {"ratio": 0.25, "protect_sinks": false}}' $k
done
one B-window64+routed25r sparse+routed_mlp '{"attn": {"window": 64}, "mlp": {"active_ratio": 0.25, "predictor": "linear", "robust_tau": 10}}' 8
one B-window64+svdllm sparse+svdllm_mlp '{"attn": {"window": 64}, "mlp": {"ratio": 0.5}}' 8
echo "$(date '+%F %T') topk17: всё готово" >> $LOG
