#!/bin/bash
# Продолжение после зависания Mac 2026-11-14 ~07:55 (две задачи на 1.7B одновременно забили память).
# Всё строго последовательно, по одной задаче. Запуск: nohup bash queue_resume.sh > logs/queue_resume.sh.out 2>&1 &
cd "$(dirname "$0")" || exit 1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LOG=logs/novelty.log
PY="caffeinate -i -s .venv/bin/python -u"
note() { echo "$(date '+%F %T') resume: $*" >> $LOG; }
R=results/1.7b/howmany
COMMON="--calib-tokens 131072 --eval-windows 32 --eval-len 1024 --lm-bs 8 --mmlu-bs 2 --tasks none"
one() {  # имя метод параметры k  (модель 1.7b)
  local out=$R/$1_k$4.json
  [ -f "$out" ] && return
  $PY run.py --model 1.7b --method $2 --k $4 --params "$3" --out $out --tag $1 $COMMON > ${out%.json}.log 2>&1
  note "1.7b $1 k=$4 rc=$?"
}
note "старт"
# 1) блок «окно 64 + отбор 25 %» на 1.7B
for k in 1 2 3 4 6 8; do
  one B-window64+topk25 sparse+topk_mlp '{"attn": {"window": 64}, "mlp": {"ratio": 0.25, "protect_sinks": false}}' $k
done
bash refresh_slides.sh "topk25 1.7b"
# 2) новые методы на кривых: 0.6B → MNN-Pocket → 1.7B
for M in 0.6b pocket0.3b 1.7b; do
  $PY sweep.py --preset mac --stage howmany_more --models $M >> logs/sweep_mac_more.log 2>&1
  note "howmany_more $M rc=$?"
  bash refresh_slides.sh "howmany_more $M"
done
# 3) k=8 у двух блоков 1.7B
one B-window64+routed25r sparse+routed_mlp '{"attn": {"window": 64}, "mlp": {"active_ratio": 0.25, "predictor": "linear", "robust_tau": 10}}' 8
one B-window64+svdllm sparse+svdllm_mlp '{"attn": {"window": 64}, "mlp": {"ratio": 0.5}}' 8
bash refresh_slides.sh "k8 1.7b"
# 4) табличное внимание: слой 0 у 1.7B, скорость, все слои 1.7B
ta() { $PY table_attention.py $1 --model $2 ${3:-} > results/$2/table_attn/$1.log 2>&1; note "table $2 $1 rc=$?"; }
mkdir -p results/1.7b/table_attn
ta layer0 1.7b
ta bench pocket0.3b
ta bench 0.6b "--devices mps"
ta alllayers 1.7b
# 5) анализы MNN-Pocket (не досчитаны из queue_pocket.sh)
M=pocket0.3b
$PY analyze_profile.py --model $M > results/$M/criterion/profile.log 2>&1
$PY planner.py --fit 0.6b --test $M > results/$M/criterion/planner.log 2>&1
$PY sink_neurons.py --model $M > results/$M/sink/neurons.log 2>&1
$PY toggle_trap.py --model $M > results/$M/trap/toggle.log 2>&1
$PY analyze_cliff.py --model $M --configs routed3,routed2,routed4,window3 --correctors-for "" > results/$M/cliff/analysis.log 2>&1
$PY mlp_mechanism.py --model $M > results/$M/criterion/mlp_mechanism.log 2>&1
$PY cats_depth.py --model $M > results/$M/criterion/cats_depth.log 2>&1
$PY leverage.py --model $M > results/$M/trap/leverage.log 2>&1
note "анализы pocket готовы"
note "всё готово"
