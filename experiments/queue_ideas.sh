#!/bin/bash
# Идеи 4/6/8 — строго после Llama (queue_smol.sh) и добора стоков (queue_sinks.sh).
# Запуск: nohup bash queue_ideas.sh > logs/queue_ideas.sh.out 2>&1 &
cd "$(dirname "$0")" || exit 1
until ! pgrep -f "queue_smol.sh" >/dev/null && ! pgrep -f "queue_sinks.sh" >/dev/null; do sleep 120; done
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LOG=logs/novelty.log
echo "$(date '+%F %T') ideas: старт (4 статическая разреженность, 6 переиспользование, 8 SVD родные настройки)" >> $LOG
for M in 0.6b 1.7b; do
  mkdir -p results/$M/criterion results/$M/native_svd
  # 4. статическая разреженность поздних MLP
  caffeinate -i -s .venv/bin/python -u static_late_mlp.py --model $M > results/$M/criterion/static_late.log 2>&1
  echo "$(date '+%F %T') ideas: static_late $M rc=$?" >> $LOG
  # 6. переиспользование активаций соседних MLP
  caffeinate -i -s .venv/bin/python -u activation_reuse.py --model $M > results/$M/criterion/activation_reuse.log 2>&1
  echo "$(date '+%F %T') ideas: activation_reuse $M rc=$?" >> $LOG
done
# 8. SVD-LLM / ASVD на родных настройках: сжатие 20 % (ratio 0.8), калибровка 256 окон по 2048 токенов,
#    с робастным фильтром строк и без; первые k = k_sink слоёв, как в таблицах.
for M in 0.6b 1.7b; do
  K=$(.venv/bin/python -c "import sweep; print(sweep.k_sink('$M'))")
  for KIND in svdllm asvd; do
    for RATIO in 0.8 0.9; do
      for TAU in none 10; do
        if [ "$TAU" = "none" ]; then P="{\"ratio\": $RATIO}"; TAG="naive"; else P="{\"ratio\": $RATIO, \"robust_tau\": $TAU}"; TAG="robust"; fi
        OUT=results/$M/native_svd/${KIND}${RATIO}_${TAG}_k${K}.json
        [ -f "$OUT" ] && continue
        caffeinate -i -s .venv/bin/python -u run.py --model $M --method ${KIND}_mlp --k $K --params "$P" \
          --calib-tokens 524288 --calib-len 2048 --calib-bs 1 --eval-windows 32 --eval-len 1024 --eval-bs 1 \
          --tasks none --out $OUT > ${OUT%.json}.log 2>&1
        echo "$(date '+%F %T') ideas: $KIND $RATIO $TAG $M k=$K rc=$?" >> $LOG
      done
    done
  done
done
echo "$(date '+%F %T') ideas: всё готово" >> $LOG
