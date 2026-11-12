#!/bin/bash
# Пункты 2 и 7: самогенерированная калибровка, доменные статистики, доверительные интервалы — после calibdom.
cd "$(dirname "$0")" || exit 1
until ! pgrep -f "queue_calibdom.sh" >/dev/null; do sleep 120; done
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LOG=logs/novelty.log
echo "$(date '+%F %T') rigor: самогенерированная калибровка 0.6B" >> $LOG
caffeinate -i -s .venv/bin/python -u selfgen_calib.py --model 0.6b > logs/selfgen_0.6b.log 2>&1
caffeinate -i -s .venv/bin/python -u eval_corpus.py --model 0.6b --calib selfgen_0.6b --configs MO4-routed25r,ML5-svdllm50,B-window+MO4-25r > logs/eval_corpus_calibselfgen_0.6b.log 2>&1
caffeinate -i -s .venv/bin/python -u domain_stats.py --model 0.6b > results/0.6b/criterion/domain_stats.log 2>&1
echo "$(date '+%F %T') rigor: прогоны с per_window для ДИ (0.6B, 1.7B)" >> $LOG
for m in 0.6b 1.7b; do
  mkdir -p results/$m/ci
  for cfg in "orig|orig|{}|0" "L1-window64|sparse|{\"window\": 64}|3" "MO4-routed25r|routed_mlp|{\"active_ratio\": 0.25, \"predictor\": \"linear\", \"robust_tau\": 10}|3" "ML5-svdllm50|svdllm_mlp|{\"ratio\": 0.5}|3" "B-window+MO4-25r|sparse+routed_mlp|{\"attn\": {\"window\": 64}, \"mlp\": {\"active_ratio\": 0.25, \"predictor\": \"linear\", \"robust_tau\": 10}}|3"; do
    IFS='|' read -r tag method params k <<< "$cfg"
    out="results/$m/ci/${tag}$( [ "$k" = 0 ] && echo "" || echo "_k$k" ).json"
    [ -f "$out" ] && continue
    caffeinate -i -s .venv/bin/python -u run.py --model $m --method "$method" --k $k --params "$params" --out "$out" --tag "$tag" \
        --calib-tokens 131072 --eval-windows 0 --eval-len 1024 --tasks none > "${out%.json}.log" 2>&1
  done
  .venv/bin/python ci_table.py --model $m --dirs ci,bench1000,final >> $LOG 2>&1
done
echo "$(date '+%F %T') rigor: готово" >> $LOG
