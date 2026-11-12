#!/usr/bin/env bash
# Запуск всех экспериментов на сервере с H100.
#   bash run_h100.sh setup        # один раз: окружение + скачивание моделей и данных
#   bash run_h100.sh smoke        # 5 минут: проверка, что всё работает на GPU
#   bash run_h100.sh all          # все этапы в фоне (nohup), лог в logs/sweep.log
#   bash run_h100.sh <этап>       # profile | ksweep | pareto | final | distill
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
mkdir -p logs

case "${1:-}" in
  setup)
    python3.12 -m venv .venv || python3 -m venv .venv
    $PY -m pip install -U pip
    $PY -m pip install -r requirements.txt
    $PY - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA недоступна"
print("GPU:", torch.cuda.get_device_name(0), "torch", torch.__version__)
from huggingface_hub import snapshot_download
for m in ["Qwen3-0.6B-Base", "Qwen3-1.7B-Base", "Qwen3-4B-Base", "Qwen3-8B-Base"]:
    snapshot_download(f"Qwen/{m}", allow_patterns=["*.json", "*.safetensors", "*.txt"])
snapshot_download("Salesforce/wikitext", repo_type="dataset",
                  allow_patterns=["wikitext-2-raw-v1/*", "wikitext-103-raw-v1/*"])
print("модели и данные скачаны")
EOF
    ;;
  smoke)
    $PY smoke_test.py sparse relpos a3 salience ridge_mlp svdllm_mlp
    $PY run.py --model 0.6b --method orig --tasks arc_easy,mmlu --limit 50 --mmlu-limit 2 \
        --eval-windows 4 --out logs/smoke_orig.json
    ;;
  all|profile|ksweep|pareto|final|distill)
    nohup $PY -u sweep.py --stage "$1" > "logs/sweep_$1.log" 2>&1 &
    echo "запущено в фоне, PID $!; прогресс: tail -f logs/sweep_$1.log"
    ;;
  *)
    sed -n 2,7p "$0"; exit 1 ;;
esac
