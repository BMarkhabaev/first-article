#!/bin/bash
# Повторять загрузку Llama-3.2-1B каждые 10 минут, пока Meta не одобрит доступ (до 12 часов).
cd "$(dirname "$0")" || exit 1
for i in $(seq 1 72); do
  if .venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('meta-llama/Llama-3.2-1B', allow_patterns=['*.json','*.safetensors','*.txt','*.model'])" >> logs/download_llama.log 2>&1; then
    echo "$(date '+%F %T') Llama-3.2-1B скачана" >> logs/novelty.log; exit 0; fi
  sleep 600
done
echo "$(date '+%F %T') Llama-3.2-1B: доступ так и не одобрен за 12 ч" >> logs/novelty.log
