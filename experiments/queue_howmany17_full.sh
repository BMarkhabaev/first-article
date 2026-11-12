#!/bin/bash
# Недостающие кривые 1.7B (O1, SVD-LLM, окно+M-O4r, окно+SVD-LLM, k≤8) → пересборка записки.
cd "$(dirname "$0")" || exit 1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "$(date '+%F %T') howmany17_full: старт" >> logs/novelty.log
caffeinate -i -s .venv/bin/python -u sweep.py --preset mac --stage howmany_other --models 1.7b >> logs/sweep_mac_howmany17.log 2>&1
.venv/bin/python make_brief.py >> logs/novelty.log 2>&1
.venv/bin/python make_report.py >> logs/novelty.log 2>&1
(cd ../brief && tectonic -X compile --bundle https://data1.fullyjustified.net/tlextras-2022.0r0.tar brief.tex > /dev/null 2>&1)
(cd .. && tectonic -X compile --bundle https://data1.fullyjustified.net/tlextras-2022.0r0.tar main.tex > /dev/null 2>&1)
echo "$(date '+%F %T') howmany17_full: готово, записка и статья пересобраны" >> logs/novelty.log
