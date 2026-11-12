#!/bin/bash
# Второе семейство (SmolLM2-1.7B, архитектура Llama): профиль → таблица → кривые → анализы; + механизм MLP для 1.7B.
cd "$(dirname "$0")" || exit 1
until ! pgrep -f "queue_howmany17_full" >/dev/null; do sleep 120; done
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LOG=logs/novelty.log
# точки k=8 для «окно+M-O4» и «окно+SVD-LLM» на 1.7B не считаем: 13 ГБ памяти, процесс зависает в свопе (21:36)
.venv/bin/python make_brief.py >> $LOG 2>&1
(cd ../brief && tectonic -X compile --bundle https://data1.fullyjustified.net/tlextras-2022.0r0.tar brief.tex > /dev/null 2>&1; tectonic -X compile --bundle https://data1.fullyjustified.net/tlextras-2022.0r0.tar slide.tex > /dev/null 2>&1)
echo "$(date '+%F %T') добор 1.7B и пересборка записки/слайда готовы" >> $LOG
echo "$(date '+%F %T') smol: механизм MLP 1.7B" >> $LOG
grep -q "Спирмен" results/1.7b/criterion/mlp_mechanism.log 2>/dev/null || caffeinate -i -s .venv/bin/python -u mlp_mechanism.py --model 1.7b > results/1.7b/criterion/mlp_mechanism.log 2>&1
for M in smol1.7b llama1b; do
if [ "$M" = "llama1b" ] && ! ls ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/*/model.safetensors >/dev/null 2>&1; then
  echo "$(date '+%F %T') второе семейство: веса Llama-3.2-1B ещё не скачаны (доступ не одобрен) — пропуск" >> $LOG; continue; fi
echo "$(date '+%F %T') второе семейство: старт $M" >> $LOG
mkdir -p results/$M/cliff results/$M/criterion results/$M/trap results/$M/sink
caffeinate -i -s .venv/bin/python -u smoke_family.py --model $M > logs/smoke_$M.log 2>&1
grep -E "^(OK|FAIL|attn)" logs/smoke_$M.log >> $LOG
caffeinate -i -s .venv/bin/python -u sweep.py --preset mac --stage profile --models $M >> logs/sweep_mac_smol.log 2>&1
caffeinate -i -s .venv/bin/python -u sweep.py --preset mac --stage final --models $M >> logs/sweep_mac_smol.log 2>&1
caffeinate -i -s .venv/bin/python -u sweep.py --preset mac --stage howmany_other --models $M >> logs/sweep_mac_smol.log 2>&1
caffeinate -i -s .venv/bin/python -u analyze_profile.py --model $M > results/$M/criterion/profile.log 2>&1
caffeinate -i -s .venv/bin/python -u analyze_cliff.py --model $M --configs routed3,routed2,routed4,window3 --correctors-for "" > results/$M/cliff/analysis.log 2>&1
caffeinate -i -s .venv/bin/python -u toggle_trap.py --model $M > results/$M/trap/toggle.log 2>&1
caffeinate -i -s .venv/bin/python -u sink_neurons.py --model $M > results/$M/sink/neurons.log 2>&1
caffeinate -i -s .venv/bin/python -u cats_depth.py --model $M > results/$M/criterion/cats_depth.log 2>&1
caffeinate -i -s .venv/bin/python -u planner.py --fit 0.6b --test $M > results/$M/criterion/planner.log 2>&1
caffeinate -i -s .venv/bin/python -u mlp_mechanism.py --model $M > results/$M/criterion/mlp_mechanism.log 2>&1
caffeinate -i -s .venv/bin/python -u leverage.py --model $M > results/$M/trap/leverage.log 2>&1
echo "$(date '+%F %T') второе семейство: $M готово" >> $LOG
done
.venv/bin/python make_report.py >> $LOG 2>&1
echo "$(date '+%F %T') второе семейство: всё готово" >> $LOG
