#!/bin/bash
# Возобновление после остановки 2026-11-08 17:40. Всё последовательно, без сторожей; готовые JSON пропускаются.
# Запуск (отвязанно от терминала):  cd experiments && nohup bash resume.sh > logs/resume.out 2>&1 &
cd "$(dirname "$0")" || exit 1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LOG=logs/resume.log
echo "$(date '+%F %T') resume: старт" >> $LOG
# 0) проверка новых путей кода (ранг, lin_rank, CATS) на крошечной калибровке — обязательно до этапа ideas
caffeinate -i -s .venv/bin/python -u smoke_ideas.py > logs/smoke_ideas.log 2>&1 || { echo "$(date '+%F %T') smoke_ideas УПАЛ — см. logs/smoke_ideas.log" >> $LOG; exit 1; }
# 1) все этапы прогона (новый порядок: profile final robust extras ideas howmany_other bench1000 distill pareto)
caffeinate -i -s .venv/bin/python -u sweep.py --preset mac --stage all >> logs/sweep_mac_all3.log 2>&1
.venv/bin/python make_brief.py >> $LOG 2>&1
# 2) анализы и дополнительные скрипты
caffeinate -i -s .venv/bin/python -u analyze_cliff.py --model 1.7b --configs window3 --correctors-for "" --out-name analysis_window3.json > results/1.7b/cliff/analysis_window3.log 2>&1
caffeinate -i -s .venv/bin/python -u analyze_profile.py --model 1.7b > results/1.7b/criterion/profile.log 2>&1
caffeinate -i -s .venv/bin/python -u analyze_cliff.py --model 0.6b --configs routed3,routedr3 --correctors-for routed3,routedr3 --robust-corr-tau 10 --out-name correctors_robust.json > results/0.6b/cliff/correctors_robust.log 2>&1
mkdir -p results/0.6b/ghost results/0.6b/long
caffeinate -i -s .venv/bin/python -u ghost_blocks.py --model 0.6b > results/0.6b/ghost/blocks.log 2>&1
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE
caffeinate -i -s .venv/bin/python -u eval_corpus.py --model 0.6b > logs/eval_corpus_0.6b.log 2>&1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
caffeinate -i -s .venv/bin/python -u eval_long.py --model 0.6b --study sinks --k 3 > results/0.6b/long/sinks_k3.log 2>&1
caffeinate -i -s .venv/bin/python -u eval_long.py --model 0.6b --study sinks --k 6 > results/0.6b/long/sinks_k6.log 2>&1
caffeinate -i -s .venv/bin/python -u eval_long.py --model 0.6b --study length --k 3 > results/0.6b/long/length_k3.log 2>&1
# 3) замер времени — в конце, когда ничего не мешает
caffeinate -i -s .venv/bin/python -u bench_time.py --model 0.6b > logs/bench_time_0.6b.log 2>&1
caffeinate -i -s .venv/bin/python -u bench_time.py --model 1.7b > logs/bench_time_1.7b.log 2>&1
.venv/bin/python make_brief.py >> $LOG 2>&1
.venv/bin/python make_report.py >> $LOG 2>&1
echo "$(date '+%F %T') resume: всё завершено" >> $LOG
