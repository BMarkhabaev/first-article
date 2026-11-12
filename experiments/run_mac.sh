#!/usr/bin/env bash
# Все эксперименты на Mac (M4, 16 ГБ) — пресет «mac» в sweep.py.
#   bash run_mac.sh all        # все этапы в фоне; лог logs/sweep_mac_all.log
#   bash run_mac.sh <этап>     # profile | ksweep | pareto | final | distill
# caffeinate -i не даёт системе уснуть, пока идёт прогон (ноутбук должен быть от сети
# и с открытой крышкой). Прогон можно прерывать: готовые JSON пропускаются.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
st="${1:-all}"
case "$st" in
  all|profile|ksweep|pareto|final|distill) ;;
  *) sed -n 2,5p "$0"; exit 1 ;;
esac
nohup caffeinate -i .venv/bin/python -u sweep.py --preset mac --stage "$st" > "logs/sweep_mac_$st.log" 2>&1 &
echo "запущено в фоне, PID $!; прогресс: tail -f logs/sweep_mac_$st.log"
