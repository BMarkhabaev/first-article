#!/bin/bash
# ksweep исключён пользователем 2026-11-08: когда текущий основной прогон (старый список этапов) дойдёт
# до ksweep — остановить его; сторож затем запустит `sweep all` с новым порядком (без ksweep).
cd "$(dirname "$0")" || exit 1
until grep -q "stage ksweep" logs/sweep_mac_all2.log 2>/dev/null; do sleep 120; done
sleep 5
pkill -f "sweep.py --preset mac --stage all"
pkill -f "results/0.6b/ksweep"
echo "$(date '+%F %T') основной прогон остановлен на этапе ksweep (исключён)" >> logs/guard.log
