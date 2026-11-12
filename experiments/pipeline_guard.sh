#!/bin/bash
# Сторож конвейера. Запускается отдельной сессией (setsid), не зависит от Claude Code / терминала.
# Ждёт, пока текущая цепочка (sweep.py / make_brief.py / run.py) завершится или погибнет,
# затем продолжает конвейер с места остановки: все этапы пропускают готовые JSON.
cd "$(dirname "$0")" || exit 1
LOG=logs/guard.log
mkdir -p logs
echo "$(date '+%F %T') сторож запущен (pid $$)" >> "$LOG"
quiet=0
while true; do
  if pgrep -f "sweep.py --preset mac" >/dev/null || pgrep -f "make_brief.py" >/dev/null \
     || pgrep -f "run.py --model" >/dev/null; then
    quiet=0
  else
    quiet=$((quiet + 1))
  fi
  [ "$quiet" -ge 3 ] && break      # 3 минуты тишины: цепочка завершилась или была убита
  sleep 60
done
echo "$(date '+%F %T') сторож: процессов цепочки нет, продолжаю конвейер" >> "$LOG"
caffeinate -i -s .venv/bin/python -u sweep.py --preset mac --stage howmany >> logs/sweep_mac_howmany.log 2>&1
.venv/bin/python make_brief.py >> "$LOG" 2>&1
caffeinate -i -s .venv/bin/python -u sweep.py --preset mac --stage all >> logs/sweep_mac_all2.log 2>&1
.venv/bin/python make_brief.py >> "$LOG" 2>&1
echo "$(date '+%F %T') сторож: конвейер завершён" >> "$LOG"
