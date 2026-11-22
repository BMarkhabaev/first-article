#!/bin/bash
# Пересборка рисунка fig_types_slide.pdf и слайда после очереди queue_topk17.sh (pid в $1).
cd "$(dirname "$0")" || exit 1
WAIT=${1:-}
[ -n "$WAIT" ] && while kill -0 "$WAIT" 2>/dev/null; do sleep 60; done
.venv/bin/python -c "import make_brief as mb; print(mb.fig_types())" > logs/slide_refresh.log 2>&1
cd ../brief && tectonic -X compile slide.tex >> ../experiments/logs/slide_refresh.log 2>&1 && mv -f slide.pdf "Мархабаев slide.pdf"
echo "$(date '+%F %T') slide: пересобран rc=$?" >> ../experiments/logs/novelty.log
