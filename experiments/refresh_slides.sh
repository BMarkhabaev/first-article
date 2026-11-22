#!/bin/bash
# Пересборка рисунка fig_types_slide.pdf и обоих слайдов: «Мархабаев slide.pdf» (основной, в git) и
# «Мархабаев slide графики.pdf» (только графики, в .gitignore). Аргумент — метка для журнала.
# tectonic иногда падает при подкачке файлов пакета (article.cls not found) — до 3 попыток с паузой.
cd "$(dirname "$0")" || exit 1
.venv/bin/python -c "import make_brief as mb; print(mb.fig_types())" >> logs/slide_refresh.log 2>&1
cd ../brief || exit 1
build() {  # исходник.tex итог.pdf
  for i in 1 2 3; do
    tectonic -X compile "$1" >> ../experiments/logs/slide_refresh.log 2>&1 && mv -f "${1%.tex}.pdf" "$2" && return 0
    sleep 60
  done
  return 1
}
build slide.tex "Мархабаев slide.pdf"; a=$?
build slide_graphs.tex "Мархабаев slide графики.pdf"; b=$?
echo "$(date '+%F %T') slide: оба слайда пересобраны после ${1:-?} rc=$a/$b" >> ../experiments/logs/novelty.log
