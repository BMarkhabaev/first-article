# Запуск на H100 (офлайн, без sudo)

## Что должно лежать на машине
1. **Веса**: `/app/models/Qwen3-4B-Base`, `…/Qwen3-8B-Base`, `…/Qwen3-32B` (Base-варианта 32B не выпущено,
   берётся пост-обученная модель той же архитектуры; допустимо без суффикса `-Base`, папка ищется в обоих
   вариантах; другой каталог — `MODELS_DIR=…`). 14B/32B считаются в bf16 на одной карте (fp32 не помещается),
   суммы калибровки всё равно в fp32/fp64.
   В каждой папке: `config.json`, `model*.safetensors`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`.
2. **Тексты** — уже в архиве, папка `data/`: WikiText-2 test (PPL/KL), префикс WikiText-103 train (калибровка),
   pile-10k (второй корпус). Пересобрать при необходимости: `python tools/export_data.py` (нужен интернет).
3. **Датасеты бенчмарков lm-eval** — второй архив `h100_hf_datasets.zip` → распаковать в `/app/hf_cache/`
   (получится `/app/hf_cache/datasets/Rowan___hellaswag`, `cais___mmlu`, `allenai___ai2_arc`, `baber___piqa`,
   `allenai___winogrande`, `EleutherAI___lambada_openai`) и выставить `HF_HOME=/app/hf_cache`.
4. **Python ≥ 3.10** с пакетами из `requirements-h100.txt` (torch — CUDA-сборка). Без интернета колёса надо
   принести: на машине с сетью и тем же Python/платформой —
   `pip download -r requirements-h100.txt -d wheels --extra-index-url https://download.pytorch.org/whl/cu124`,
   затем на H100 `pip install --no-index --find-links wheels -r requirements-h100.txt`.

## Запуск
```bash
cd experiments
export MODELS_DIR=/app/models HF_HOME=/app/hf_cache \
       HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nohup python -u h100.py --models 4b,8b,32b --gpus 8 > logs/h100.out 2>&1 &
```
`h100.py` сам считает профили, затем раздаёт независимые задания (`run.py`, по одному на карту) и в конце
запускает анализы и `make_report.py` / `make_brief.py`. Готовые JSON пропускаются — перезапуск безопасен.
Только посмотреть задания: `python h100.py --models 4b --plan`. Подмножество этапов: `--stages final,howmany`.

Что считается: `final` (таблица, полные бенчмарки), `robust` (робастная калибровка), `extras` (робастный A3),
`ideas` (ранг линейной части, низкоранговая поправка M-O4, CATS), `howmany` (k до 12 для 28 слоёв, до 18 для 36),
`distill` (0.6B/1.7B/4B, 1000 шагов), `pareto`; анализы: обрыв, критерий, призраки блоков, второй корпус,
стоки и длина контекста, замер времени. Для 8B/32B в `final` — только столбцы таблицы (FINAL_MAIN + робастные);
методы с последовательной калибровкой (M-O4, M-O1) на них считаются до k=12, остальные — до половины глубины.

## Результаты
`results/<model>/…/*.json` и журналы `logs/h100/`. Забрать: `tar czf results.tgz results logs/h100/summary.log`;
на Mac распаковать в `experiments/` и выполнить `python make_report.py && python make_brief.py`.
