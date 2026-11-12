# Эксперименты: детерминированная аппроксимация первых слоёв Qwen3

Код к статье «Детерминированные алгоритмы аппроксимации первых слоёв LLM».
План, решения и журнал — в [PLAN.md](PLAN.md).

## Запуск на Mac (основной вариант статьи)

```bash
bash run_mac.sh all            # ~21 ч, в фоне под caffeinate; прогресс: tail -f logs/sweep_mac_all.log
```

Пресет `--preset mac`: Qwen3-0.6B и 1.7B (4B/8B в fp32 не помещаются в 16 ГБ), калибровка 131k
токенов, оценка 32 окна × 1024, бенчи по 200 примеров, MMLU 5 вопросов на подзадачу, пакет
lm-eval 8 (MMLU — 2: автоподбор на MPS не работает). Порядок этапов: profile → final → distill →
pareto → ksweep. Ноутбук — от сети с открытой крышкой; прерванный прогон продолжается повторным
запуском. Затем `python make_report.py` → `../figures`, `../tables`.

## Запуск на H100

```bash
# 0. Скопировать папку experiments/ на сервер (без .venv, results_mac, logs_mac).
export HF_TOKEN=...            # желательно: без токена HF Hub ограничивает скорость, lm-eval может «висеть»
bash run_h100.sh setup         # окружение + Qwen3 0.6B/1.7B/4B/8B Base + WikiText (~35 ГБ)
bash run_h100.sh smoke         # ~5 мин: методы и lm-eval работают на GPU
bash run_h100.sh all           # все этапы в фоне; прогресс: tail -f logs/sweep_all.log
```

Прогон можно прерывать: `sweep.py` пропускает уже готовые JSON. Повторный запуск того же
этапа продолжит с места остановки. Этапы можно запускать и по отдельности
(`bash run_h100.sh final`).

**Вернуть результаты:** скопировать обратно папку `experiments/results/` (JSON + логи, единицы МБ).

### Этапы и ориентировочное время на одной H100

| Этап | Что делает | Запусков | Время |
|---|---|---|---|
| `profile` | профиль всех слоёв 4 моделей (внимание, нормы q/k, линейность MLP, sink) | 4 | ~0.5 ч |
| `ksweep` | 0.6B: KL/PPL каждого метода при k = 1, 2, 3, 4, 6 | 130 | ~3 ч |
| `pareto` | 0.6B, k = k_sink: качество vs бюджет (доля параметров MLP, окно, доля точных голов) | 36 | ~1 ч |
| `final` | все модели, k = k_sink: KL/PPL + бенчи (0.6B — все методы, 1.7B/4B/8B — 10 основных) | 59 | ~6 ч |
| `distill` | 0.6B, 1.7B: дистилляционный трек (1000 шагов AdamW), помечен отдельно | 12 | ~1.5 ч |

`k_sink` — число первых слоёв до появления массивной активации на первом токене;
определяется автоматически по профилю (для 0.6B это 3).

### Протокол

- KL, PPL, top-1 — в fp32 (шум bf16 даёт KL ≈ 1e-3), WikiText-2 test, окна 2048.
- Бенчи (lm-eval 0.4.13) — в bf16: LAMBADA, HellaSwag, ARC-e/c, PIQA, WinoGrande — 0-shot,
  до 1000 примеров; MMLU — 5-shot, 50 вопросов на подзадачу (~2850).
- Калибровка — WikiText-103 train (не пересекается с тестом), 262k токенов
  (n-граммные и табличные методы — 2–4M).

## Структура

- `detlayers/common.py` — загрузка модели и данных, ручной проход по слоям.
- `detlayers/attention.py` — замены внимания (окно, strided, Performer, линейное, позиционное, O1, O4).
- `detlayers/methods.py` — калибровка и сборка методов, A3, O2 (ridge-свёртка), O3 (n-граммы), диспетчер.
- `detlayers/mlp.py` — замены MLP (A3, top-k/Deja Vu, ASVD, SVD-LLM, MoEfication, наши ridge/table/cluster).
- `detlayers/distill.py` — дистилляционный трек.
- `detlayers/metrics.py` — KL/PPL/top-1 и lm-eval.
- `run.py` — один эксперимент; `sweep.py` — все этапы; `profile_layers.py` — профиль слоёв.
- `smoke_test.py` — быстрая проверка методов (вырожденные режимы должны давать KL = 0).

## Методы

Список с номерами из литературы — в [PLAN.md](PLAN.md), раздел «Методы».
Комбинации «внимание + MLP» задаются через `+`:
`python run.py --method sparse+ridge_mlp --k 3 --params '{"attn": {"window": 64}, "mlp": {"keep_ratio": 0.25}}'`.

### Дополнительные этапы (добавлены 2026-11-07)

- `python sweep.py --preset mac --stage howmany_other` — сокращённый «сколько слоёв» для 1.7B
  (окно 64, M-O4, оракул Deja Vu; k до 8) → `results/1.7b/howmany/`.
- `python sweep.py --preset mac --stage bench1000` — главные столбцы (orig, окно 64, M-O4, SVD-LLM, окно+M-O4)
  с бенчами на 1000 примерах (MMLU 20 на подзадачу) → `results/<model>/bench1000/`.
- Порядок `--stage all`: profile → final → howmany_other → bench1000 → distill → pareto → ksweep.
