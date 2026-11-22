"""Общие тенденции голов внимания двух моделей: PDF со сравнением (что совпадает, что различается).

Читает результаты head_mechanisms_deep.py, head_circuits.py, head_freq_adv.py, mlp_writers.py, head_ramp_check.py и
head_mechanisms.py для обеих моделей и собирает brief/heads_compare_<a>_<b>.pdf.

  python make_heads_compare.py --models pocket0.3b,0.6b
"""
from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from make_heads_report import BUNDLE, INK, OUT, PAL, TITLE, classify, gridify, mid, near8, num, ranges, sgn  # noqa: E402

KINDS = ["предыдущий токен", "сам токен", "локальное окно ≤ 8", "окно ≤ 64", "тот же токен", "индукция",
         "собирает определённые токены", "дальний контекст"]
KNAME = {"предыдущий токен": "Предыдущий токен", "сам токен": "Сам токен", "локальное окно ≤ 8": "Локальное окно ≤ 8", "окно ≤ 64": "Окно ≤ 64",
         "тот же токен": "Тот же токен", "индукция": "Индукция", "собирает определённые токены": "Собирает определённые токены",
         "дальний контекст": "Дальний контекст"}
TYP = {"предыдущий токен": "prev", "сам токен": "self", "тот же токен": "same", "индукция": "ind", "сток": "sink"}


def mean(v):
    v = list(v)
    return sum(v) / max(len(v), 1)


def stats(m):
    """Все числа одной модели, которые нужны для сравнения."""
    base = ROOT / "results" / m / "heads"
    res = json.loads((base / "all_layers.json").read_text())
    circ = json.loads((base / "circuits.json").read_text())
    fa = json.loads((base / "freq_adv.json").read_text())
    mw = json.loads((base / "mlp_writers.json").read_text())
    ramp = json.loads((base / "ramp_check.json").read_text())
    l0 = json.loads((base / "layer0.json").read_text())
    hs, H, NL = res["heads"], res["H"], res["layers"]
    by = [[x for x in hs if x["layer"] == l] for l in range(NL)]
    lab = {(x["layer"], x["head"]): classify(x)[1] for x in hs}
    kind = {k_: v.replace(" + сток", "").replace("равномерное усреднение", "дальний контекст") for k_, v in lab.items()}
    K = {k_: [x for x in hs if kind[(x["layer"], x["head"])] == k_] for k_ in KINDS}
    K["сток"] = [x for x in hs if lab[(x["layer"], x["head"])].endswith("+ сток")]
    fh = {(x["layer"], x["head"]): x for x in fa["heads"]}
    ch = {(x["layer"], x["head"]): x for x in circ["heads"]}

    def fshare(xs, typ, key):
        tot = sum(fh[(x["layer"], x["head"])]["adv"][typ]["logits"] for x in xs) or 1e-9
        return sum(fh[(x["layer"], x["head"])]["adv"][typ][key] for x in xs) / tot

    def av(l, f):
        return mean(f(x) for x in by[l])

    ind_top = [ch[(x["layer"], x["head"])]["adv"]["ind"]["key"]["top"][0][0] for x in K["индукция"]]

    def is_prev(nm):
        if not nm.startswith("L"):
            return False
        l_, h_ = nm[1:].split(".H")
        return kind[(int(l_), int(h_))] == "предыдущий токен"

    ind_tot = sum(ch[(x["layer"], x["head"])]["adv"]["ind"]["logits"] for x in K["индукция"]) or 1e-9
    sink_l = [av(l, lambda x: x["text"]["pos0"]) for l in range(NL)]
    r2 = [av(l, lambda x: x["terms_std_logits"]["token_r2_q"]) for l in range(NL)]
    n90 = mw["sink"]["n_for_90"]
    acts = mw["sink"]["act_pos0_over_typical"][:max(1, min(n90, 5))]
    return {
        "name": TITLE.get(m, m), "H": H, "NL": NL, "G": res["G"], "n": NL * H, "K": K,
        "sink_layers": sorted({x["layer"] for x in K["сток"]}),
        "fast_prev": fshare(K["предыдущий токен"], "prev", "fast_logits"),
        "slow": {k_: fshare(K[k_], TYP[k_], "slow_logits") for k_ in ("тот же токен", "индукция", "сток") if K[k_]},
        "adv": {k_: mean(ch[(x["layer"], x["head"])]["adv"][TYP[k_]]["logits"] for x in K[k_]) for k_ in TYP if K[k_]},
        "ind_prev": sum(1 for n_ in ind_top if is_prev(n_)), "ind_n": len(ind_top),
        "ind_key_heads": sum(ch[(x["layer"], x["head"])]["adv"]["ind"]["key"]["agg"]["внимание"] * ch[(x["layer"], x["head"])]["adv"]["ind"]["logits"]
                             for x in K["индукция"]) / ind_tot,
        "sink_mlp": mw["sink_layer"], "sink_n90": n90, "sink_act": (min(acts), max(acts)), "rms": (mw["rms_ratio_before"], mw["rms_ratio_after"]),
        "sink_last": (min(sink_l[NL - 10:]), max(sink_l[NL - 10:])), "sink_first": next((l for l in range(NL) if sink_l[l] > 0.05), None),
        "sink_max_head": max(x["text"]["pos0"] for x in hs),
        "kdrop": {k_: mean(x["kernel"]["P"]["1"] - x["kernel"]["P"]["64"] for x in K[k_]) for k_ in ("локальное окно ≤ 8", "окно ≤ 64", "дальний контекст")},
        "k_gamma": ramp["k_gamma"], "k_median": ramp["k_median"], "k_period": ramp["period_tokens"],
        "q_ch": max(abs(h_["salience"]["mean_q_on_channel"]) + h_["salience"]["std_q_on_channel"] for h_ in l0["heads"]),
        "far": (mean(ramp["original"]["far256"]), mean(ramp["k_to_median"]["far256"])),
        "r2": (r2[0], r2[min(5, NL - 1)], min(r2[min(12, NL - 1):]), max(r2[min(12, NL - 1):])),
        "src_mlp": max(av(l, lambda x: x["sources"][0]["q"]["agg"]["MLP"]) for l in range(1, min(16, NL))),
        "src_heads_last": av(NL - 1, lambda x: x["sources"][0]["q"]["agg"]["внимание"]),
        "series": {"сток": sink_l, "до 8 токенов": [av(l, near8) for l in range(NL)], "R²": r2,
                   "ключей": [av(l, lambda x: x["text"]["effective_keys"]) for l in range(NL)],
                   "позиционные": [mean(kind[(x["layer"], x["head"])] in ("предыдущий токен", "сам токен", "локальное окно ≤ 8", "окно ≤ 64") for x in by[l]) for l in range(NL)],
                   "MLP": [av(l, lambda x: x["sources"][0]["q"]["agg"]["MLP"]) for l in range(NL)]},
    }


def figure(A, B, stem):
    panels = [("сток", "Доля внимания на первый токен (сток)"), ("до 8 токенов", "Доля внимания в пределах 8 токенов"),
              ("позиционные", "Доля голов, которые смотрят по расстоянию"), ("R²", "Доля запроса, объяснимая токеном (R²)"),
              ("ключей", "Эффективное число ключей"), ("MLP", "Доля MLP в источниках главной частоты запроса")]
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.0), sharex=True)
    for ax, (key, title) in zip(axes.flat, panels):
        for S, col in ((A, PAL[0]), (B, PAL[1])):  # цвет закреплён за моделью на всех панелях
            ax.plot(range(S["NL"]), S["series"][key], color=col, lw=1.6, marker="o", ms=2.6, label=S["name"])
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_ylim(0, None)
        ax.tick_params(labelsize=7.5)
        ax.set_xticks(range(0, max(A["NL"], B["NL"]), 3))
    axes[0][0].legend(fontsize=7.5, frameon=False, handlelength=1.4, loc="upper left")
    for a in axes[1]:
        a.set_xlabel("слой", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}_fig.pdf")
    plt.close(fig)


def build(models):
    A, B = stats(models[0]), stats(models[1])
    stem = "heads_compare_" + "_".join(m.replace(".", "") for m in models)
    figure(A, B, stem)

    def pct(S, k_):
        return f"{len(S['K'][k_])} ({100 * len(S['K'][k_]) / S['n']:.0f}\\,\\%)"

    def both(f):
        return f"{f(A)} & {f(B)}"

    def rng(t, nd=2):
        return f"{num(t[0], nd)}–{num(t[1], nd)}"

    def whole(v):
        return "целиком" if v > 0.995 else num(v)

    n_pos = lambda S: sum(len(S["K"][k_]) for k_ in ("предыдущий токен", "сам токен", "локальное окно ≤ 8", "окно ≤ 64"))  # noqa: E731
    n_cont = lambda S: sum(len(S["K"][k_]) for k_ in ("тот же токен", "индукция", "собирает определённые токены"))  # noqa: E731
    a, b = A["name"], B["name"]
    ramp_a, ramp_b = A["far"][1] - A["far"][0] > 0.15, B["far"][1] - B["far"][0] > 0.15
    T = [r"\documentclass[10pt]{article}", r"\usepackage{fontspec}", r"\setmainfont{Times New Roman}", r"\usepackage[russian]{babel}", r"\usepackage{amsmath}",
         r"\usepackage[a4paper,landscape,margin=1.3cm]{geometry}", r"\usepackage{graphicx,xcolor,colortbl,array,enumitem}",
         r"\arrayrulecolor[HTML]{B5B4AB}\definecolor{head}{HTML}{F0EFEC}\definecolor{ink2}{HTML}{52514E}",
         r"\setlist{nosep,leftmargin=1.3em,itemsep=0.25em}", r"\setlength{\parindent}{0pt}\setlength{\parskip}{0.45em}",
         r"\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.1}", r"\begin{document}", r"\selectlanguage{russian}",
         rf"{{\LARGE\bfseries Головы внимания {a} и {b}: общие тенденции}}\\[0.2em]",
         rf"{{\color{{ink2}} {a}: {A['NL']} слоёв $\times$ {A['H']} голов $=$ {A['n']}. {b}: {B['NL']} слоёв $\times$ {B['H']} голов $=$ {B['n']}. "
         r"Одна архитектура (Qwen3), один проверочный текст (WikiText-2, 8192 токена). Там, где стоят два числа, первое относится к " + rf"{a}, второе~— к {b}.}}", "",
         r"\section*{1. Что общее}", r"\begin{itemize}",
         rf"\item \textbf{{Большинство голов смотрит по расстоянию.}} По расстоянию смотрят {n_pos(A)} из {A['n']} и {n_pos(B)} из {B['n']} голов, "
         rf"дальний контекст усредняют {len(A['K']['дальний контекст'])} и {len(B['K']['дальний контекст'])}, по содержанию выбирают только {n_cont(A)} и {n_cont(B)}.",
         rf"\item \textbf{{Быстрые частоты RoPE кодируют расстояние, медленные~— содержание.}} «Предыдущий токен» берёт {num(A['fast_prev'])} и {num(B['fast_prev'])} "
         r"преимущества цели на частотах с периодом меньше 20 токенов. На частотах с периодом больше 20 тысяч токенов «Тот же токен» берёт "
         rf"{num(A['slow']['тот же токен'])} и {num(B['slow']['тот же токен'])}, индукция~— {num(A['slow']['индукция'])} и {num(B['slow']['индукция'])}, "
         rf"сток~— {num(A['slow']['сток'])} и {num(B['slow']['сток'])}.",
         rf"\item \textbf{{Сток создают отдельные нейроны одного MLP.}} На первом токене они срабатывают в тысячи раз сильнее обычного "
         rf"({A['sink_act'][1]:.0f} и {B['sink_act'][1]:.0f} раз), и норма остаточного потока там вырастает до {num(A['rms'][1], 0)} и {num(B['rms'][1], 0)} от обычной. "
         r"Ключи читают эту запись на медленных частотах.",
         rf"\item \textbf{{Индукция собрана одинаково.}} Ключ приходит из предыдущих голов, главный источник~— голова «Предыдущий токен» "
         rf"(у {A['ind_prev']} из {A['ind_n']} и у {B['ind_prev']} из {B['ind_n']} индукционных голов); запрос несёт сам текущий токен.",
         rf"\item \textbf{{Ширину окна задаёт крутизна позиционного ядра.}} Между расстояниями 1 и 64 ядро падает на {num(A['kdrop']['локальное окно ≤ 8'], 1)} и "
         rf"{num(B['kdrop']['локальное окно ≤ 8'], 1)} логита у «Локального окна ≤ 8», на {num(A['kdrop']['окно ≤ 64'], 1)} и {num(B['kdrop']['окно ≤ 64'], 1)} у «Окна ≤ 64», "
         rf"на {num(A['kdrop']['дальний контекст'], 1)} и {num(B['kdrop']['дальний контекст'], 1)} у «Дальнего контекста».",
         rf"\item \textbf{{Чем глубже, тем меньше решает сам токен.}} Он объясняет {num(A['r2'][0])} и {num(B['r2'][0])} запроса в слое 0, "
         rf"{num(A['r2'][1])} и {num(B['r2'][1])} в слое 5, {rng(A['r2'][2:])} и {rng(B['r2'][2:])} после слоя 12.",
         r"\end{itemize}",
         r"\section*{2. Чем различаются}", r"\begin{itemize}",
         rf"\item \textbf{{Сток у {b} раньше и сильнее.}} Его создаёт MLP слоя {B['sink_mlp']} ({B['sink_n90']} нейронов), у {a}~— MLP слоя {A['sink_mlp']} "
         rf"({A['sink_n90']} нейрон). Приписку «+ сток» имеют {len(B['K']['сток'])} из {B['n']} голов против {len(A['K']['сток'])} из {A['n']}; "
         rf"в последних десяти слоях на первый токен уходит {rng(B['sink_last'])} внимания против {rng(A['sink_last'])}.",
         rf"\item \textbf{{Большой вес нормализации ключей работает по-разному.}} "
         + (rf"У {a} вес {num(A['k_gamma'], 1)} (медиана {num(A['k_median'], 1)}) стоит на частоте с периодом {A['k_period']:.0f} токенов и задаёт спад с расстоянием: "
            rf"замена на медиану поднимает долю внимания дальше 256 токенов с {num(A['far'][0])} до {num(A['far'][1])}. " if ramp_a else
            rf"У {a} вес {num(A['k_gamma'], 1)} почти не влияет: {num(A['far'][0])} → {num(A['far'][1])}. ")
         + (rf"У {b} вес {num(B['k_gamma'], 1)} стоит на частоте с периодом {B['k_period']:.0f} токенов и тоже задаёт спад: {num(B['far'][0])} → {num(B['far'][1])}." if ramp_b else
            rf"У {b} вес {num(B['k_gamma'], 1)} (медиана {num(B['k_median'], 1)}) стоит на частоте с периодом {B['k_period']:.0f} токенов, но запрос на этой компоненте "
            rf"почти нулевой (не больше {num(B['q_ch'])}), и замена веса ничего не меняет: {num(B['far'][0])} → {num(B['far'][1])}."),
         rf"\item \textbf{{Состав голов.}} «Сам токен»: {pct(A, 'сам токен')} у {a} и {pct(B, 'сам токен')} у {b}. "
         rf"«Собирает определённые токены»: {pct(A, 'собирает определённые токены')} и {pct(B, 'собирает определённые токены')}.",
         r"\end{itemize}",
         r"\noindent\begin{minipage}{\linewidth}",
         r"\section*{3. Графики по слоям}",
         rf"\begin{{center}}\includegraphics[width=0.9\linewidth]{{{stem}_fig.pdf}}\end{{center}}",
         r"\end{minipage}", r"\par",
         r"\noindent\begin{minipage}{\linewidth}",
         r"\section*{4. Сравнение в числах}",
         r"\begin{center}{\small\begin{tabular}{|p{11.2cm}|r|r|}\toprule",
         rf"\rowcolor{{head}}\textbf{{Показатель}} & \textbf{{{a}}} & \textbf{{{b}}} \\ \midrule",
         r"Слоёв $\times$ голов; групп ключей и значений & " + both(lambda S: f"{S['NL']} $\\times$ {S['H']}; {S['G']}") + r" \\"]
    for k_ in KINDS:
        T.append(rf"Вид «{KNAME[k_]}»: голов (доля) & " + both(lambda S, k_=k_: pct(S, k_)) + r" \\")
    T += [r"Приписка «+ сток»: голов (доля); слои & " + both(lambda S: pct(S, "сток") + "; " + ranges(S["sink_layers"])) + r" \\",
          r"Сток: слой MLP; нейронов на 90\,\% записи; активация на первом токене относительно обычной & "
          + both(lambda S: f"{S['sink_mlp']}; {S['sink_n90']}; {S['sink_act'][0]:.0f}–{S['sink_act'][1]:.0f}" if S["sink_n90"] > 1 else f"{S['sink_mlp']}; 1; {S['sink_act'][1]:.0f}") + r" \\",
          r"Норма остаточного потока на первом токене относительно обычной: до и после этого MLP & " + both(lambda S: f"{num(S['rms'][0], 1)} → {num(S['rms'][1], 1)}") + r" \\",
          r"Доля внимания на первый токен в последних десяти слоях; максимум у одной головы & " + both(lambda S: f"{rng(S['sink_last'])}; {num(S['sink_max_head'])}") + r" \\",
          r"«Предыдущий токен»: доля преимущества цели на частотах с периодом меньше 20 токенов & " + both(lambda S: num(S["fast_prev"])) + r" \\",
          r"Доля преимущества на частотах с периодом больше 20 тысяч токенов: тот же токен / индукция / сток & "
          + both(lambda S: " / ".join(num(S["slow"][k_]) for k_ in ("тот же токен", "индукция", "сток"))) + r" \\",
          r"Преимущество цели, логиты: предыдущий токен / сам токен / тот же токен / индукция / сток & "
          + both(lambda S: " / ".join(sgn(S["adv"][k_]) for k_ in ("предыдущий токен", "сам токен", "тот же токен", "индукция", "сток"))) + r" \\",
          r"Индукция: у скольких голов главный источник ключа~— голова «Предыдущий токен»; доля ключа из голов & "
          + both(lambda S: f"{S['ind_prev']} из {S['ind_n']}; {whole(S['ind_key_heads'])}") + r" \\",
          r"Падение ядра между расстояниями 1 и 64, логиты: окно ≤ 8 / окно ≤ 64 / дальний контекст & "
          + both(lambda S: " / ".join(num(S["kdrop"][k_], 1) for k_ in ("локальное окно ≤ 8", "окно ≤ 64", "дальний контекст"))) + r" \\",
          r"Наибольший вес нормализации ключей в слое 0 (медиана); период его частоты, токенов & " + both(lambda S: f"{num(S['k_gamma'], 1)} ({num(S['k_median'], 1)}); {S['k_period']:.0f}") + r" \\",
          r"Доля внимания дальше 256 токенов в слое 0: исходные веса → этот вес заменён на медиану & " + both(lambda S: f"{num(S['far'][0])} → {num(S['far'][1])}") + r" \\",
          r"Доля запроса, объяснимая токеном: слой 0 / слой 5 / после слоя 12 & " + both(lambda S: f"{num(S['r2'][0])} / {num(S['r2'][1])} / {rng(S['r2'][2:])}") + r" \\",
          r"Источники главной частоты запроса: наибольшая доля MLP в слоях 1–15; доля голов в последнем слое & " + both(lambda S: f"{num(S['src_mlp'])}; {num(S['src_heads_last'])}") + r" \\",
          r"\bottomrule\end{tabular}}\end{center}", r"\end{minipage}", r"\par",
          r"\textbf{Границы сравнения.} Обе модели проверены на одном коротком тексте, виды голов определены по одним и тем же порогам. "
          r"Сравнение касается того, куда головы смотрят; что они записывают в остаточный поток, сюда не входит.",
          r"\end{document}", ""]
    texf = OUT / f"{stem}.tex"
    texf.write_text(gridify("\n".join(T)))
    rc = subprocess.run(["tectonic", "-X", "compile", "--bundle", BUNDLE, texf.name], cwd=OUT, capture_output=True, text=True)
    if rc.returncode:
        print(rc.stdout[-600:], rc.stderr[-1500:])
    print("сохранено", texf.with_suffix(".pdf") if rc.returncode == 0 else "ОШИБКА СБОРКИ")
    return rc.returncode


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="pocket0.3b,0.6b")
    sys.exit(build(ap.parse_args().models.split(",")))
