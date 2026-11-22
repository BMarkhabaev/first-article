"""Отчёт «что делают головы внимания» по всем слоям модели: PDF с пояснениями, тенденциями и таблицами.

Читает results/<m>/heads/all_layers.json (head_mechanisms_deep.py) и ramp_check.json (head_ramp_check.py),
рисует графики тенденций и собирает brief/heads_<m>.tex → PDF (tectonic).

  python make_heads_report.py --model pocket0.3b
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
from make_report import GRID, INK, INK2, PAL  # noqa: E402,F401  (стиль rcParams задаётся при импорте)

OUT = ROOT.parent / "brief"
BUNDLE = "https://data1.fullyjustified.net/tlextras-2022.0r0.tar"  # тот же пакет, что у статьи (в нём есть longtable)
TITLE = {"pocket0.3b": "MNN-Pocket-0.3B", "0.6b": "Qwen3-0.6B", "1.7b": "Qwen3-1.7B"}
SEQ = ["#86b6ef", "#5598e7", "#256abf", "#104281"]  # один оттенок, светлый → тёмный: расстояние растёт
CATS = ["предыдущий или сам токен", "локальное окно ≤ 8", "окно ≤ 64", "тот же токен", "индукция",
        "собирает определённые токены", "дальний контекст"]
# фиксированные цвета категорий; оранжевый (PAL[1]) занят стоком на верхней панели, «дальний контекст» — нейтральный серый
KIND_CAT = {"предыдущий токен": "предыдущий или сам токен", "сам токен": "предыдущий или сам токен",
            "локальное окно ≤ 8": "локальное окно ≤ 8", "окно ≤ 64": "окно ≤ 64", "тот же токен": "тот же токен",
            "индукция": "индукция", "собирает определённые токены": "собирает определённые токены",
            "дальний контекст": "дальний контекст"}
CAT_COLOR = dict(zip(CATS, [PAL[0], PAL[2], PAL[3], PAL[4], PAL[5], PAL[6], "#c3c2b7"]))


def near8(x):
    m = x["text"]["mass_by_distance"]
    return m["0"] + m["1"] + m["2"] + m["3-4"] + m["5-8"]


def mid(x):
    m = x["text"]["mass_by_distance"]
    return m["9-16"] + m["17-32"] + m["33-64"]


def classify(x):
    """(категория для подсчёта, подпись функции). Метка описывает внимание, не ушедшее в сток: доли по расстоянию
    делятся на (1 − доля позиции 0); сток — приписка. Пороги см. в разделе «Пояснения»."""
    t, s = x["text"], x["terms_std_logits"]
    m = t["mass_by_distance"]
    sink = t["pos0"]
    k = 1.0 / max(1.0 - sink, 1e-6)
    n8, n64 = near8(x) * k, (near8(x) + mid(x)) * k
    if t["ind_mass"] > 0.15 and t["ind_mass"] > 5 * t["ind_baseline"]:
        cat, lab = "индукция", "индукция"
    elif t["dup_mass"] > 0.2 and t["dup_mass"] > 5 * t["dup_baseline"]:
        cat, lab = "тот же токен", "тот же токен"
    elif m["1"] * k > 0.4:
        cat, lab = "предыдущий или сам токен", "предыдущий токен"
    elif m["0"] * k > 0.4:
        cat, lab = "предыдущий или сам токен", "сам токен"
    elif n8 > 0.6:
        cat, lab = "локальное окно ≤ 8", "локальное окно ≤ 8"
    elif n64 > 0.75:
        cat, lab = "окно ≤ 64", "окно ≤ 64"
    elif s["key_salience"] >= 2.5:
        cat, lab = "собирает определённые токены", "собирает определённые токены"
    else:
        cat, lab = "дальний контекст", "равномерное усреднение" if t["effective_keys"] > 80 else "дальний контекст"
    if sink > 0.2:
        lab += " + сток"
    return cat, lab


def chip(cat):
    """Цветной квадрат вида головы — тот же цвет, что на рисунке 1."""
    return r"\textcolor[HTML]{" + CAT_COLOR[cat].lstrip("#").upper() + r"}{\rule{0.62em}{0.62em}}\,"


def gridify(t):
    """Сетка во всех таблицах: вертикальные линии в описаниях столбцов задаются на месте, здесь — горизонтальные."""
    out = []
    for line in t.split("\n"):
        line = line.replace(r"\toprule", r"\hline").replace(r"\midrule", r"\hline").replace(r"\bottomrule", r"\hline")
        if line.rstrip().endswith("\\\\") and "&" in line:
            line = line.rstrip() + r" \hline"
        out.append(line)
    return "\n".join(out)


def disp(lab):
    """Название вида головы для таблиц — одно и то же в таблице видов и в таблицах по слоям."""
    sink = lab.endswith(" + сток")
    b = lab[:-len(" + сток")] if sink else lab
    name = "Дальний контекст (равномерное усреднение)" if b == "равномерное усреднение" else b[0].upper() + b[1:]
    return name + (" + сток" if sink else "")


def tex(s):
    out = []
    for ch in str(s):
        if ch == "\\":
            out.append(r"\textbackslash{}")
        elif ch in "&%$#_{}":
            out.append("\\" + ch)
        elif ch == "~":
            out.append(r"\textasciitilde{}")
        elif ch == "^":
            out.append(r"\textasciicircum{}")
        elif ch == " ":
            out.append(r"\textvisiblespace{}")
        elif ch == "-":
            out.append("{-}")
        elif ord(ch) < 32 or (ord(ch) > 0x24F and not 0x400 <= ord(ch) <= 0x4FF) or ch == "�":
            out.append("?")
        else:
            out.append(ch)
    return "".join(out)


def ranges(ls):
    """Список слоёв диапазонами: [0, 1, 2, 3, 5, 7, 8] → «0–3, 5, 7, 8» (подряд идущие три и больше — через тире)."""
    ls = sorted(set(ls))
    if not ls:
        return "—"
    out, a = [], 0
    for i in range(1, len(ls) + 1):
        if i == len(ls) or ls[i] != ls[i - 1] + 1:
            run = ls[a:i]
            out.append(rf"\mbox{{{run[0]}–{run[-1]}}}" if len(run) >= 3 else ", ".join(map(str, run)))  # диапазон не рвётся
            a = i
    return ", ".join(out)


def tok_list(items, n=3):
    return ", ".join(r"\texttt{" + tex(i.rsplit(" (", 1)[0]) + "}" for i in items[:n])


def num(v, nd=2):
    return f"{v + 0.0:.{nd}f}".replace("-0." + "0" * nd, "0." + "0" * nd).replace(".", ",")


def sgn(v, nd=1):
    t = f"{abs(v):.{nd}f}"
    return t.replace(".", ",") if float(t) == 0 else ("$+$" if v >= 0 else "$-$") + t.replace(".", ",")


def figures(res, name):
    H, NL = res["H"], res["layers"]
    hs = res["heads"]
    by = [[x for x in hs if x["layer"] == l] for l in range(NL)]
    xs = list(range(NL))

    def av(l, f):
        return sum(f(x) for x in by[l]) / H

    # --- рис. 1: куда уходит внимание и что делают головы
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 4.4), sharex=True)
    ax = axes[0]
    parts = [("сам и предыдущий токен (Δ ≤ 1)", lambda x: x["text"]["mass_by_distance"]["0"] + x["text"]["mass_by_distance"]["1"], SEQ[0]),
             ("2–8 токенов назад", lambda x: near8(x) - x["text"]["mass_by_distance"]["0"] - x["text"]["mass_by_distance"]["1"], SEQ[1]),
             ("9–64", mid, SEQ[2]),
             ("дальше 64 (без позиции 0)", lambda x: max(x["text"]["mass_by_distance"]["65+"] - x["text"]["pos0"], 0.0), SEQ[3]),
             ("позиция 0 (сток)", lambda x: x["text"]["pos0"], PAL[1])]
    bottom = [0.0] * NL
    for lab, f, col in parts:
        vals = [av(l, f) for l in xs]
        ax.bar(xs, vals, bottom=bottom, color=col, width=0.78, edgecolor="white", linewidth=0.8, label=lab)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("доля внимания")
    ax.set_title("Куда смотрят головы: средняя по слою доля внимания по расстоянию", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=7.5, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.24), frameon=False, handlelength=1.2, columnspacing=1.2)
    ax = axes[1]
    counts = [collections.Counter(classify(x)[0] for x in by[l]) for l in xs]
    bottom = [0] * NL
    for c in CATS:
        vals = [counts[l][c] for l in xs]
        ax.bar(xs, vals, bottom=bottom, color=CAT_COLOR[c], width=0.78, edgecolor="white", linewidth=0.8, label=c[0].upper() + c[1:])
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylim(0, H)
    ax.set_yticks(range(0, H + 1, 2))
    ax.set_ylabel(f"число голов (из {H})")
    ax.set_xlabel("слой")
    ax.set_xticks(xs)
    ax.set_title("Что делают головы: число голов каждого типа в слое", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=7.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.24), frameon=False, handlelength=1.2, columnspacing=1.2)
    for a in axes:
        a.tick_params(labelsize=7.5)
        a.grid(axis="x", visible=False)
        a.set_axisbelow(True)
        a.margins(x=0.01)
    fig.tight_layout(h_pad=2.2)
    fig.savefig(OUT / f"heads_{name}_fig1.pdf")
    plt.close(fig)

    # --- рис. 2: тенденции по глубине (малые панели, у каждой своя ось)
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.0), sharex=True)

    def panel(ax, title, series, ylabel=None, zero=False):
        for (lab, vals), col in zip(series, [PAL[0], PAL[1], PAL[2], PAL[3]]):
            ax.plot(xs, vals, color=col, lw=1.6, marker="o", ms=2.6, label=lab)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=8)
        if zero:
            ax.axhline(0, color=INK2, lw=0.8)
        if len(series) > 1:
            ax.legend(fontsize=7, frameon=False, handlelength=1.4, loc="best")
        ax.tick_params(labelsize=7.5)
        ax.set_xticks(range(0, NL, 3))

    panel(axes[0][0], "Спад главной частоты на 512 токенах", [("", [av(l, lambda x: x["kernel"]["dominant_freq_change_over_512"]) for l in xs])], "логиты", zero=True)
    panel(axes[0][1], "Доля запроса и ключа, объяснимая токеном (R²)",
          [("запрос", [av(l, lambda x: x["terms_std_logits"]["token_r2_q"]) for l in xs]),
           ("ключ", [av(l, lambda x: x["terms_std_logits"]["token_r2_k"]) for l in xs])])
    axes[0][1].set_ylim(0, 1.05)
    panel(axes[0][2], "Источники главной частоты запроса",
          [(k, [av(l, lambda x, k=k: x["sources"][0]["q"]["agg"][k]) for l in xs]) for k in ("эмб", "внимание", "MLP")], "доля")
    axes[0][2].legend(["эмбеддинг", "головы внимания", "MLP"], fontsize=7, frameon=False, handlelength=1.4)
    panel(axes[1][0], "Разброс слагаемых логита",
          [(n, [av(l, lambda x, k=k: x["terms_std_logits"][k]) for l in xs])
           for n, k in (("позиция", "position"), ("заметность ключа", "key_salience"), ("запрос", "query_modulation"), ("пара", "pair"))], "логиты")
    nw = res["norm_weights"]
    panel(axes[1][1], "Веса нормализации запросов и ключей",
          [("k_norm, максимум", [abs(nw[l]["k_norm"]["top_vals"][0]) for l in xs]),
           ("q_norm, максимум", [abs(nw[l]["q_norm"]["top_vals"][0]) for l in xs]),
           ("k_norm, медиана", [nw[l]["k_norm"]["median_abs"] for l in xs])])
    axes[1][1].set_ylim(0, None)
    panel(axes[1][2], "Эффективное число ключей", [("", [av(l, lambda x: x["text"]["effective_keys"]) for l in xs])])
    axes[1][2].set_ylim(0, None)
    for a in axes[1]:
        a.set_xlabel("слой", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"heads_{name}_fig2.pdf")
    plt.close(fig)
    return by, counts


def build(args):
    name = args.model
    base = ROOT / "results" / name / "heads"
    res = json.loads((base / "all_layers.json").read_text())
    ramp = json.loads((base / "ramp_check.json").read_text()) if (base / "ramp_check.json").exists() else None
    circ = json.loads((base / "circuits.json").read_text())  # head_circuits.py: откуда берётся выбор головы
    mw = json.loads((base / "mlp_writers.json").read_text())  # mlp_writers.py: какие нейроны MLP пишут сток
    fa = json.loads((base / "freq_adv.json").read_text())  # head_freq_adv.py: какие частоты несут выбор головы
    l0 = json.loads((base / "layer0.json").read_text())  # head_mechanisms.py: слой 0 по весам
    H, G, NL = res["H"], res["G"], res["layers"]
    by, counts = figures(res, name.replace(".", ""))
    fname = name.replace(".", "")
    hs = res["heads"]
    M = res["D"] // 2
    nw = res["norm_weights"]
    kfreq = collections.Counter(nw[l]["k_norm"]["top_freq"][0] for l in range(NL))
    qfreq = collections.Counter(nw[l]["q_norm"]["top_freq"][0] for l in range(NL))
    mk, mk_n = kfreq.most_common(1)[0]
    dom = collections.Counter(x["kernel"]["top_freqs"][0]["m"] for x in hs)
    per = res["rope_period_tokens"]

    def av(l, f):
        return sum(f(x) for x in by[l]) / H

    def heads_where(cond, val):
        return ", ".join(f"{x['layer']}.{x['head']} ({num(val(x))})" for x in hs if cond(x))

    def is_kind(x, kind):  # вид головы без приписки «+ сток»
        return classify(x)[1].replace(" + сток", "") == kind

    prev = [x for x in hs if is_kind(x, "предыдущий токен")]
    prev_fast = sum(1 for x in prev if any(f["period"] < 8 and 0.5 < f["peak_delta"] < 1.5 for f in x["kernel"]["top_freqs"]))
    def layers_of(cond):
        return ranges(x["layer"] for x in hs if cond(x))

    first_sink = next((l for l in range(NL) if av(l, lambda x: x["text"]["pos0"]) > 0.05), None)
    max_sink = max(x["text"]["pos0"] for x in hs)

    PRE = [r"\documentclass[10pt]{article}", r"\usepackage{fontspec}", r"\setmainfont{Times New Roman}",
           r"\setmonofont[Scale=0.85]{Menlo}", r"\usepackage[russian]{babel}", r"\usepackage{amsmath}",
           r"\usepackage[a4paper,landscape,margin=1.3cm]{geometry}", r"\usepackage{graphicx,xcolor,colortbl,array,enumitem}",
           r"\arrayrulecolor[HTML]{B5B4AB}\definecolor{head}{HTML}{F0EFEC}",
           r"\setlist{nosep,leftmargin=1.3em,itemsep=0.25em}", r"\setlength{\parindent}{0pt}\setlength{\parskip}{0.45em}",
           r"\setlength{\tabcolsep}{3.2pt}\renewcommand{\arraystretch}{1.08}", r"\definecolor{ink2}{HTML}{52514E}",
           r"\begin{document}", r"\selectlanguage{russian}"]
    SUB = (rf"{{\color{{ink2}} {NL} слоёв $\times$ {H} голов $=$ {NL * H} голов. Головы делят ключи и значения: групп {G}, по {H // G} в каждой. "
           rf"Текст для проверки: {res['text_tokens']} токенов теста WikiText-2 (окна по 512 токенов).}}")

    def title(t):
        return [rf"{{\LARGE\bfseries {t}}}\\[0.2em]", SUB, ""]

    T = []  # разделы помечаются строками %%SEC:<имя> и в конце раскладываются по двум файлам

    # ---------- механизмы видов голов из точного разложения по источникам (circuits.json)
    ch = {(x["layer"], x["head"]): x for x in circ["heads"]}
    hrec = {(x["layer"], x["head"]): x for x in hs}

    def cadv(x, typ):
        return ch[(x["layer"], x["head"])]["adv"][typ]

    def agg_w(xs, typ, side):
        """Доли источников (эмб, головы, MLP), усреднённые по головам с весом преимущества."""
        tot = sum(cadv(x, typ)["logits"] for x in xs) or 1e-9
        return [sum(cadv(x, typ)[side]["agg"][k_] * cadv(x, typ)["logits"] for x in xs) / tot for k_ in ("эмб", "внимание", "MLP")]

    def src_is_prev(nm):
        if not nm.startswith("L"):
            return False
        l_, h_ = nm[1:].split(".H")
        return is_kind(hrec[(int(l_), int(h_))], "предыдущий токен")

    def mean(v):
        v = list(v)
        return sum(v) / max(len(v), 1)

    K = {k_: [x for x in hs if is_kind(x, k_)] for k_ in ("предыдущий токен", "сам токен", "тот же токен", "индукция")}
    K["сток"] = [x for x in hs if classify(x)[1].endswith("+ сток")]
    TYP = {"предыдущий токен": "prev", "сам токен": "self", "тот же токен": "same", "индукция": "ind", "сток": "sink"}
    fh = {(x["layer"], x["head"]): x for x in fa["heads"]}

    def whole(v):  # доля источника словами: больше единицы бывает, когда остальные источники вычитают
        return "целиком" if v > 0.995 else f"на {num(v)}"

    prev_fq = sum(1 for x in K["предыдущий токен"]
                  if fh[(x["layer"], x["head"])]["adv"]["prev"]["fast_logits"] >= 0.25 * fh[(x["layer"], x["head"])]["adv"]["prev"]["logits"])
    cos_s = {k_: mean(ch[(x["layer"], x["head"])]["cos_qk_same_pos"] for x in v) for k_, v in K.items()}
    cos_o = {k_: mean(ch[(x["layer"], x["head"])]["cos_qk_other_pos"] for x in v) for k_, v in K.items()}
    adv_m = {k_: mean(cadv(x, TYP[k_])["logits"] for x in v) for k_, v in K.items()}
    ind_top = [(x, cadv(x, "ind")["key"]["top"][0]) for x in K["индукция"]]
    ind_prev = [(x, t_) for x, t_ in ind_top if src_is_prev(t_[0])]
    ind_ex = ", ".join(f"{x['layer']}.{x['head']} ← {t_[0][1:].replace('.H', '.')} ({num(t_[1])})" for x, t_ in ind_prev[:4])
    ind_k, ind_q = agg_w(K["индукция"], "ind", "key"), agg_w(K["индукция"], "ind", "query")
    same_k, same_q = agg_w(K["тот же токен"], "same", "key"), agg_w(K["тот же токен"], "same", "query")
    self_common = mean(len({n_ for n_, _ in cadv(x, "self")["key"]["top"][:5]} & {n_ for n_, _ in cadv(x, "self")["query"]["top"][:5]}) for x in K["сам токен"])
    sl = mw["sink_layer"]
    sink_top = collections.Counter(cadv(x, "sink")["key"]["top"][0][0] for x in K["сток"])
    p0 = circ["pos0"][sl + 1]

    def fshare(xs, typ, key):
        tot = sum(fh[(x["layer"], x["head"])]["adv"][typ]["logits"] for x in xs) or 1e-9
        return sum(fh[(x["layer"], x["head"])]["adv"][typ][key] for x in xs) / tot

    fast = {k_: fshare(v, TYP[k_], "fast_logits") for k_, v in K.items()}  # частоты с периодом < 20 токенов
    slow = {k_: fshare(v, TYP[k_], "slow_logits") for k_, v in K.items()}  # частоты с периодом > 20 000 токенов
    ramp_eff = mean(ramp["k_to_median"]["far256"]) - mean(ramp["original"]["far256"])
    ramp_like = ramp_eff > 0.15  # большой вес нормализации управляет расстоянием (MNN-Pocket) или нет (Qwen3-0.6B)
    q_ch = max(abs(h_["salience"]["mean_q_on_channel"]) + h_["salience"]["std_q_on_channel"] for h_ in l0["heads"])
    self0 = sum(1 for x in K["сам токен"] if x["kernel"]["argmax_delta"] == 0)
    align = cos_s["сам токен"] - cos_o["сам токен"] >= 0.05
    mech_self = (rf"У {self0} из {len(K['сам токен'])} таких голов пик позиционного ядра приходится на $\Delta{{=}}0$: быстрые частоты стоят без сдвига "
                 rf"и дают {num(fast['сам токен'])} преимущества цели. "
                 + (rf"Кроме того, запрос и ключ читают одни и те же направления остаточного потока: косинус между ними {num(cos_s['сам токен'])} "
                    rf"на своей позиции против {num(cos_o['сам токен'])} на чужой. " if align else "")
                 + rf"Цель получает в среднем {sgn(adv_m['сам токен'])} логита.")
    mech_same = (rf"Запрос и ключ собраны из одних и тех же источников, которые несут сам токен (эмбеддинг / головы / MLP: "
                 rf"{num(same_q[0])} / {num(same_q[1])} / {num(same_q[2])} у запроса и {num(same_k[0])} / {num(same_k[1])} / {num(same_k[2])} у ключа). "
                 rf"У одинаковых токенов они совпадают; совпадение идёт по медленным частотам ({num(slow['тот же токен'])} преимущества), "
                 rf"поэтому не зависит от расстояния. Прошлое вхождение получает {sgn(adv_m['тот же токен'])} логита.")
    mech_ind = (rf"Ключ {whole(ind_k[1])} приходит из предыдущих голов, главным образом из голов «Предыдущий токен» ({ind_ex}): они записывают в позицию, "
                rf"какой токен стоял перед ней. Запрос несёт сам текущий токен (эмбеддинг / головы / MLP: {num(ind_q[0])} / {num(ind_q[1])} / {num(ind_q[2])}). "
                rf"Совпадение даёт цели {sgn(adv_m['индукция'])} логита.")
    n90 = mw["sink"]["n_for_90"]
    neu = ", ".join(str(n_) for n_ in mw["sink"]["neurons"][:max(1, min(n90, 5))])
    act_lo, act_hi = min(mw["sink"]["act_pos0_over_typical"][:max(1, min(n90, 5))]), max(mw["sink"]["act_pos0_over_typical"][:max(1, min(n90, 5))])
    who = (rf"Один нейрон MLP слоя {sl} (номер {neu}) на позиции 0 срабатывает в {act_hi:.0f} раз сильнее обычного и даёт {num(mw['sink']['top1'])} записи MLP в этой позиции."
           if n90 == 1 else
           rf"{n90} нейронов MLP слоя {sl} (номера {neu}) на позиции 0 срабатывают в {act_lo:.0f}–{act_hi:.0f} раз сильнее обычного и дают 0,90 записи MLP в этой позиции.")
    mech_sink = (who + rf" Норма остаточного потока на позиции 0 вырастает с {num(mw['rms_ratio_before'], 1)} до "
                 rf"{num(mw['rms_ratio_after'], 1)} от обычной, {num(p0['top_dims_share'])} энергии в координатах {', '.join(map(str, p0['top_dims']))}. "
                 rf"Проекция ключей читает этот вектор на медленных частотах ({num(slow['сток'])} преимущества): "
                 rf"у {sink_top[f'MLP{sl}']} из {len(K['сток'])} голов со стоком главный источник ключа позиции 0~— MLP слоя {sl}.")
    def kdrop(kind):  # на сколько логитов ядро на расстоянии 1 выше, чем на расстоянии 64 (среднее по головам вида)
        xs = [x for x in hs if is_kind(x, kind) or (kind == "дальний контекст" and is_kind(x, "равномерное усреднение"))]
        return mean(x["kernel"]["P"]["1"] - x["kernel"]["P"]["64"] for x in xs)

    mech_win8 = (rf"Позиционное ядро круто падает с расстоянием: логит на расстоянии 1 выше, чем на расстоянии 64, в среднем на {num(kdrop('локальное окно ≤ 8'), 1)} логита."
                 + (" Спад задаёт медленная частота, усиленная весом нормализации." if ramp_like else ""))
    mech_win64 = rf"То же ядро, но положе: {num(kdrop('окно ≤ 64'), 1)} логита между расстояниями 1 и 64."
    mech_far = rf"Ядро почти плоское: {num(kdrop('дальний контекст'), 1)} логита между расстояниями 1 и 64."

    # короткие формулировки для таблицы видов (подробные — во втором файле)
    short_self = (rf"У {self0} из {len(K['сам токен'])} голов пик позиционного ядра стоит на $\Delta{{=}}0$ (быстрые частоты без сдвига)"
                  + (rf"; запрос и ключ читают одни направления: косинус {num(cos_s['сам токен'])} на своей позиции против {num(cos_o['сам токен'])} на чужой." if align else "."))
    short_same = rf"Запрос и ключ несут сам токен и у одинаковых токенов совпадают; совпадение идёт по медленным частотам ({num(slow['тот же токен'])})."
    short_ind = rf"Ключ {whole(ind_k[1])} приходит из предыдущих голов, главным образом «Предыдущий токен»; запрос~— сам текущий токен."
    short_sink = ((rf"Нейрон {neu} MLP слоя {sl} срабатывает на первом токене в {act_hi:.0f} раз сильнее обычного" if n90 == 1 else
                   rf"{n90} нейронов MLP слоя {sl} срабатывают на первом токене в {act_lo:.0f}–{act_hi:.0f} раз сильнее обычного")
                  + "; проекция ключей читает их запись на медленных частотах.")

    # ---------- 1. виды голов
    labs = [classify(x)[1] for x in hs]
    base = [lb.replace(" + сток", "") for lb in labs]

    def kind_row(names, title, what, mech):
        idx = [i for i, b_ in enumerate(base) if b_ in names]
        ls = sorted({hs[i]["layer"] for i in idx})
        if not idx:
            return ""
        return rf"{chip(KIND_CAT[names[0]])}\textbf{{{title}}} & {what} & {mech} & {len(idx)} & {ranges(ls)} \\"

    n_sink = sum(1 for lb in labs if lb.endswith("+ сток"))
    T_note = "Цветной квадрат перед названием — цвет этого вида на рисунке 1 и в таблицах по слоям."
    sink_layers = sorted({x["layer"] for x, lb in zip(hs, labs) if lb.endswith("+ сток")})
    T += ["%%SEC:kinds", r"\section*{1. Виды голов}",
          rf"Каждая из {NL * H} голов получила одну метку по тому, куда она на самом деле смотрит на тексте. Правила меток и пороги~— в разделе 2. " + T_note,
          r"\begin{center}{\small\begin{tabular}{|p{4.3cm}|p{8.9cm}|p{7.3cm}|r|p{3.0cm}|}", r"\toprule",
          r"\rowcolor{head}\textbf{Вид головы} & \textbf{Что делает} & \textbf{Механизм в весах} & \textbf{Голов} & \textbf{В каких слоях} \\ \midrule",
          kind_row(["предыдущий токен"], "Предыдущий токен", "Смотрит на токен прямо перед текущим: передаёт каждому токену сведения о соседе слева.",
                   rf"Быстрые частоты RoPE (период меньше 20 токенов) со сдвигом пика на один токен: они дают {num(fast['предыдущий токен'])} преимущества цели."),
          kind_row(["сам токен"], "Сам токен", "Смотрит на текущий токен: ещё раз пропускает через себя его собственное состояние.", short_self),
          kind_row(["локальное окно ≤ 8"], "Локальное окно ≤ 8", "Усредняет несколько ближайших токенов: собирает словосочетание вокруг текущего места.",
                   mech_win8),
          kind_row(["окно ≤ 64"], "Окно ≤ 64", "Усредняет ближайшие десятки токенов: предложение или абзац.", mech_win64),
          kind_row(["тот же токен"], "Тот же токен", "Находит прошлые вхождения того же токена, что стоит сейчас.",
                   short_same),
          kind_row(["индукция"], "Индукция", "Находит токен, который в прошлый раз шёл сразу после текущего: подсказывает продолжение повторяющегося фрагмента.", short_ind),
          kind_row(["собирает определённые токены"], "Собирает определённые токены",
                   "Смотрит на токены определённого вида, где бы они ни стояли и кто бы ни спрашивал. Например, собирает служебные слова со всего контекста.",
                   r"Прибавка к логиту зависит только от токена-ключа (разброс $\ge 2{,}5$ логита)."),
          kind_row(["дальний контекст", "равномерное усреднение"], "Дальний контекст",
                   "Размазывает внимание по всему тексту: даёт среднее по контексту. Если внимание размазано больше чем по 80 токенам, "
                   "к названию дописано «(равномерное усреднение)».",
                   mech_far),
          rf"\midrule \textbf{{Приписка «+ сток»}} & Больше 20\,\% внимания головы уходит на первый токен текста. Туда голова сбрасывает внимание, "
          rf"когда смотреть не на что. Приписка добавляется к любому виду. & {short_sink} & {n_sink} & {ranges(sink_layers)} \\",
          r"\bottomrule\end{tabular}}\end{center}"]

    # ---------- 1. метод
    T += ["%%SEC:method", r"\section*{4. Как получены результаты}",
          r"Логит внимания любой головы точно равен сумме по частотам RoPE:",
          r"\[ s_h(i,j) \;=\; \frac{1}{\sqrt{D}}\,\mathrm{Re}\sum_{m} z^{q}_{h,m}(i)\,\overline{z^{k}_{g,m}(j)}\,e^{\,i\theta_m (i-j)}, \qquad \theta_m = 10^{-12m/D}, \]",
          r"где $z^q$, $z^k$~— запрос и ключ после нормализации (QK-norm), записанные комплексными числами по частотам "
          r"($m$-я частота~— компоненты $m$ и $m+D/2$), $\Delta=i-j$~— расстояние между токенами. "
          r"В слое 0 запрос и ключ зависят только от токена, дальше~— от контекста, поэтому берутся из прогона текста.",
          r"\begin{itemize}",
          rf"\item \textbf{{Внимание пересчитано по этой формуле}} и совпадает с выходом модели: относительная ошибка не больше "
          rf"${res['recon_err_max']:.0e}".replace("e-0", r"\cdot 10^{-").replace("e-", r"\cdot 10^{-") + r"}$.",
          r"\item \textbf{Источники запроса и ключа найдены точно.} Вход слоя~— сумма эмбеддинга и записей всех предыдущих голов и MLP "
          r"в остаточный поток. Нормализация и проекции линейны с точностью до одного множителя на позицию, поэтому каждая компонента "
          r"запроса и ключа раскладывается по источникам без приближений (ошибка суммы не больше "
          rf"${res['attr_err_max']:.0e}".replace("e-0", r"\cdot 10^{-").replace("e-", r"\cdot 10^{-") + r"}$).",
          r"\item \textbf{Позиционное ядро} головы~— логит при средних запросе и ключе как функция расстояния. Оно показывает, "
          r"как голова смотрит на расстояние независимо от токенов, и раскладывается по частотам: у каждой частоты амплитуда и сдвиг пика.",
          r"\end{itemize}"]

    # ---------- 2. пояснения к столбцам
    T += ["%%SEC:cols", r"\section*{2. Пояснения к столбцам таблиц}",
          r"\begin{center}\begin{tabular}{|p{3.3cm}|p{22.0cm}|}", r"\toprule \rowcolor{head}\textbf{Столбец} & \textbf{Что означает} \\ \midrule",
          r"h & Номер головы в слое. Головы " + f"0–{H // G - 1}" + " и " + f"{H // G}–{H - 1}" + r" используют разные таблицы ключей и значений. \\",
          r"Вид головы & Название из таблицы видов (раздел 1). Ставится автоматически по измеренному вниманию, правила ниже. \\",
          r"$\Delta{=}0$, $\Delta{=}1$, 2–8, 9–64, 65+ & Доля внимания головы на токен с таким расстоянием назад: 0~— сам токен, 1~— предыдущий. "
          r"Среднее по запросам, у которых есть хотя бы 64 предыдущих токена. В столбце 65+ позиция 0 не учитывается. \\",
          r"поз.\,0 & Доля внимания на самый первый токен текста. Это сток: туда голова сбрасывает внимание, когда смотреть не на что. "
          r"Шесть столбцов от $\Delta{=}0$ до поз.\,0 в сумме дают 1. \\",
          r"повтор & Доля внимания на прошлые вхождения того же токена, что и запрос. В скобках~— сколько получилось бы при равномерном внимании. \\",
          r"индукц. & Доля внимания на токены, стоящие сразу после прошлых вхождений текущего токена (так модель продолжает повторяющийся фрагмент). "
          r"В скобках~— уровень при равномерном внимании. \\",
          r"ключей & Эффективное число ключей: на скольких токенах размазано внимание (экспонента энтропии). 1~— смотрит в одну точку. \\",
          r"частота (период) & Частота RoPE с наибольшей амплитудой в позиционном ядре головы и её период в токенах. \\",
          r"спад & На сколько логитов главная частота меняет логит при удалении ключа на 512 токенов. Минус~— дальние токены получают меньше внимания. \\",
          r"$\sigma$ ключа & Разброс логита (в логитах) из-за того, какой токен стоит в ключе, при среднем запросе: насколько голова выбирает ключи по их токену. \\",
          r"$R^2$ & Доля разброса запроса, которую объясняет сам токен в этой позиции (1~— запрос зависит только от токена, как в слое 0). \\",
          r"притягивает & Токены-ключи, которые получают наибольшую прибавку к логиту (по токенным средним); показаны при $\sigma$ ключа $\ge 1$. "
          r"Знак \texttt{\textvisiblespace}~— пробел в начале токена. \\",
          r"\bottomrule\end{tabular}\end{center}",
          r"\textbf{Правила меток.} Метка описывает ту часть внимания, которая не ушла в сток: доли по расстоянию делятся на (1 $-$ поз.\,0). "
          r"Правила проверяются по порядку: \emph{Индукция}~— на индукционные цели больше 0,15 и в 5 раз больше равномерного; "
          r"\emph{Тот же токен}~— на повторы больше 0,2 и в 5 раз больше равномерного; \emph{Предыдущий токен}~— $\Delta{=}1$ больше 0,4; "
          r"\emph{Сам токен}~— $\Delta{=}0$ больше 0,4; \emph{Локальное окно ≤ 8}~— в пределах 8 токенов больше 0,6; "
          r"\emph{Окно ≤ 64}~— в пределах 64 токенов больше 0,75; \emph{Собирает определённые токены}~— $\sigma$ ключа не меньше 2,5; "
          r"иначе \emph{Дальний контекст}; если ключей больше 80, к названию дописано «(равномерное усреднение)». Приписка «+ сток»~— на позицию 0 уходит больше 0,2."]

    # ---------- 3. тенденции
    r2 = [av(l, lambda x: x["terms_std_logits"]["token_r2_q"]) for l in range(NL)]
    sink_l = [av(l, lambda x: x["text"]["pos0"]) for l in range(NL)]
    srcq = [[av(l, lambda x, k=k: x["sources"][0]["q"]["agg"][k]) for k in ("эмб", "внимание", "MLP")] for l in range(NL)]
    rampv = [av(l, lambda x: x["kernel"]["dominant_freq_change_over_512"]) for l in range(NL)]
    strong = sorted(range(NL), key=lambda l: rampv[l])[:2]
    n_pos = sum(1 for x in hs if any(is_kind(x, k_) for k_ in ("локальное окно ≤ 8", "окно ≤ 64", "предыдущий токен", "сам токен")))
    n_far = sum(1 for x in hs if is_kind(x, "дальний контекст") or is_kind(x, "равномерное усреднение"))
    n_cont = sum(1 for x in hs if classify(x)[0] in ("тот же токен", "индукция", "собирает определённые токены"))
    dm, dn = dom.most_common(1)[0]
    kmax = [abs(nw[l]["k_norm"]["top_vals"][0]) for l in range(NL)]
    kmed = sum(nw[l]["k_norm"]["median_abs"] for l in range(NL)) / NL
    # ---------- главное (коротко) и графики
    S = [rf"\item \textbf{{Большинство голов смотрит по расстоянию, а не по смыслу.}} Из {NL * H} голов {n_pos} смотрят на заданное расстояние "
         rf"(сам токен, предыдущий, окно), {n_far} усредняют дальний контекст и только {n_cont} выбирают токены по содержанию.",
         rf"\item \textbf{{Быстрые частоты RoPE кодируют расстояние, медленные~— содержание.}} «Предыдущий токен» берёт {num(fast['предыдущий токен'])} преимущества цели "
         rf"на частотах с периодом меньше 20 токенов. «Тот же токен», индукция и сток берут {num(slow['тот же токен'])}, {num(slow['индукция'])} и {num(slow['сток'])} "
         r"на частотах с периодом больше 20 тысяч токенов."]
    o, a_ = ramp["original"], ramp["k_to_median"]
    if ramp_like:
        S += [rf"\item \textbf{{Спад с расстоянием задаёт один вес нормализации.}} Частота $m{{=}}{dm}$ (период {per[dm]:.0f} токенов) главная у {dn} голов; "
              rf"её усиливает вес нормализации ключей (до {num(max(kmax), 1)} при медиане {num(kmed, 1)}). Если в слое 0 заменить его на медиану, все {H} голов "
              rf"начинают смотреть далеко: доля внимания дальше 256 токенов растёт с {num(mean(o['far256']))} до {num(mean(a_['far256']))} в среднем."]
    else:
        S += [rf"\item \textbf{{Самый большой вес нормализации не работает.}} В слое 0 вес ключа на компоненте {ramp['k_dim']} равен {num(ramp['k_gamma'], 1)} "
              rf"(медиана {num(ramp['k_median'], 1)}), но запрос на этой компоненте почти нулевой (не больше {num(q_ch)}) у всех {H} голов. "
              rf"Замена веса на медиану меняет долю внимания дальше 256 токенов с {num(mean(o['far256']))} до {num(mean(a_['far256']))} в среднем."]
    S += [rf"\item \textbf{{«Предыдущий токен» держится на быстрых частотах на любой глубине.}} Таких голов {len(prev)} (слои {layers_of(lambda x: is_kind(x, 'предыдущий токен'))}), "
          rf"у {prev_fq} из них частоты с периодом меньше 20 токенов дают не меньше четверти преимущества цели"
          + (rf"; у {prev_fast} самая быстрая частота сдвинута ровно на один токен." if prev_fast == len(prev) else "."),
          rf"\item \textbf{{Сток создают отдельные нейроны MLP.}} " + who + rf" Норма остаточного потока там становится в {num(mw['rms_ratio_after'], 0)} раз больше, "
          rf"чем на других позициях. Сток заметен со слоя {first_sink}, приписку «+ сток» имеют {len(K['сток'])} голов из {NL * H}; в слоях {NL - 10}–{NL - 1} на первый токен уходит {num(min(sink_l[NL - 10:]))}–{num(max(sink_l[NL - 10:]))} внимания.",
          rf"\item \textbf{{Индукция собрана из двух голов.}} Ключ индукционной головы {whole(ind_k[1])} приходит из предыдущих голов "
          rf"(у {len(ind_prev)} из {len(ind_top)} главный источник~— голова «Предыдущий токен»), запрос~— сам текущий токен.",
          rf"\item \textbf{{Чем глубже, тем меньше решает сам токен.}} Он объясняет {num(r2[0])} запроса в слое 0, {num(r2[min(5, NL - 1)])} в слое 5 и "
          rf"{num(min(r2))}–{num(max(r2[min(12, NL - 1):]))} после слоя 12.",
          rf"\item \textbf{{Главную частоту запроса питают сначала MLP, потом головы.}} В слоях 1–15 до {num(max(s_[2] for s_ in srcq[1:16]))} главной частоты приходит из MLP "
          rf"предыдущих слоёв, в последних слоях до {num(srcq[NL - 1][1])}~— из голов внимания."]
    T += ["%%SEC:summary", r"\section*{1. Главное}", r"\begin{itemize}", *S, r"\end{itemize}"]
    T += ["%%SEC:figs", r"\section*{2. Графики}",
          r"\noindent\begin{minipage}{\linewidth}",
          r"\textbf{Рисунок 1.} Сверху~— куда в среднем уходит внимание голов слоя; снизу~— сколько голов каждого вида в слое.",
          rf"\begin{{center}}\includegraphics[width=0.76\linewidth]{{heads_{fname}_fig1.pdf}}\end{{center}}", r"\end{minipage}", r"\par",
          r"\noindent\begin{minipage}{\linewidth}",
          r"\textbf{Рисунок 2.} Средние по головам слоя: спад с расстоянием, доля запроса и ключа от токена, источники главной частоты, "
          r"разброс слагаемых логита, веса нормализации, число ключей.",
          rf"\begin{{center}}\includegraphics[width=0.78\linewidth]{{heads_{fname}_fig2.pdf}}\end{{center}}", r"\end{minipage}", r"\par"]

    # ---------- механизм в весах
    T += ["%%SEC:mech", r"\section*{3. Механизм в весах}"]
    part = "косинусная" if ramp["k_dim"] < M else "синусная"
    partq = "косинусная" if ramp["q_dim"] < M else "синусная"
    if ramp_like:
        lead = [rf"\textbf{{Спад с расстоянием~— одна пара весов нормализации.}} В слое 0 вес k\_norm на компоненте {ramp['k_dim']} равен "
                rf"{num(ramp['k_gamma'], 1)}, вес q\_norm на компоненте {ramp['q_dim']}~— {num(ramp['q_gamma'], 1)}; медиана остальных {num(ramp['k_median'], 1)}. "
                rf"Это {part} часть ключа и {partq} часть запроса одной частоты $m{{=}}{ramp['freq']}$. "
                r"Их произведение даёт вклад в логит $-A\sin(\theta_m\Delta)$: почти линейный спад с расстоянием $\Delta$. У голов отличается только сила $A$. "
                r"Ниже~— что происходит с головами слоя 0, если вес k\_norm заменить на медиану (запросы с контекстом 1024–2048 токенов)."]
    else:
        lead = [rf"\textbf{{Самый большой вес нормализации не управляет вниманием.}} В слое 0 вес k\_norm на компоненте {ramp['k_dim']} равен "
                rf"{num(ramp['k_gamma'], 1)} при медиане {num(ramp['k_median'], 1)}. Это {part} часть частоты $m{{=}}{ramp['freq']}$ с периодом {ramp['period_tokens']:.0f} токенов: "
                r"на длине контекста она не зависит от расстояния. Значения ключа на этой компоненте большие (сотни), но запрос на ней почти нулевой "
                rf"(не больше {num(q_ch)} у всех {H} голов), поэтому в логит компонента почти ничего не даёт. "
                r"Ниже~— что происходит с головами слоя 0, если этот вес заменить на медиану (запросы с контекстом 1024–2048 токенов): почти ничего."]
    T += [r"\noindent\begin{minipage}{\linewidth}", *lead,
          r"\begin{center}\begin{tabular}{|l|" + "r|" * H + r"}\toprule \rowcolor{head}\textbf{Голова} & " + " & ".join(rf"\textbf{{{h}}}" for h in range(H)) + r" \\ \midrule",
          r"Дальше 256 токенов, исходные веса & " + " & ".join(num(v) for v in o["far256"]) + r" \\",
          r"Дальше 256 токенов, вес заменён & " + " & ".join(num(v) for v in a_["far256"]) + r" \\",
          r"Среднее расстояние, исходные веса & " + " & ".join(f"{v:.0f}" for v in o["mean_distance"]) + r" \\",
          r"Среднее расстояние, вес заменён & " + " & ".join(f"{v:.0f}" for v in a_["mean_distance"]) + r" \\ \bottomrule\end{tabular}\end{center}",
          r"\end{minipage}", r"\par"]
    rows_f = []
    for k_ in ("предыдущий токен", "сам токен", "тот же токен", "индукция", "сток"):
        if K[k_]:
            mid_ = 1 - fast[k_] - slow[k_]
            rows_f.append(rf"{disp(k_) if k_ != 'сток' else 'Приписка «+ сток»'} & {len(K[k_])} & {sgn(adv_m[k_])} & {num(fast[k_])} & {num(mid_)} & {num(slow[k_])} \\")
    T += [r"\noindent\begin{minipage}{\linewidth}",
          r"\textbf{Какие частоты несут выбор.} Логит~— сумма по частотам RoPE, поэтому преимущество цели точно делится между частотами. "
          r"Быстрые частоты поворачиваются на длине нескольких токенов и различают расстояние; медленные на длине контекста почти не поворачиваются и сравнивают содержание.",
          r"\begin{center}{\small\begin{tabular}{|l|r|r|r|r|r|}\toprule",
          r"\rowcolor{head}\textbf{Вид головы} & \textbf{Голов} & \textbf{Преим., логиты} & \textbf{Период $<$ 20 токенов} & \textbf{20–20 000} & \textbf{$>$ 20 000 токенов} \\ \midrule",
          *rows_f, r"\bottomrule\end{tabular}}\end{center}", r"\end{minipage}", r"\par"]
    rows_c = []
    for k_, tname in (("предыдущий токен", "токен перед текущим"), ("сам токен", "текущий токен"), ("тот же токен", "прошлые вхождения того же токена"),
                      ("индукция", "токен после прошлого вхождения текущего"), ("сток", "первый токен текста")):
        xs = K[k_]
        if not xs:
            continue
        kk, qq = agg_w(xs, TYP[k_], "key"), agg_w(xs, TYP[k_], "query")
        tops = collections.Counter(cadv(x, TYP[k_])["key"]["top"][0][0] for x in xs).most_common(2)
        rows_c.append(rf"{disp(k_) if k_ != 'сток' else 'Приписка «+ сток»'} & {tname} & {len(xs)} & {sgn(adv_m[k_])} & {num(kk[0])} / {num(kk[1])} / {num(kk[2])} & "
                      rf"{num(qq[0])} / {num(qq[1])} / {num(qq[2])} & {num(cos_s[k_])} / {num(cos_o[k_])} & " + ", ".join(f"{tex(n_)} ({c_} гол.)" for n_, c_ in tops) + r" \\")
    T += [r"\noindent\begin{minipage}{\linewidth}",
          r"\textbf{Откуда берётся выбор головы.} Логит линеен и по запросу, и по ключу, а оба точно раскладываются по источникам остаточного потока. "
          r"Поэтому преимущество цели над остальными токенами (в логитах) делится между источниками без приближений "
          rf"(ошибка суммы {circ['decomposition_err_logits']:.0e}".replace("e-0", r"$\cdot 10^{-").replace("e-", r"$\cdot 10^{-") + r"}$ логита).",
          r"\begin{center}{\small\begin{tabular}{|l|p{3.7cm}|r|r|c|c|c|p{3.5cm}|}\toprule",
          r"\rowcolor{head}\textbf{Вид головы} & \textbf{Цель} & \textbf{Голов} & \textbf{Преим., логиты} & \textbf{Ключ: эмб. / головы / MLP} & "
          r"\textbf{Запрос: эмб. / головы / MLP} & \textbf{Косинус: своя / чужая} & \textbf{Главный источник ключа} \\ \midrule",
          *rows_c, r"\bottomrule\end{tabular}}\end{center}",
          r"«Косинус»~— между запросом и ключом одной позиции и разных позиций. У стока доли выходят за пределы 0–1: MLP создают ключ первого токена, "
          r"а предыдущие головы частично его гасят.", r"\end{minipage}", r"\par",
          r"\begin{itemize}",
          rf"\item \textbf{{Индукция.}} {mech_ind}",
          rf"\item \textbf{{Сам токен.}} {mech_self} Источники у запроса и ключа одни и те же: в среднем {num(self_common, 1)} из 5 главных совпадают.",
          rf"\item \textbf{{Тот же токен.}} {mech_same}",
          rf"\item \textbf{{Сток.}} {mech_sink} Постоянный вектор MLP слоя 0, наоборот, пишут многие нейроны: на 90\,\% записи нужно {mw['const_layer0']['n_for_90']} из {mw['neurons_per_mlp']}.",
          r"\end{itemize}"]
    l1 = by[1] if NL > 1 else []
    T += [r"\begin{itemize}",
          r"\item \textbf{Предыдущий токен.} Быстрые частоты RoPE поворачиваются на заметный угол за один токен (самая быстрая~— на 1 радиан). "
          r"Фазы запроса и ключа сдвинуты так, что пик приходится на $\Delta{=}1$." + rf" Частоты с периодом меньше 20 токенов дают {num(fast['предыдущий токен'])} преимущества цели.",
          *([r"\item \textbf{Спад в следующих слоях включает MLP.} " +
             (rf"В слое 1 от {num(min(x['sources'][0]['q']['agg']['MLP'] for x in l1))} до {num(max(x['sources'][0]['q']['agg']['MLP'] for x in l1))} "
              r"главной частоты запроса приходит из MLP слоя 0: " if l1 else "") +
             r"MLP пишет в остаточный поток постоянный вектор, а проекции следующего слоя читают его на этой частоте."] if ramp_like else []),
          *([r"\item \textbf{Головы с общими ключами делают похожую работу.} В слое 1 одна группа собирает служебные слова, другая ищет тот же токен; "
             r"в слое 2 одна группа позиционная, другая смотрит далеко."] if name == "pocket0.3b" else []),
          r"\end{itemize}"]
    caveats = [r"\textbf{Границы отчёта.} Виды голов определены по порогам на одном корпусе (WikiText-2, " + f"{res['text_tokens']}" + r" токенов). "
               r"Разложение по источникам объясняет, откуда берутся запрос и ключ, то есть куда голова смотрит; что она после этого записывает "
               r"в остаточный поток, в отчёт не входит."]

    # ---------- 5. таблицы
    T += ["%%SEC:tables", r"\section*{3. Таблицы по слоям}"]
    for l in range(NL):
        w = nw[l]
        c = collections.Counter()
        for x in by[l]:
            for n_, v in x["sources"][0]["q"]["top"][:3]:
                c[n_] += v / H
        s = [av(l, lambda x, k=k: x["sources"][0]["q"]["agg"][k]) for k in ("эмб", "внимание", "MLP")]

        def part(o_):
            return "cos" if o_["top_part"][0] == "cos" else "sin"
        T += [r"\noindent\begin{minipage}{\linewidth}", rf"\subsection*{{Слой {l}}}",
              rf"{{\small\color{{ink2}} Нормализация: k\_norm макс. {num(abs(w['k_norm']['top_vals'][0]), 1)} (комп. {w['k_norm']['top_dims'][0]}, $m{{=}}{w['k_norm']['top_freq'][0]}$, {part(w['k_norm'])}), "
              rf"медиана {num(w['k_norm']['median_abs'], 1)}; q\_norm макс. {num(abs(w['q_norm']['top_vals'][0]), 1)} (комп. {w['q_norm']['top_dims'][0]}, $m{{=}}{w['q_norm']['top_freq'][0]}$, {part(w['q_norm'])}). "
              rf"Источники главной частоты запроса (среднее по головам): эмбеддинг {num(s[0])}, головы {num(s[1])}, MLP {num(s[2])}"
              + ("; крупнейшие: " + ", ".join(f"{tex(n_)} {num(v)}" for n_, v in c.most_common(3)) if l > 0 else "") + r".}",
              r"\begin{center}{\footnotesize\begin{tabular}{|r|l|r|r|r|r|r|r|l|l|r|l|r|r|r|l|}", r"\toprule",
              r"\rowcolor{head}\textbf{h} & \textbf{Вид головы} & $\boldsymbol{\Delta{=}0}$ & $\boldsymbol{\Delta{=}1}$ & \textbf{2–8} & \textbf{9–64} & \textbf{65+} & \textbf{поз.\,0} & \textbf{повтор} & \textbf{индукц.} & \textbf{ключей} & \textbf{частота (период)} & \textbf{спад} & $\boldsymbol{\sigma}$ \textbf{ключа} & $\boldsymbol{R^2}$ & \textbf{притягивает} \\ \midrule",]
        for x in by[l]:
            t, k, st = x["text"], x["kernel"], x["terms_std_logits"]
            m = t["mass_by_distance"]
            cat, lab = classify(x)
            f = k["top_freqs"][0]
            far = max(m["65+"] - t["pos0"], 0.0)
            lab_t = chip(cat) + (disp(lab) if cat == "дальний контекст" else r"\textbf{" + disp(lab) + "}")
            attr = tok_list(x["salience"]["top"], 4 if cat == "собирает определённые токены" else 3) if st["key_salience"] >= 1.0 else "—"
            T.append(f"{x['head']} & {lab_t} & {num(m['0'])} & {num(m['1'])} & {num(near8(x) - m['0'] - m['1'])} & {num(mid(x))} & {num(far)} & {num(t['pos0'])} "
                     f"& {num(t['dup_mass'])} ({num(t['dup_baseline'])}) & {num(t['ind_mass'])} ({num(t['ind_baseline'])}) & {t['effective_keys']:.0f} "
                     f"& $m{{=}}{f['m']}$ ({f['period']:.0f}) & {sgn(k['dominant_freq_change_over_512'])} & {num(st['key_salience'], 1)} & {num(st['token_r2_q'])} & {attr} \\\\")
        T += [r"\bottomrule\end{tabular}}\end{center}", r"\end{minipage}", r"\par\vspace{0.4em}"]
    parts, cur = {}, None
    for line in T:
        if line.startswith("%%SEC:"):
            cur = line[6:]
            parts[cur] = []
        else:
            parts[cur].append(line)
    model_t = TITLE.get(name, name)
    docs = {f"heads_{fname}_layers": PRE + title(f"Головы внимания {model_t}: виды голов и все {NL} слоёв")
            + parts["kinds"] + parts["cols"] + [r"\newpage"] + parts["tables"] + [r"\end{document}", ""],
            f"heads_{fname}_trends": PRE + title(f"Головы внимания {model_t}: главное, графики и механизм")
            + parts["summary"] + parts["figs"] + parts["mech"] + parts["method"] + caveats + [r"\end{document}", ""]}
    for old in (OUT / f"heads_{fname}.tex", OUT / f"heads_{fname}.pdf"):  # прежний единый файл
        old.unlink(missing_ok=True)
    rcode = 0
    for stem, lines in docs.items():
        texf = OUT / f"{stem}.tex"
        texf.write_text(gridify("\n".join(lines)))
        rc = subprocess.run(["tectonic", "-X", "compile", "--bundle", BUNDLE, texf.name], cwd=OUT, capture_output=True, text=True)
        if rc.returncode:
            print(rc.stdout[-600:], rc.stderr[-1500:])
        print("сохранено", texf.with_suffix(".pdf") if rc.returncode == 0 else f"ОШИБКА СБОРКИ {stem}")
        rcode |= rc.returncode
    return rcode



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pocket0.3b")
    sys.exit(build(ap.parse_args()))
