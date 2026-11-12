#!/usr/bin/env python
"""Таблицы (LaTeX) и рисунки (PDF) для статьи из results/*.json.

  python make_report.py                       # results/ -> ../figures, ../tables
  python make_report.py --results results_mac # другой каталог результатов

Отсутствующие результаты пропускаются (в таблице — «—»).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from sweep import CONFIGS, DISTILL, FINAL_MAIN, KSWEEP, PARETO, k_sink  # noqa: E402

# эталонная категориальная палитра (dataviz, светлый режим), фиксированный порядок
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.6, "lines.markersize": 4, "legend.frameon": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
    # шрифт как в статье: Times New Roman (системный), математика в стиле Times (STIX)
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

# подписи для статьи (русские)
LABEL = {
    "orig": "Исходная", "drop": "Удаление блоков",
    "L1-window64": "Окно+sink [17,19]", "L2-strided": "Strided [17]", "L3-performer-s0": "Performer [14]",
    "L4-linear-elu": "Линейное [11]", "L5-relpos": "Позиционное [21]", "L6-a3-attn": "A3 (внимание) [22]",
    "O1-headtyped50": "O1: типы голов", "O2-ridgeconv": "O2: ridge-свёртка", "O4-salience": "O4: ядро×значимость",
    "O4-salience-ex25": "O4 + 25% точных", "O4-salience-ex25-w64": "O4 + 25% точных (окно)",
    "ML1-a3mlp50": "A3 FFN [22]", "ML2-topk25-dejavu-oracle": "Deja Vu (оракул) [28]", "ML4-asvd50": "ASVD [26]",
    "ML5-svdllm50": "SVD-LLM [27]", "ML7-moe32x25": "MoEfication [29]",
    "MO1-ridge": "M-O1: линейный MLP", "MO1-ridge-keep25": "M-O1: линейный + 25%", "MO2-table": "M-O2: таблица",
    "MO3-cluster16": "M-O3: кластеры", "MO4-routed25": "M-O4: предиктор 25%", "MO4-routed10": "M-O4: предиктор 10%",
    "O5-windowtail16": "O5: окно 16 + хвост", "B-window+MO4-25": "Окно + M-O4", "B-O1+MO4-25": "O1 + M-O4",
    "L6-a3-full": "A3 (блок) [22]", "B-window+svdllm": "Окно + SVD-LLM", "B-window+ridge-keep25": "Окно + M-O1",
    "B-salience+ridge-keep25": "O4 + M-O1", "O3-ngram": "O3: n-граммы",
    "MO4-routed25r": "M-O4: предиктор 25% (робастн.)", "B-window+MO4-25r": "Окно + M-O4 (робастн.)",
    "MO1-ridge-keep25r": "M-O1: линейный + 25% (робастн.)", "ML1-a3mlp50r": "A3 FFN (робастн.)",
    "L6-a3-full-r": "A3 (блок, робастн.)",
}
# столбцы итоговой таблицы статьи (не совпадают с FINAL_MAIN прогона: добавлены робастные варианты)
TABLE_MAIN = ["orig", "L1-window64", "L1-window64s", "L5-relpos", "L6-a3-full", "ML5-svdllm50", "B-window+svdllm",
              "O1-headtyped50", "MO4-routed25", "MO4-routed25r", "B-window+MO4-25r"]
SHORT = {  # короткие заголовки столбцов таблицы
    "orig": "Исх.", "L1-window64": "Окно", "L1-window64s": "Окно$_m$", "L5-relpos": "Поз.", "L6-a3-full": "A3",
    "ML5-svdllm50": "SVD-LLM", "B-window+svdllm": "Окно+SVD", "O4-salience-ex25-w64": "O4",
    "MO1-ridge-keep25": "M-O1", "B-window+ridge-keep25": "Окно+M-O1", "B-salience+ridge-keep25": "O4+M-O1",
    "O1-headtyped50": "O1", "MO4-routed25": "M-O4", "B-window+MO4-25": "Окно+M-O4", "B-O1+MO4-25": "O1+M-O4",
    "MO4-routed25r": "M-O4r", "B-window+MO4-25r": "Окно+M-O4r", "MO1-ridge-keep25r": "M-O1r",
    "ML1-a3mlp50r": "A3 FFNr", "L6-a3-full-r": "A3r",
}
TASKS = [("lambada_openai", "acc", "LAMBADA"), ("hellaswag", "acc_norm", "HellaSwag"),
         ("arc_easy", "acc_norm", "ARC-e"), ("arc_challenge", "acc_norm", "ARC-c"),
         ("piqa", "acc_norm", "PIQA"), ("winogrande", "acc", "WinoGrande"), ("mmlu", "acc", "MMLU (5-shot)")]


def load(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def is_lit(name):
    return name.startswith(("L", "ML")) or name in ("B-window+svdllm", "B-window64+svdllm")


def kind_marker(name):
    """Маркер по типу замены: блоки — треугольники, MLP — квадраты, внимание — круги."""
    if name.startswith("B-"):
        return "^"
    if name.startswith(("ML", "MO")):
        return "s"
    return "o"


def style(name, i):
    return dict(color=PAL[i % len(PAL)], linestyle="--" if is_lit(name) else "-", marker=kind_marker(name))


# ============================================================ таблицы

def fmt(x, d=3):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def mlp_frac(name, r):
    """Доля умножений MLP заменённых слоёв относительно исходного (1.0 — MLP не тронут)."""
    method, params, group = CONFIGS[name]
    if method == "a3":
        return params.get("ratio_ffn", 0.5) if params.get("mlp", True) else 1.0
    mods = r.get("build", {}).get("modules", {})
    fr = [x["flop_frac"] for x in mods.values() if isinstance(x, dict) and "flop_frac" in x]
    if fr:
        return sum(fr) / len(fr)
    return 1.0 if group in ("base", "lit-attn", "our-attn") else None


def final_table(res_dir: Path, model: str, out: Path):
    k = k_sink(model)
    cols = [n for n in TABLE_MAIN]
    data = {}
    for n in cols:
        p = res_dir / model / "final" / ("orig.json" if n == "orig" else f"{n}_k{k}.json")
        data[n] = load(p)
    if not any(data.values()):
        return False
    rows = [("Что заменено", "kind", None), ("Доля вычислений MLP", "mlp_frac", 2),
            ("PPL WikiText-2 ↓", lambda r: r["lm"]["ppl"], 2),
            ("KL к исходной ↓", lambda r: r["lm"]["kl"], 3),
            ("Совпадение top-1 ↑", lambda r: 100 * r["lm"]["top1_agree"], 1)]
    for t, m, lab in TASKS:
        rows.append((f"{lab} ↑", (lambda t=t, m=m: lambda r: 100 * r["lm_eval"][t][m])(), 1))

    def avg(r):
        v = [100 * r["lm_eval"][t][m] for t, m, _ in TASKS if r["lm_eval"].get(t, {}).get(m) is not None]
        return sum(v) / len(v) if v else None
    rows.append(("Среднее по бенчам ↑", avg, 1))
    lines = [r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
             " & " + " & ".join(SHORT.get(c, c) for c in cols) + r" \\", r"\midrule"]
    kind_ru = {"base": "—", "lit-attn": "вним.", "our-attn": "вним.", "lit-mlp": "MLP", "our-mlp": "MLP",
               "block": "блок"}
    for lab, f, d in rows:
        vals = []
        for c in cols:
            r = data[c]
            if f == "kind":
                vals.append(kind_ru[CONFIGS[c][2]])
                continue
            if f == "mlp_frac":
                x = mlp_frac(c, r) if r else None
                vals.append(fmt(x, d) if x is not None else "—")
                continue
            try:
                vals.append(fmt(f(r), d) if r else "—")
            except (KeyError, TypeError):
                vals.append("—")
        lines.append(lab + " & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines).replace("↓", r"$\downarrow$").replace("↑", r"$\uparrow$") + "\n",
                   encoding="utf-8")
    return True


def full_table(res_dir: Path, model: str, out: Path):
    """Все методы для 0.6B (+ дистилляция): строки — методы, столбцы — метрики."""
    k = k_sink(model)
    names = list(CONFIGS) + [n + "+distill" for n in DISTILL]
    head = ["Метод", "Группа", "PPL", "KL", "top-1", "Ср. бенч", "Доля MLP"]
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule", " & ".join(head) + r" \\", r"\midrule"]
    grp_ru = {"base": "—", "lit-attn": "лит., внимание", "our-attn": "наш, внимание", "lit-mlp": "лит., MLP",
              "our-mlp": "наш, MLP", "block": "блок"}
    n_rows = 0
    for n in names:
        base = n.replace("+distill", "")
        p = res_dir / model / "final" / ("orig.json" if n == "orig" else f"{n}_k{k}.json")
        r = load(p)
        if not r:
            continue
        g = grp_ru[CONFIGS[base][2]] + (" + дист." if n.endswith("+distill") else "")
        ev = r.get("lm_eval", {})
        v = [100 * ev[t][m] for t, m, _ in TASKS if ev.get(t, {}).get(m) is not None]
        mods = r.get("build", {}).get("modules", {})
        pf = [x["param_frac"] for x in mods.values() if isinstance(x, dict) and "param_frac" in x]
        lines.append(" & ".join([LABEL.get(base, base), g, fmt(r["lm"]["ppl"], 2), fmt(r["lm"]["kl"], 3),
                                 fmt(100 * r["lm"]["top1_agree"], 1), fmt(sum(v) / len(v), 1) if v else "—",
                                 fmt(sum(pf) / len(pf), 2) if pf else "—"]) + r" \\")
        n_rows += 1
    lines += [r"\bottomrule", r"\end{tabular}"]
    if n_rows:
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return n_rows > 0


# ============================================================ рисунки

def fig_profile(res_dir: Path, out: Path):
    profs = {m: load(res_dir / m / "profile.json") for m in ["0.6b", "1.7b", "4b", "8b"]}
    profs = {m: p for m, p in profs.items() if p}
    if not profs:
        return False
    panels = [("sink", "Масса внимания на sink", lambda p: [sum(h) / len(h) for h in p["heads"]["sink"]]),
              ("tv", "TV до позиционного ядра", lambda p: [sum(h) / len(h) for h in p["heads"]["tv_relpos"]]),
              ("win", "Масса в окне 64", lambda p: [sum(h) / len(h) for h in p["heads"]["win_mass"]]),
              ("k", "Медиана ‖k‖·√s", lambda p: p["k_norm_scaled_median"]),
              ("r2", "R² линейной замены MLP", lambda p: p["mlp_linear_r2"]),
              ("h0", "‖h₀‖ / медиана ‖hₜ‖ (лог.)",
               lambda p: [a / max(b, 1e-6) for a, b in zip(p["resid_norm_pos0"], p["resid_norm_other"])])]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.2))
    for ax, (key, title, f) in zip(axes.flat, panels):
        for i, (m, p) in enumerate(profs.items()):
            y = f(p)
            ax.plot(range(len(y)), y, color=PAL[i], label=f"Qwen3-{m.upper()}")
            ks = k_sink(m) if (res_dir / m / "profile.json").exists() else None
            if ks and key == "sink":
                ax.axvline(ks - 0.5, color=PAL[i], lw=0.8, ls=":")
        ax.set_title(title, color=INK)
        ax.set_xlabel("слой")
        if key == "h0":
            ax.set_yscale("log")
    axes.flat[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return True


KSWEEP_PANELS = [
    ("Внимание: литература", ["L1-window64", "L2-strided", "L5-relpos", "L6-a3-attn", "L3-performer-s0", "L4-linear-elu"]),
    ("Внимание: наши", ["O1-headtyped50", "O2-ridgeconv", "O4-salience", "O4-salience-ex25", "O4-salience-ex25-w64"]),
    ("MLP", ["ML1-a3mlp50", "ML2-topk25-dejavu-oracle", "ML4-asvd50", "ML5-svdllm50", "MO1-ridge", "MO1-ridge-keep25",
             "MO2-table", "MO3-cluster16"]),
    ("Блок целиком", ["drop", "L6-a3-full", "B-window+svdllm", "B-window+ridge-keep25", "B-salience+ridge-keep25",
                      "O3-ngram"]),
]


def fig_ksweep(res_dir: Path, out: Path, model="0.6b"):
    any_ = False
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.6), sharey=True)
    for ax, (title, names) in zip(axes, KSWEEP_PANELS):
        for i, n in enumerate(names):
            xs, ys = [], []
            for k in KSWEEP:
                r = load(res_dir / model / "ksweep" / f"{n}_k{k}.json")
                if r:
                    xs.append(k)
                    ys.append(max(r["lm"]["kl"], 1e-5))
            if xs:
                any_ = True
                ax.plot(xs, ys, label=LABEL.get(n, n), **style(n, i))
        ax.set_yscale("log")
        ax.set_title(title, color=INK)
        ax.set_xticks(KSWEEP)
        ax.set_xlim(0.7, max(KSWEEP) + 0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="lower right", fontsize=5.5)
    axes[0].set_ylabel("KL к исходной, нат/токен")
    fig.supxlabel("k — число заменённых первых слоёв", fontsize=8, color=INK)
    fig.tight_layout()
    if any_:
        fig.savefig(out)
    plt.close(fig)
    return any_


HOWMANY_PANELS = [
    ("Внимание и блок целиком", ["L1-window64", "O1-headtyped50", "B-window64+svdllm", "B-window64+routed25",
                                 "B-window64+routed25r"]),
    ("MLP", ["ML2-topk25-dejavu-oracle", "ML5-svdllm50", "MO4-routed25", "MO4-routed25r"]),
]
HM_LABEL = {"L1-window64": "Окно 64 + сток [17,19]", "O1-headtyped50": "O1: половина голов → ядро",
            "B-window64+svdllm": "Окно + SVD-LLM", "B-window64+routed25": "Окно + M-O4 (наивная калибр.)",
            "B-window64+routed25r": "Окно + M-O4 (робастная)", "ML2-topk25-dejavu-oracle": "Отбор 25% по активациям [28]",
            "ML5-svdllm50": "SVD-LLM 50% [27]", "MO4-routed25": "M-O4 25% (наивная калибр.)",
            "MO4-routed25r": "M-O4 25% (робастная)"}


def _logy_pct(ax):
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def fig_howmany(res_dir: Path, out: Path, model="0.6b"):
    """Рост PPL от числа заменённых первых слоёв k (этап howmany), наивная и робастная калибровка."""
    from sweep import HOWMANY_K
    any_ = False
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    for ax, (title, names) in zip(axes, HOWMANY_PANELS):
        for i, n in enumerate(names):
            xs, ys = [], []
            for k in HOWMANY_K:
                r = load(res_dir / model / "howmany" / f"{n}_k{k}.json")
                if r:
                    xs.append(k)
                    ys.append(100 * max(r["lm"]["ppl"] / r["lm"]["ppl_teacher"] - 1, 1e-3))
            if xs:
                any_ = True
                ax.plot(xs, ys, label=HM_LABEL.get(n, n), **style(n, i))
        for th in (2, 5):
            ax.axhline(th, color=INK2, lw=0.7, ls=":")
        _logy_pct(ax)
        ax.set_title(title, color=INK)
        ax.set_xticks(HOWMANY_K)
        ax.legend(fontsize=5.8, loc="lower right")
    axes[0].set_ylabel("рост PPL WikiText-2, %")
    fig.supxlabel("k — число заменённых первых слоёв (из 28)", fontsize=8, color=INK)
    fig.tight_layout()
    if any_:
        fig.savefig(out)
    plt.close(fig)
    return any_


CLIFF_LABEL = {"routed2": "M-O4, k=2", "routed3": "M-O4, k=3", "routed4": "M-O4, k=4",
               "window3": "окно 64, k=3", "svdllm3": "SVD-LLM, k=3"}


def fig_cliff(res_dir: Path, out: Path, model="0.6b"):
    """Обрыв на k=3: ошибка остаточного потока по слоям и её доля в каналах массивной активации."""
    r = load(res_dir / model / "cliff" / "analysis.json")
    if not r:
        return False
    cfgs = r["configs"]
    order = [c for c in ["routed2", "routed3", "routed4", "window3", "svdllm3"] if c in cfgs]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [3, 2]})
    ax = axes[0]
    for i, c in enumerate(order):
        e = cfgs[c]["diag"]["rel_err_by_layer_input"][:13]
        ours = c.startswith("routed")
        ax.plot(range(len(e)), e, label=CLIFF_LABEL[c], color=PAL[i], marker="s" if ours else "o",
                ls="-" if ours else "--")
    ks = cfgs[order[0]]["k"] if order else 3
    ax.axvline(3, color=INK2, lw=0.7, ls=":")
    ax.text(3.15, 0.44, "вход слоя-стока", fontsize=6.5, color=INK2)
    ax.set_xlabel("i — вход слоя i")
    ax.set_ylabel("отн. ошибка остаточного потока")
    ax.set_xticks(range(0, 13, 2))
    ax.legend(fontsize=6)
    ax = axes[1]
    fr = [100 * cfgs[c]["diag"]["massive_channels"]["err_S_in"] for c in order]
    sig = [100 * cfgs[c]["diag"]["massive_channels"]["signal_T_in"] for c in order]
    ax.bar(range(len(order)), fr, color=[PAL[i] for i in range(len(order))], width=0.6, label="ошибка ученика")
    ax.scatter(range(len(order)), sig, marker="_", s=220, color=INK, zorder=3, label="сигнал учителя")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([CLIFF_LABEL[c].replace(", ", "\n") for c in order], fontsize=6)
    ax.set_ylabel("энергия в 8 массивных каналах, %")
    ax.legend(fontsize=6, loc="upper right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return True


def fig_map(res_dir: Path, out: Path, model="0.6b"):
    """Карта заменяемости по глубине и аддитивность ошибок."""
    r = load(res_dir / model / "criterion" / "profile.json")
    if not r:
        return False
    p = r["profile"]
    L = len(p["kl_attn"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [3, 2]})
    ax = axes[0]
    ax.plot(range(L), p["kl_attn"], color=PAL[0], marker="o", label="внимание → окно 64 + сток")
    ax.plot(range(L), p["kl_mlp"], color=PAL[1], marker="s", label="MLP → 25% нейронов по активациям")
    ax.plot(range(L), p["kl_noise"], color=INK2, ls=":", label=f"гауссов шум {100 * r['eps']:.0f}% на входе слоя")
    ax.set_yscale("log")
    ax.set_xlabel("ℓ — заменён только слой ℓ")
    ax.set_ylabel("KL к исходной, нат/токен")
    ax.set_xticks(range(0, L, 3))
    ax.set_ylim(4e-5, 3e-2)
    ax.legend(fontsize=6, loc="lower right")
    ax = axes[1]
    for i, (kind, lab) in enumerate([("attn_window", "окно"), ("mlp_topk", "25% нейронов")]):
        a = r["additivity"][kind]
        ks = sorted(int(k) for k, v in a.items() if "kl" in v)
        ax.plot(ks, [a[str(k)]["sum_single_kl"] for k in ks], color=PAL[i], ls="--", marker="o",
                label=f"{lab}: сумма одиночных")
        ax.plot(ks, [a[str(k)]["kl"] for k in ks], color=PAL[i], ls="-", marker="s", label=f"{lab}: измерено")
    ax.set_yscale("log")
    ax.set_xticks(ks)
    ax.set_xlabel("k — первые k слоёв")
    ax.set_ylabel("KL, нат/токен")
    ax.legend(fontsize=5.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return True


def table_criterion(res_dir: Path, out: Path, model="0.6b"):
    r = load(res_dir / model / "criterion" / "profile.json")
    if not r:
        return False
    sp = r["spearman"]
    rows = [("Внимание", "масса внимания вне окна", sp["attn"]["mass_out→kl"]),
            ("", "локальная ошибка окна (отн. выхода внимания)", sp["attn"]["e_attn_local→kl"]),
            ("", "ошибка относительно остаточного потока", sp["attn"]["inj_attn→kl"]),
            ("", "то же $\\times$ чувствительность слоя", sp["attn"]["inj_attn²·sens→kl"]),
            ("MLP", "$1-$ доля вклада 25\\% сильнейших нейронов", sp["mlp"]["1−top25_share→kl"]),
            ("", "локальная ошибка отбора (отн. выхода MLP)", sp["mlp"]["e_mlp_local→kl"]),
            ("", "ошибка относительно остаточного потока", sp["mlp"]["inj_mlp→kl"]),
            ("", "то же $\\times$ чувствительность слоя", sp["mlp"]["inj_mlp²·sens→kl"])]
    lines = [r"\begin{tabular}{llc}", r"\toprule",
             r"Блок & Статистика (без запуска замены) & Спирмен с KL \\", r"\midrule"]
    for a, b, c in rows:
        lines.append(f"{a} & {b} & {c:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def fig_exchange(res_dir: Path, out: Path, T=2048):
    """«Обменный курс»: рост PPL против доли сэкономленных умножений всей модели, все методы и обе модели."""
    from make_brief import CFG, HOWMANY as HM, macs_per_token, model_saving
    pts = []  # (model, family, saving%, dppl%)
    fams = {"L1-window64": "окно 64 + сток", "MO4-routed25r": "M-O4 (робастн.)", "ML2-topk25-dejavu-oracle": "отбор по активациям",
            "ML5-svdllm50": "SVD-LLM", "B-window64+routed25r": "окно + M-O4", "ML8-cats25": "CATS 25%"}
    for m in ("0.6b", "1.7b"):
        c = CFG[m]
        for name, fam in fams.items():
            if name not in HM:
                continue
            for p in (res_dir / m / "howmany").glob(f"{name}_k*.json"):
                r = load(p)
                if not r:
                    continue
                k = int(p.stem.split("_k")[1])
                pts.append((m, fam, 100 * model_saving(name, k, c, T), 100 * (r["lm"]["ppl"] / r["lm"]["ppl_teacher"] - 1)))
        # ширина окна (pareto, k=3)
        for p in (res_dir / m / "pareto").glob("sparse-window*_k3.json"):
            r = load(p)
            if r:
                w = int(p.stem.split("window")[1].split("_")[0])
                full = c["L"] * macs_per_token(c, T) + c["d"] * c["V"]
                sav = 3 * (macs_per_token(c, T) - macs_per_token(c, T, ("window", w))) / full
                pts.append((m, "окно w, k=3", 100 * sav, 100 * (r["lm"]["ppl"] / r["lm"]["ppl_teacher"] - 1)))
        # призраки блоков
        g = load(res_dir / m / "ghost" / "blocks.json")
        if g:
            full = c["L"] * macs_per_token(c, T) + c["d"] * c["V"]
            for key, b in g["blocks"].items():
                n = b["layers"][1] - b["layers"][0] + 1
                for mname, v in b["maps"].items():
                    extra = c["d"] ** 2 if mname == "full" else (2 * c["d"] * int(mname[4:]) if mname.startswith("rank") else 0)
                    sav = (n * macs_per_token(c, T) - extra) / full
                    pts.append((m, "призрак блока (карта)" if "map" in key else "призрак блока (ShortGPT)", 100 * sav, v["dppl_pct"]))
    if not pts:
        return False
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    colors = {"окно 64 + сток": PAL[0], "M-O4 (робастн.)": PAL[1], "отбор по активациям": PAL[2], "SVD-LLM": PAL[3],
              "окно + M-O4": PAL[4], "CATS 25%": PAL[5], "окно w, k=3": PAL[6], "призрак блока (карта)": PAL[7],
              "призрак блока (ShortGPT)": "#7f7f7f"}
    fam_list = [f for f in colors if any(ff == f for _, ff, _, _ in pts)]
    for fam in fam_list:
        for m, mk in (("0.6b", "o"), ("1.7b", "s")):
            xs = [x for mm, f, x, y in pts if mm == m and f == fam]
            ys = [min(max(y, 0.05), 300) for mm, f, x, y in pts if mm == m and f == fam]
            if xs:
                ax.scatter(xs, ys, color=colors[fam], marker=mk, s=22 if m == "0.6b" else 26,
                           facecolors=colors[fam] if m == "0.6b" else "none", linewidths=1.2,
                           label=f"{fam}" if m == "0.6b" else None)
    ax.set_ylim(0.04, 400)
    xs_line = [0.3, 30]
    ax.plot(xs_line, xs_line, color=INK2, lw=0.8, ls=":")
    ax.text(20, 24, "1 % PPL за 1 % вычислений", fontsize=6.5, color=INK2, rotation=28)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"экономия умножений всей модели, % (контекст {T})")
    ax.set_ylabel("рост PPL, %")
    ax.scatter([], [], marker="o", color=INK2, label="Qwen3-0.6B")
    ax.scatter([], [], marker="s", facecolors="none", edgecolors=INK2, label="Qwen3-1.7B")
    ax.legend(fontsize=6, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return True


def fig_toggle(res_dir: Path, out: Path):
    """Дозовая кривая ловушки: рост PPL наивного M-O4 (k=3) от числа отравленных окон в калибровке, обе модели."""
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    any_ = False
    for i, m in enumerate(("0.6b", "1.7b")):
        r = load(res_dir / m / "trap" / "toggle.json")
        if not r:
            continue
        any_ = True
        runs = r["runs"]
        xs = sorted(v["n_poisoned"] for k, v in runs.items() if not k.endswith("_robust"))
        ys = [next(v["dppl_pct"] for k, v in runs.items() if not k.endswith("_robust") and v["n_poisoned"] == x) for x in xs]
        ax.plot([x if x else 0.5 for x in xs], [max(y, 0.1) for y in ys], color=PAL[i], marker="s", label=f"Qwen3-{m.upper()}, наивная")
        rb = [v for k, v in runs.items() if k.endswith("_robust")]
        if rb:
            xr, yr = max(rb[0]["n_poisoned"], 0.5), max(rb[0]["dppl_pct"], 0.1)
            ax.scatter([xr], [yr], color=PAL[i], marker="*", s=90, zorder=5, label=f"Qwen3-{m.upper()}, робастная")
            y_naive = next((v["dppl_pct"] for k, v in runs.items() if not k.endswith("_robust")
                            and v["n_poisoned"] == rb[0]["n_poisoned"]), None)
            if y_naive and y_naive > 3 * yr:  # стрелка «фильтр строк»: наивная → робастная при той же дозе
                ax.annotate("", xy=(xr, yr * 1.6), xytext=(xr, y_naive / 1.6),
                            arrowprops=dict(arrowstyle="->", color=PAL[i], lw=1.0))
                ax.text(xr / 1.15, (yr * y_naive) ** 0.5, "фильтр\nстрок", fontsize=6, color=PAL[i], ha="right", va="center")
    if not any_:
        plt.close(fig)
        return False
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([0.5, 1, 4, 16])
    ax.set_xticklabels(["0", "1", "4", "16"])
    ax.set_xlabel("отравленных окон в калибровке (из 256)")
    ax.set_ylabel("рост PPL, %  (M-O4, k=3)")
    ax.legend(fontsize=6, loc="upper left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return True


def fig_planner(res_dir: Path, out: Path):
    """Планировщик по априорным статистикам против замены префикса: KL от бюджета m (подгонка на 0.6B)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    any_ = False
    for ax, m in zip(axes, ("0.6b", "1.7b")):
        r = load(res_dir / m / "criterion" / "planner.json")
        if not r:
            continue
        any_ = True
        ms = sorted(int(x) for x in r["budgets"])
        for i, (key, lab) in enumerate((("planner", "план по критерию"), ("prefix_blocks", "первые m/2 слоёв целиком"),
                                        ("prefix_attn", "окно в первых m слоях"))):
            ax.plot(ms, [r["budgets"][str(b)][key]["kl"] for b in ms], color=PAL[i], marker="s" if i == 0 else "o",
                    ls="-" if i == 0 else "--", label=lab)
        ax.plot(ms, [r["budgets"][str(b)]["planner"]["predicted_kl"] for b in ms], color=PAL[0], ls=":", lw=1, label="план: предсказано")
        ax.set_yscale("log")
        ax.set_xticks(ms)
        ax.set_title(f"Qwen3-{m.upper()}" + (" (подгонка)" if m == r["fit"] else f" (проверка, подгонка на {r['fit'].upper()})"), color=INK)
        ax.legend(fontsize=5.8, loc="lower right")
    axes[0].set_ylabel("KL к исходной, нат/токен")
    fig.supxlabel("m — число заменённых компонент (внимание или MLP одного слоя)", fontsize=8, color=INK)
    fig.tight_layout()
    if any_:
        fig.savefig(out)
    plt.close(fig)
    return any_


def fig_cats_depth(res_dir: Path, out: Path):
    """CATS (порог по gate) против отбора по вкладу: отношение KL одиночной замены по глубине."""
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    any_ = False
    for i, m in enumerate(("0.6b", "1.7b")):
        r = load(res_dir / m / "criterion" / "cats_depth.json")
        if not r:
            continue
        any_ = True
        ratio = [a / max(b, 1e-9) for a, b in zip(r["kl_cats"], r["kl_topk"])]
        ax.plot(range(len(ratio)), ratio, color=PAL[i], marker="o", ms=3, label=f"Qwen3-{m.upper()}")
    if not any_:
        plt.close(fig)
        return False
    ax.axhline(1, color=INK2, lw=0.8, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel("ℓ — заменён MLP только слоя ℓ (25 % нейронов)")
    ax.set_ylabel("KL(CATS) / KL(отбор по вкладу)")
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return True


def fig_domain(res_dir: Path, out: Path, model="0.6b"):
    """Домен калибровки: рост PPL на WikiText-2 и pile-10k при калибровке на WikiText-103, pile-10k и самогенерации."""
    cfgs = [("MO4-routed25r", "M-O4 (робастн.)"), ("B-window+MO4-25r", "окно + M-O4"), ("ML5-svdllm50", "SVD-LLM")]
    srcs = [("corpus2", "калибровка WikiText-103"), ("corpus2_calib_pile10k", "калибровка pile-10k"),
            (f"corpus2_calib_selfgen_{model}", "самогенерированная")]
    data = {}
    for sub, lab in srcs:
        for name, nl in cfgs:
            r = load(res_dir / model / sub / f"{name}_k3.json")
            if not r:
                continue
            pile = 100 * (r["lm"]["ppl"] / r["lm"]["ppl_teacher"] - 1)
            if r.get("lm_wikitext"):
                wt = 100 * (r["lm_wikitext"]["ppl"] / r["lm_wikitext"]["ppl_teacher"] - 1)
            else:
                f = load(res_dir / model / "final" / f"{name}_k3.json")
                wt = 100 * (f["lm"]["ppl"] / f["lm"]["ppl_teacher"] - 1) if f else None
            data[(lab, nl)] = (wt, pile)
    if not data:
        return False
    fig, axes = plt.subplots(1, len(cfgs), figsize=(7.0, 2.4), sharey=True)
    for ax, (name, nl) in zip(axes, cfgs):
        labs = [lab for _, lab in srcs if (lab, nl) in data]
        x = range(len(labs))
        wt = [data[(lab, nl)][0] or 0 for lab in labs]
        pl = [data[(lab, nl)][1] for lab in labs]
        ax.bar([i - 0.18 for i in x], wt, width=0.36, color=PAL[0], label="оценка: WikiText-2")
        ax.bar([i + 0.18 for i in x], pl, width=0.36, color=PAL[1], label="оценка: pile-10k")
        ax.set_xticks(list(x))
        ax.set_xticklabels([l.replace("калибровка ", "калибр.\n").replace("самогенерированная", "само-\nгенерация") for l in labs], fontsize=6)
        ax.set_title(nl, color=INK)
    axes[0].set_ylabel("рост PPL, % (k=3)")
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return True


def table_sink(res_dir: Path, out: Path):
    rows = []
    for m in ("0.6b", "1.7b"):
        r = load(res_dir / m / "sink" / "neurons.json")
        if not r:
            continue
        a = r["ablations"]
        rows.append((f"Qwen3-{m.upper()}", r["layer"], r["n_sink_neurons"], 100 * r["share_of_pos0_contribution"],
                     a["remove_sink_all_positions"]["dppl_pct"], a["remove_sink_all_positions"]["pos0_norm_after_L"],
                     a["remove_sink_positions_ge1_only"]["dppl_pct"], a["remove_random_all_positions"]["dppl_pct"]))
    if not rows:
        return False
    nl = "\\\\"
    lines = [r"\begin{tabular}{lcccrrrr}", r"\toprule",
             r"Модель & слой & нейронов & доля вклада в поз.~0 & \multicolumn{2}{c}{удалить везде} & удалить при $t\ge1$ & случайные " + nl,
             r" & & & & $\Delta$PPL & $\|h_0\|$ после слоя & $\Delta$PPL & $\Delta$PPL " + nl, r"\midrule"]
    for m, L, n, sh, d1, h0, d2, d3 in rows:
        lines.append(f"{m} & {L} & {n} & {sh:.0f}\\% & {d1:+.1f}\\% & {h0:.0f} & {d2:+.1f}\\% & {d3:+.1f}\\% " + nl)
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def table_robust(res_dir: Path, out: Path, model="0.6b"):
    """Наивная против робастной калибровки при k_sink."""
    pairs = [("SVD-LLM 50\\%", "ML5-svdllm50", "ML5-svdllm50r"), ("ASVD 50\\%", "ML4-asvd50", "ML4-asvd50r"),
             ("A3 FFN 50\\%", "ML1-a3mlp50", "ML1-a3mlp50r"), ("A3 (блок)", "L6-a3-full", "L6-a3-full-r"),
             ("M-O2: таблица по токену", "MO2-table", "MO2-table-r"), ("M-O1: линейный + 25\\%", "MO1-ridge-keep25", "MO1-ridge-keep25r"),
             ("M-O4: предиктор 25\\%", "MO4-routed25", "MO4-routed25r"), ("Окно + M-O4", "B-window+MO4-25", "B-window+MO4-25r")]
    k = k_sink(model)
    nl = "\\\\"
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Метод & \multicolumn{2}{c}{наивная калибровка} & \multicolumn{2}{c}{робастная} " + nl,
             r" & $\Delta$PPL & KL & $\Delta$PPL & KL " + nl, r"\midrule"]
    any_ = False
    for lab, a, b in pairs:
        ra, rb = load(res_dir / model / "final" / f"{a}_k{k}.json"), load(res_dir / model / "final" / f"{b}_k{k}.json")
        if not (ra and rb):
            continue
        any_ = True
        f = lambda r: (100 * (r["lm"]["ppl"] / r["lm"]["ppl_teacher"] - 1), r["lm"]["kl"])  # noqa: E731
        (da, ka), (db, kb) = f(ra), f(rb)
        lines.append(f"{lab} & {da:.1f}\\% & {ka:.3f} & {db:.1f}\\% & {kb:.3f} " + nl)
    lines += [r"\bottomrule", r"\end{tabular}"]
    if any_:
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return any_


def table_long(res_dir: Path, out: Path, model="0.6b"):
    """Длина контекста и число стоков (окно 64)."""
    L = load(res_dir / model / "long" / "length_k3.json")
    S3 = load(res_dir / model / "long" / "sinks_k3.json")
    S6 = load(res_dir / model / "long" / "sinks_k6.json")
    if not L:
        return False
    nl = "\\\\"
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Контекст $T$ & 1024 & 2048 & 4096 & 8192 " + nl, r"\midrule"]
    for key, lab in (("band_sink1", "$k=3$, 1 сток"), ("band_sink4", "$k=3$, 4 стока")):
        vals = [L["results"].get(str(T), {}).get(key, {}).get("dppl_pct") for T in (1024, 2048, 4096, 8192)]
        lines.append(lab + " & " + " & ".join(f"{v:.2f}\\%" if v is not None else "—" for v in vals) + " " + nl)
    if S3 and S6:
        lines.append(r"\midrule")
        for S, kk in ((S3, 3), (S6, 6)):
            for key, lab in (("band_sink2", f"$k={kk}$, 2 стока"), ("dense_sink1+newlines", f"$k={kk}$, сток + переводы строки")):
                vals = [S["results"].get(str(T), {}).get(key, {}).get("dppl_pct") for T in (1024, 2048)]
                lines.append(lab + " & " + " & ".join(f"{v:.2f}\\%" if v is not None else "—" for v in vals) + " & — & — " + nl)
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def table_timing(res_dir: Path, out: Path):
    rows = []
    for m in ("0.6b", "1.7b"):
        r = load(res_dir / m / "timing.json")
        if not r:
            continue
        for T, v in r["model_prefill"].items():
            if "speedup" in v:
                lay = r["layer"].get(T, {})
                rows.append((f"Qwen3-{m.upper()}", int(T), lay.get("full_ms"), lay.get("band_ms"), v["full_ms"], v["band_ms"], v["speedup"]))
    if not rows:
        return False
    nl = "\\\\"
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Модель & $T$ & \multicolumn{2}{c}{слой внимания, мс} & \multicolumn{2}{c}{prefill модели, мс} & отношение " + nl,
             r" & & полное (SDPA) & лента (PyTorch) & исходная & первые 3 слоя --- лента & " + nl, r"\midrule"]
    for m, T, lf, lb, mf, mb, sp in rows:
        lines.append(f"{m} & {T} & {lf:.1f} & {lb:.1f} & {mf:.0f} & {mb:.0f} & {sp:.2f} " + nl)
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def fig_pareto(res_dir: Path, out: Path, model="0.6b"):
    k = k_sink(model)
    fams_mlp = ["a3_mlp", "asvd_mlp", "svdllm_mlp", "topk_mlp", "moe_mlp", "ridge_mlp_r", "routed_mlp_r"]  # наши — робастная калибровка
    fams_att = ["sparse", "salience", "head_typed"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    any_ = False
    for ax, fams, xlab, title in (
            (axes[0], fams_mlp, "доля вычислений MLP (на токен)", f"MLP, k={k}"),
            (axes[1], fams_att, "доля голов/позиций с точным q·k", f"Внимание, k={k}")):
        for i, fam in enumerate(fams):
            method, key, values, *_ = PARETO[fam]
            pts = []
            for v in values:
                r = load(res_dir / model / "pareto" / f"{fam}-{key}{v}_k{k}.json")
                if not r:
                    continue
                mods = r["build"].get("modules", {})
                if fams is fams_mlp:
                    fr = [x.get("flop_frac") for x in mods.values() if isinstance(x, dict) and "flop_frac" in x]
                    x = sum(fr) / len(fr) if fr else None
                else:
                    x = {"sparse": min(1.0, v / 2048), "salience": v, "head_typed": 1 - v}[fam]
                if x is not None:
                    pts.append((x, max(r["lm"]["kl"], 1e-5)))
            if pts:
                any_ = True
                pts.sort()
                lit = fam not in ("ridge_mlp", "routed_mlp", "salience", "head_typed")
                ax.plot(*zip(*pts), color=PAL[i], ls="--" if lit else "-", marker="o" if lit else "s",
                        label={"a3_mlp": "A3 FFN", "asvd_mlp": "ASVD", "svdllm_mlp": "SVD-LLM",
                               "topk_mlp": "Deja Vu (оракул)", "moe_mlp": "MoEfication", "ridge_mlp": "M-O1 (наш)",
                               "ridge_mlp_r": "M-O1 (наш, робастн.)", "routed_mlp_r": "M-O4 (наш, робастн.)",
                               "routed_mlp": "M-O4 (наш)",
                               "sparse": "Окно+sink", "salience": "O4 (наш)", "head_typed": "O1 (наш)"}[fam])
        ax.set_yscale("log")
        ax.set_xlabel(xlab)
        ax.set_title(title, color=INK)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=6)
    axes[0].set_ylabel("KL, нат/токен")
    fig.tight_layout()
    if any_:
        fig.savefig(out)
    plt.close(fig)
    return any_


def fig_favor(res_dir: Path, out: Path):
    """Относительная ошибка оценки exp(xᵀy) случайными признаками FAVOR+ vs норма входа,
    и диапазон норм q, k в первых слоях (до стока) Qwen3-0.6B."""
    from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
    from detlayers.attention import orthogonal_gaussian
    g = torch.Generator().manual_seed(0)
    d = 128
    norms = [0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 16, 32, 64]
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    table = {}
    for i, m in enumerate([256, 4096]):
        w = orthogonal_gaussian(m, d, g).double()
        med = []
        for nv in norms:
            errs = []
            for _ in range(64):
                x = torch.randn(d, generator=g, dtype=torch.float64)
                y = torch.randn(d, generator=g, dtype=torch.float64)
                x, y = x / x.norm() * nv, y / y.norm() * nv
                phi = lambda z: torch.exp(w @ z - (z @ z) / 2) / math.sqrt(m)  # noqa: E731
                est = (phi(x) @ phi(y)).item()
                if not (math.isfinite(est) and est > 0):
                    errs.append(1.0)  # оценка выродилась (0 или переполнение)
                    continue
                ratio = math.exp(max(min(math.log(est) - float(x @ y), 50.0), -50.0))
                errs.append(min(abs(ratio - 1), 1.0))  # ошибка ограничена сверху 1
            med.append(sorted(errs)[len(errs) // 2])
        table[m] = dict(zip(norms, med))
        ax.plot(norms, med, color=PAL[i], marker="o", label=f"m = {m} признаков")
    prof = load(res_dir / "0.6b" / "profile.json") or load(ROOT / "results_mac" / "profile_0.6b_16k.json")
    if prof:
        ks = k_sink("0.6b")
        lo = min(prof["q_norm_scaled_median"][:ks])
        hi = max(prof["k_norm_scaled_median"][:ks])
        ax.axvspan(lo, hi, color=PAL[3], alpha=0.15, lw=0)
        ax.text(math.sqrt(lo * hi), 0.25, f"q, k в слоях 0–{ks - 1}\nQwen3-0.6B", color=INK2, fontsize=6.5,
                ha="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(FixedLocator([0.25, 1, 4, 16, 64]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylim(2e-3, 1.5)
    ax.set_xlabel("‖x‖·√s = ‖y‖·√s")
    ax.set_ylabel("медианная отн. ошибка ядра")
    ax.set_title("FAVOR+: ошибка растёт с нормой", color=INK)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    for m, row in table.items():
        print(f"favor m={m}: " + ", ".join(f"{k:g}→{v:.3f}" for k, v in row.items()))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--fig-dir", default=str(ROOT.parent / "figures"))
    ap.add_argument("--tab-dir", default=str(ROOT.parent / "tables"))
    args = ap.parse_args()
    res, fd, td = Path(args.results), Path(args.fig_dir), Path(args.tab_dir)
    fd.mkdir(parents=True, exist_ok=True)
    td.mkdir(parents=True, exist_ok=True)
    done = {
        "fig_profile": fig_profile(res, fd / "profile.pdf"),
        "fig_ksweep": fig_ksweep(res, fd / "ksweep.pdf"),
        "fig_pareto": fig_pareto(res, fd / "pareto.pdf"),
        "fig_favor": fig_favor(res, fd / "favor.pdf"),
        "fig_howmany": fig_howmany(res, fd / "howmany.pdf"),
        "fig_cliff": fig_cliff(res, fd / "cliff.pdf"),
        "fig_map": fig_map(res, fd / "map.pdf"),
        "tab_criterion": table_criterion(res, td / "criterion.tex"),
        "fig_exchange": fig_exchange(res, fd / "exchange.pdf"),
        "fig_toggle": fig_toggle(res, fd / "toggle.pdf"),
        "fig_planner": fig_planner(res, fd / "planner.pdf"),
        "fig_cats_depth": fig_cats_depth(res, fd / "cats_depth.pdf"),
        "fig_domain": fig_domain(res, fd / "domain.pdf"),
        "tab_sink": table_sink(res, td / "sink.tex"),
        "tab_robust": table_robust(res, td / "robust.tex"),
        "tab_long": table_long(res, td / "long.tex"),
        "tab_timing": table_timing(res, td / "timing.tex"),
        "tab_full_0.6b": full_table(res, "0.6b", td / "full_0.6b.tex"),
    }
    for m in ["0.6b", "1.7b", "4b", "8b", "32b", "smol1.7b", "llama1b"]:
        done[f"tab_final_{m}"] = final_table(res, m, td / f"final_{m}.tex")
    for k_, v in done.items():
        print(f"{'OK ' if v else '-- '} {k_}")


if __name__ == "__main__":
    main()
