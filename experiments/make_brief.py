#!/usr/bin/env python
"""Числа и рисунок для двухстраничной записки (brief/brief.tex) по двум моделям (0.6B и 1.7B).

Читает results/<model>/howmany/*.json и пишет в ../brief/:
  brief_table.tex    — «сколько первых слоёв можно заменить»: k при росте PPL ≤2% и ≤5% для 0.6B / 1.7B,
                       экономия умножений модели (0.6B) при этих k для контекста T=2048 и 8192;
  brief_numbers.tex  — макросы с ключевыми числами для текста записки (суффикс B — 1.7B);
  fig_howmany.pdf    — рост PPL от k, две панели (0.6B, 1.7B).
Пороги: рост PPL WikiText-2 ≤ 2% («практически без потерь») и ≤ 5%.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from make_report import INK, INK2, PAL  # noqa: E402,F401  (стиль rcParams задаётся при импорте)
from sweep import CONFIGS, HOWMANY, HOWMANY_K  # noqa: E402

OUT = ROOT.parent / "brief"
MODELS = ["0.6b", "1.7b", "smol1.7b"]  # таблица (три столбца k) и макросы (0.6b/1.7b)
KMAX_EXPECTED = {"0.6b": 12, "1.7b": 8, "smol1.7b": 8, "llama1b": 8}  # последний k развёртки; меньше — кривая не досчитана
FIG_MODELS = ["0.6b", "1.7b", "smol1.7b", "llama1b"]  # панели рисунка (берутся те, у которых есть кривые)
TITLE = {"0.6b": "Qwen3-0.6B", "1.7b": "Qwen3-1.7B", "smol1.7b": "SmolLM2-1.7B", "llama1b": "Llama-3.2-1B"}
CFG = {"0.6b": dict(L=28, d=1024, H=16, Hkv=8, D=128, I=3072, V=151936),
       "1.7b": dict(L=28, d=2048, H=16, Hkv=8, D=128, I=6144, V=151936),
       "smol1.7b": dict(L=24, d=2048, H=32, Hkv=32, D=64, I=8192, V=49152),
       "llama1b": dict(L=16, d=2048, H=32, Hkv=8, D=64, I=8192, V=128256),
       "4b": dict(L=36, d=2560, H=32, Hkv=8, D=128, I=9728, V=151936),
       "8b": dict(L=36, d=4096, H=32, Hkv=8, D=128, I=12288, V=151936),
       "14b": dict(L=40, d=5120, H=40, Hkv=8, D=128, I=17408, V=151936),
       "32b": dict(L=64, d=5120, H=64, Hkv=8, D=128, I=25600, V=151936)}

# что показываем (порядок = строки таблицы и линии графика): имя -> (подпись, короткая подпись, наш?, макро-суффикс)
SHOW = {
    "L1-window64": ("Окно 64 + сток", "окно 64 + сток", False, "Win"),
    "L1-window64s": ("Окно 64 + массивные стоки", "окно 64 + массивные стоки", True, "WinS"),
    "O1-headtyped50": ("O1: половина голов → позиционное ядро", "O1: половина голов → ядро", True, "HeadT"),
    "ML2-topk25-dejavu-oracle": ("Отбор 25\\% нейронов по активациям (Deja Vu)", "отбор 25\\% по активациям", False, "Oracle"),
    "MO4-routed25r": ("M-O4: предиктор нейронов, 25\\%", "M-O4: предиктор, 25\\%", True, "Routed"),
    "ML5-svdllm50": ("SVD-LLM, ранг 50\\%", "SVD-LLM 50\\%", False, "Svd"),
    "B-window64+routed25r": ("Окно 64 + сток + M-O4", "окно 64 + сток + M-O4", True, "WinRouted"),
    "B-window64s+routed25r": ("Окно 64 + массивные стоки + M-O4", "окно 64 + массивные стоки + M-O4", True, "WinSRouted"),
    "B-window64+svdllm": ("Окно 64 + сток + SVD-LLM", "окно 64 + сток + SVD-LLM", False, "WinSvd"),
}
NAIVE = {"MO4-routed25": "RoutedNaive", "B-window64+routed25": "WinRoutedNaive"}  # только макросы для текста
# линии только для рисунка (не строки таблицы): окно с правилом «сток = массивный ключ»
SHOW_FIG = dict(SHOW)
KIND = {"L1-window64": "внимание", "L1-window64s": "внимание", "O1-headtyped50": "внимание", "ML2-topk25-dejavu-oracle": "MLP",
        "MO4-routed25r": "MLP", "ML5-svdllm50": "MLP", "B-window64+routed25r": "блок", "B-window64s+routed25r": "блок",
        "B-window64+svdllm": "блок"}


def macs_per_token(c, T, attn=("full", None), mlp_frac=1.0, mlp_extra=0):
    proj = c["d"] * c["H"] * c["D"] * 2 + 2 * c["d"] * c["Hkv"] * c["D"]
    keys = (T + 1) / 2 if attn[0] == "full" else min(attn[1] + 1, (T + 1) / 2)
    scores = 2 * c["H"] * c["D"] * keys
    return proj + scores + 3 * c["d"] * c["I"] * mlp_frac + mlp_extra


def layer_cost(name, c, T):
    method, params = HOWMANY[name]
    attn, mlpf, extra = ("full", None), 1.0, 0
    am = method.split("+")[0]
    pa = params.get("attn", params) if "+" in method else params
    pm = params.get("mlp", params) if "+" in method else params
    if am in ("sparse", "window_tail"):
        attn = ("window", pa.get("window", 64))
    head_frac = pa.get("frac", 0.0) if am == "head_typed" else 0.0
    mm = method.split("+")[-1]
    if mm == "routed_mlp":  # κ нейронов + линейная поправка d² + предиктор (d·64 + 64·I)
        mlpf, extra = pm.get("active_ratio", 0.25), c["d"] * c["d"] + 64 * (c["d"] + c["I"])
    elif mm in ("svdllm_mlp", "a3_mlp"):
        mlpf = pm.get("ratio", 0.5)
    elif mm == "topk_mlp":  # gate/up целиком, down — доля
        mlpf = (2 + pm.get("ratio", 0.25)) / 3
    cost = macs_per_token(c, T, attn, mlpf, extra)
    if head_frac:
        cost -= head_frac * c["H"] * c["D"] * (T + 1) / 2
    return cost


def _kint(k):
    """k из таблицы может быть строкой «$\\ge$k» (кривая не досчитана) — для расчёта экономии берём число."""
    return int(re.sub(r"\D", "", str(k))) if k else 0


def model_saving(name, k, c, T):
    k = _kint(k)
    full = c["L"] * macs_per_token(c, T) + c["d"] * c["V"]
    return 1 - (full - k * (macs_per_token(c, T) - layer_cost(name, c, T))) / full


def kv_saving(name, k, c, T):
    k = _kint(k)
    method, params = HOWMANY[name]
    if method.split("+")[0] not in ("sparse", "window_tail"):
        return 0.0
    w = (params.get("attn", params)).get("window", 64)
    return k / c["L"] * max(0.0, 1 - (w + 1) / T)


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def load_curves(model, names):
    curves = {}
    for name in names:
        pts = []
        for k in HOWMANY_K:
            r = load(ROOT / "results" / model / "howmany" / f"{name}_k{k}.json")
            if r:
                pts.append((k, r["lm"]["ppl"] / r["lm"]["ppl_teacher"] - 1, r["lm"]["kl"]))
        if pts:
            curves[name] = sorted(pts)
    return curves


def kmax(pts, th, expected=None):
    """Наибольший k, при котором порог выполнен для всех k' ≤ k. Если кривая не досчитана и порог ещё не
    нарушен, возвращается строка «≥k»."""
    best = 0
    broken = False
    for k, dp, *_ in pts:
        if dp <= th:
            best = k
        else:
            broken = True
            break
    if expected and not broken and pts and pts[-1][0] < expected and best:
        return f"$\\ge${best}"
    return best


def pct(x):
    return f"{100 * x:.1f}\\%"


def dash(x):
    return str(x) if x else "—"


def main():
    OUT.mkdir(exist_ok=True)
    base = load(ROOT / "results" / "0.6b" / "final" / "orig.json")
    cur = {m: load_curves(m, list(SHOW) + list(NAIVE)) for m in MODELS}
    c6 = CFG["0.6b"]
    mac = [f"\\newcommand{{\\BasePPL}}{{{base['lm']['ppl']:.2f}}}" if base else "\\newcommand{\\BasePPL}{—}",
           f"\\newcommand{{\\NumMethods}}{{{len([n for n in CONFIGS if n != 'orig'])}}}",
           f"\\newcommand{{\\KMaxPlan}}{{{HOWMANY_K[-1]}}}"]
    rows = []
    for name, (label, _, ours, suf) in SHOW.items():
        k = {m: (kmax(cur[m][name], 0.02, KMAX_EXPECTED.get(m)), kmax(cur[m][name], 0.05, KMAX_EXPECTED.get(m)))
             if name in cur[m] else (None, None) for m in MODELS}
        k2, k5 = k["0.6b"]
        s = {(kk, T): model_saving(name, kk, c6, T) for kk in (k2, k5) if kk for T in (2048, 8192)}
        rows.append((label, KIND[name], ours, k, s))
        for m, sm in (("0.6b", ""), ("1.7b", "B")):
            kk2, kk5 = k[m]
            pts = cur[m].get(name)
            mac += [f"\\newcommand{{\\kTwo{suf}{sm}}}{{{dash(kk2) if pts else '—'}}}",
                    f"\\newcommand{{\\kFive{suf}{sm}}}{{{dash(kk5) if pts else '—'}}}",
                    f"\\newcommand{{\\kDone{suf}{sm}}}{{{max(p[0] for p in pts) if pts else '—'}}}",
                    f"\\newcommand{{\\dpplLast{suf}{sm}}}{{{pct(pts[-1][1]) if pts else '—'}}}"]
            for th, kk in (("Two", kk2), ("Five", kk5)):
                for T, tn in ((2048, "Short"), (8192, "Long")):
                    mac.append(f"\\newcommand{{\\save{th}{suf}{sm}{tn}}}{{{pct(model_saving(name, kk, CFG[m], T)) if kk else '—'}}}")
                mac.append(f"\\newcommand{{\\kv{th}{suf}{sm}}}{{{pct(kv_saving(name, kk, CFG[m], 8192)) if kk else '—'}}}")
    # макросы наивной калибровки (для текста про ловушку): рост PPL при k=2,3,4
    for name, suf in NAIVE.items():
        for m, sm in (("0.6b", ""), ("1.7b", "B")):
            d = {k: dp for k, dp, _ in cur[m].get(name, [])}
            for kk, kn in ((2, "KTwo"), (3, "KThree"), (4, "KFour")):  # в имени макроса цифры недопустимы
                mac.append(f"\\newcommand{{\\dppl{suf}{sm}{kn}}}{{{pct(d[kk]) if kk in d else '—'}}}")
    for name, (label, _, ours, suf) in SHOW.items():
        for m, sm in (("0.6b", ""), ("1.7b", "B")):
            d = {k: dp for k, dp, _ in cur[m].get(name, [])}
            for kk, kn in ((2, "KTwo"), (3, "KThree"), (4, "KFour")):  # в имени макроса цифры недопустимы
                mac.append(f"\\newcommand{{\\dppl{suf}{sm}{kn}}}{{{pct(d[kk]) if kk in d else '—'}}}")
    # фраза про 1.7B — целиком из данных
    w, r, o = (cur["1.7b"].get(n) for n in ("L1-window64", "MO4-routed25r", "ML2-topk25-dejavu-oracle"))
    if w or r or o:
        parts = []
        if w:
            parts.append(f'окно 64 + сток — {dash(kmax(w, 0.02))} слоя при $\\le 2\\,\\%$ и {dash(kmax(w, 0.05))} при $\\le 5\\,\\%$')
        if r:
            parts.append(f'M-O4 — {dash(kmax(r, 0.02))} и {dash(kmax(r, 0.05))}')
        if o:
            parts.append(f'отбор по активациям — {dash(kmax(o, 0.02))} и {dash(kmax(o, 0.05))}')
        done = max(max(p[0] for p in x) for x in (w, r, o) if x)
        info_b = ("На Qwen3-1.7B (тоже 28 слоёв, вдвое шире; посчитано до $k=%d$): " % done) + "; ".join(parts) + "."
    else:
        info_b = "Для Qwen3-1.7B кривые считаются; таблица и график дополнятся автоматически."
    mac.append(f"\\newcommand{{\\InfoB}}{{{info_b}}}")
    (OUT / "brief_numbers.tex").write_text("\n".join(mac) + "\n", encoding="utf-8")
    # таблица
    lines = [r"\begin{tabular}{llcccc}", r"\toprule",
             r"Метод & Что заменено & \multicolumn{2}{c}{Сколько слоёв $k$: Qwen3-0.6B / 1.7B / SmolLM2} & "
             r"\multicolumn{2}{c}{Экономия умножений (0.6B)} \\",
             r" & & при $\Delta$PPL$\le$2\% & при $\Delta$PPL$\le$5\% & при $k_{2\%}$ & при $k_{5\%}$ \\",
             r" & & & & \scriptsize $T{=}2048$ / $8192$ & \scriptsize $T{=}2048$ / $8192$ \\", r"\midrule"]
    for label, kind, ours, k, s in rows:
        lab = (r"\textbf{" + label + "}") if ours else label
        k2, k5 = k["0.6b"]
        kb2, kb5 = k["1.7b"]
        ks2, ks5 = k["smol1.7b"]
        cell = lambda v: "…" if v is None else dash(v)  # None — конфигурация ещё не считалась; 0 — порог не выполнен
        c2 = " / ".join(cell(v) for v in (k2, kb2, ks2))
        c5 = " / ".join(cell(v) for v in (k5, kb5, ks5))
        e2 = f"{pct(s[(k2, 2048)])} / {pct(s[(k2, 8192)])}" if k2 else "—"
        e5 = f"{pct(s[(k5, 2048)])} / {pct(s[(k5, 8192)])}" if k5 else "—"
        lines.append(f"{lab} & {kind} & {c2} & {c5} & {e2} & {e5} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "brief_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # рисунок: панель на модель (все, у кого есть кривые); вариант «slide» — крупнее шрифты, для brief/slide.tex
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    curf = {m: load_curves(m, list(SHOW_FIG)) for m in FIG_MODELS}
    fig_models = [m for m in FIG_MODELS if curf[m] and max(len(v) for v in curf[m].values()) >= 2]  # панель только при >=2 точках по k
    YTOP = 130.0  # значения выше обрезаются и помечаются треугольником
    for fname, panel_w, fs in (("fig_howmany.pdf", 2.3, 1.0), ("fig_howmany_slide.pdf", 3.0, 1.35)):
      kmaxs = [max([p[0] for n in curf[m] for p in curf[m][n]] + [4]) for m in fig_models]
      fig, axes = plt.subplots(1, len(fig_models), figsize=(panel_w * len(fig_models) + 0.6, 2.6 * fs), sharey=True,
                               gridspec_kw={"width_ratios": [2 + km / 4 for km in kmaxs]})
      axes = list(axes) if len(fig_models) > 1 else [axes]
      for ax, m in zip(axes, fig_models):
          ks_all = [k for k in HOWMANY_K if k <= max([p[0] for n in curf[m] for p in curf[m][n]] + [4])]
          for i, n in enumerate(SHOW_FIG):
              if n not in curf[m]:
                  continue
              pts = curf[m][n]
              _, short, ours, _ = SHOW_FIG[n]
              ys = [100 * max(p[1], 1e-3) for p in pts]
              mk = {"блок": "^", "MLP": "s", "внимание": "o"}[KIND[n]]  # треугольники — блоки, квадраты — MLP, круги — внимание
              ax.plot([p[0] for p in pts], [min(y, YTOP) for y in ys], color=PAL[i % len(PAL)],
                      ls="-" if ours else "--", marker=mk, ms=3.4 * fs, label=short.replace("\\%", "%"))
              clipped = [(p[0], y) for p, y in zip(pts, ys) if y > YTOP]
              if clipped:
                  ax.scatter([c[0] for c in clipped], [YTOP] * len(clipped), marker="*", s=26 * fs, color=PAL[i % len(PAL)], zorder=4)
          for th in (2, 5):  # пунктир заканчивается до подписи
              ax.plot([0.6, ks_all[-1] + 0.25], [th, th], color=INK2, lw=1.0, ls=":", zorder=1)
              ax.text(ks_all[-1] + 0.45, th, f"{th}%", va="center", ha="left", fontsize=7 * fs, color=INK, fontweight="bold")
          ax.set_yscale("log")
          ax.set_ylim(0.08, YTOP * 1.3)
          ax.yaxis.set_major_locator(FixedLocator([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]))
          ax.yaxis.set_minor_locator(NullLocator())
          ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
          ax.set_xticks(ks_all)
          ax.set_xlim(0.6, ks_all[-1] + 1.0)
          ax.set_title(f"{TITLE[m]} ({CFG[m]['L']} сл.)", color=INK, fontsize=9 * fs)
          ax.tick_params(labelsize=7 * fs)
      axes[0].set_ylabel("рост PPL, %", fontsize=8 * fs)
      fig.supxlabel("k — число заменённых первых слоёв", fontsize=8 * fs, color=INK)
      h, l = axes[0].get_legend_handles_labels()
      fig.legend(h, l, fontsize=6.3 * fs, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.0), frameon=False,
                 handlelength=2.2, columnspacing=1.2)
      fig.tight_layout()
      fig.savefig(OUT / fname)
      plt.close(fig)
    for label, kind, _, k, s in rows:
        print(f"{label:44s} 0.6B k2%={dash(k['0.6b'][0])} k5%={dash(k['0.6b'][1])} | 1.7B k2%={dash(k['1.7b'][0])} k5%={dash(k['1.7b'][1])}")
    print(info_b)


if __name__ == "__main__":
    main()
