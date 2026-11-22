"""Какие нейроны MLP пишут два вектора, на которых держатся механизмы голов.

1) Вектор стока: выход MLP на позиции 0 в слое, после которого норма остаточного потока на позиции 0 резко
   растёт (слой берётся из circuits.json). Вклад нейрона n в выход вдоль его направления u:
   c_n = a_n(поз.0) · (W_down[:, n] · u). Считается, сколько нейронов дают этот вектор и во сколько раз их
   активация на позиции 0 больше обычной.
2) Постоянный вектор MLP слоя 0: средний по позициям выход, который следующие слои читают на главной частоте.
   Та же концентрация по нейронам.

  python mlp_writers.py --model pocket0.3b
Результат: results/<m>/heads/mlp_writers.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from detlayers.common import load_model, wikitext_windows  # noqa: E402


def conc(c):
    """Концентрация вкладов: доли 1, 3, 10, 50 крупнейших нейронов и число нейронов для 90 % суммы."""
    tot = c.sum()
    s, idx = torch.sort(c, descending=True)
    cum = torch.cumsum(s, 0) / tot
    return {"top1": float(cum[0]), "top3": float(cum[2]), "top10": float(cum[9]), "top50": float(cum[49]),
            "n_for_90": int((cum < 0.9).sum()) + 1, "neurons": idx[:5].tolist(), "shares": (s[:5] / tot).tolist()}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pocket0.3b")
    ap.add_argument("--windows", type=int, default=6)
    args = ap.parse_args()
    base = ROOT / "results" / args.model / "heads"
    circ = json.loads((base / "circuits.json").read_text())
    ratio = [p["rms_ratio"] for p in circ["pos0"]]
    jump = max(range(len(ratio) - 1), key=lambda l: ratio[l + 1] / max(ratio[l], 1e-9))  # MLP этого слоя создаёт сток
    model, tok = load_model(args.model, torch.device("cpu"), "fp32")
    wins = wikitext_windows(tok, "wikitext-2-raw-v1", "test", 512, max_windows=args.windows)
    cap = {}
    hooks = [model.model.layers[l].mlp.down_proj.register_forward_pre_hook(lambda m_, i, l=l: cap.__setitem__(l, i[0][0].detach().double()))
             for l in {0, jump}]
    I = model.model.layers[0].mlp.down_proj.weight.shape[1]
    acc = {"sink": torch.zeros(I, dtype=torch.float64), "sink_act0": torch.zeros(I, dtype=torch.float64),
           "sink_act_med": torch.zeros(I, dtype=torch.float64), "const": torch.zeros(I, dtype=torch.float64), "n": 0}
    norm0, norm_typ = 0.0, 0.0
    for w in wins:
        model(w[None])
        a = cap[jump]
        Wd = model.model.layers[jump].mlp.down_proj.weight.detach().double()  # (d, I)
        out0 = Wd @ a[0]
        u = out0 / out0.norm()
        acc["sink"] += a[0] * (u @ Wd)
        acc["sink_act0"] += a[0].abs()
        acc["sink_act_med"] += a[1:].abs().median(0).values
        norm0 += float(out0.norm())
        norm_typ += float((a[1:] @ Wd.T).norm(dim=-1).median())
        a = cap[0]
        acc["const"] += a[1:].mean(0)
        acc["n"] += 1
    for h_ in hooks:
        h_.remove()
    n = acc["n"]
    Wd0 = model.model.layers[0].mlp.down_proj.weight.detach().double()
    am = acc["const"] / n
    om = Wd0 @ am
    c_const = (am * ((om / om.norm()) @ Wd0)).clamp_min(0)
    c_sink = (acc["sink"] / n).clamp_min(0)
    s = conc(c_sink)
    s["act_pos0_over_typical"] = [float(acc["sink_act0"][i] / acc["sink_act_med"][i].clamp_min(1e-9)) for i in s["neurons"]]
    res = {"model": args.model, "neurons_per_mlp": int(I), "sink_layer": jump, "sink_write_norm_over_typical": norm0 / norm_typ,
           "rms_ratio_before": ratio[jump], "rms_ratio_after": ratio[jump + 1], "sink": s, "const_layer0": conc(c_const)}
    (base / "mlp_writers.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
