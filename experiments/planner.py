#!/usr/bin/env python
"""Пункт 4: планировщик замен по априорным статистикам, проверка на другой модели.
Подгонка на модели A (profile.json из analyze_profile): log KL_attn ≈ a + b·log(локальная ошибка окна),
log KL_mlp ≈ a' + b'·log(ошибка отбора отн. остаточного потока); суперaддитивность внимания из additivity:
ratio_k ≈ k^γ. На модели B по её статистикам предсказываются одиночные KL, жадно набирается план из m компонент
(внимание→окно / MLP→25% нейронов) с минимальным предсказанным итогом, затем план исполняется и меряется KL/PPL.
Базовые линии при том же m: первые m слоёв целиком (m/2 слоёв) и окно в первых m слоях.
Результат: results/<B>/criterion/planner.json."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detlayers.attention import SparseAttention  # noqa: E402
from detlayers.common import load_model, pick_device, save_json, seed_all, wikitext_windows  # noqa: E402
from detlayers.methods import Replacement  # noqa: E402
from detlayers.mlp import ExactPos0, TopKMLP  # noqa: E402

ROOT = Path(__file__).resolve().parent


def fit_loglog(x, y):
    lx, ly = torch.log(torch.tensor(x)).double(), torch.log(torch.tensor(y)).double()
    b = ((lx - lx.mean()) * (ly - ly.mean())).sum() / ((lx - lx.mean()) ** 2).sum()
    a = ly.mean() - b * lx.mean()
    return float(a), float(b)


def load_profile(m):
    return json.loads((ROOT / "results" / m / "criterion" / "profile.json").read_text())


@torch.no_grad()
def kl_eval(model, rep, wins, dev):
    nll_t = nll_s = kl = 0.0
    n = 0
    for s in range(wins.shape[0]):
        ids = wins[s:s + 1].to(dev)
        rep.disable()
        lt = model(ids).logits[:, :-1].float()
        rep.enable()
        ls = model(ids).logits[:, :-1].float()
        rep.disable()
        tgt = ids[:, 1:]
        lpt, lps = F.log_softmax(lt, -1), F.log_softmax(ls, -1)
        nll_t += -lpt.gather(-1, tgt[..., None]).sum().item()
        nll_s += -lps.gather(-1, tgt[..., None]).sum().item()
        kl += (lpt.exp() * (lpt - lps)).sum().item()
        n += tgt.numel()
    return {"kl": kl / n, "dppl_pct": 100 * (math.exp(nll_s / n) / math.exp(nll_t / n) - 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="0.6b")
    ap.add_argument("--test", default="1.7b")
    ap.add_argument("--budgets", default="3,6,9,12")
    ap.add_argument("--windows", type=int, default=32)
    args = ap.parse_args()
    A, B = load_profile(args.fit), load_profile(args.test)
    pa, pb = A["profile"], B["profile"]
    # подгонка на A
    a_att, b_att = fit_loglog(pa["e_attn_local"], pa["kl_attn"])
    a_mlp, b_mlp = fit_loglog(pa["inj_mlp"], pa["kl_mlp"])
    add = A["additivity"]["attn_window"]
    ks = [int(k) for k, v in add.items() if "kl" in v and int(k) > 1]
    gamma = fit_loglog(ks, [add[str(k)]["kl"] / add[str(k)]["sum_single_kl"] for k in ks])[1]
    pred_att = [math.exp(a_att + b_att * math.log(max(x, 1e-9))) for x in pb["e_attn_local"]]
    pred_mlp = [math.exp(a_mlp + b_mlp * math.log(max(x, 1e-9))) for x in pb["inj_mlp"]]
    L = len(pred_att)
    comps = sorted([("attn", l, pred_att[l]) for l in range(L)] + [("mlp", l, pred_mlp[l]) for l in range(L)], key=lambda t: t[2])
    seed_all(0)
    dev = pick_device(None)
    model, tok = load_model(args.test, dev, "fp32")
    layers = model.model.layers
    kap = int(0.25 * model.config.intermediate_size)
    wins = wikitext_windows(tok, "wikitext-2-raw-v1", "test", 1024, max_windows=args.windows)

    def make_rep(plan):
        rep = Replacement(model)
        for kind, l in plan:
            if kind == "attn":
                rep.add(l, "self_attn", SparseAttention(layers[l].self_attn, window=64, n_sink=1))
            else:
                rep.add(l, "mlp", ExactPos0(layers[l].mlp, TopKMLP(layers[l].mlp, kap)))
        return rep

    def predict(plan):
        na = sum(1 for k_, _ in plan if k_ == "attn")
        s_att = sum(pred_att[l] for k_, l in plan if k_ == "attn") * (na ** gamma if na else 1)
        return s_att + sum(pred_mlp[l] for k_, l in plan if k_ == "mlp")

    res = {"fit": args.fit, "test": args.test, "fit_params": {"attn": [a_att, b_att], "mlp": [a_mlp, b_mlp], "gamma": gamma},
           "spearman_pred_vs_actual_single": None, "budgets": {}}
    # качество предсказания одиночных KL на B (если B уже измерена analyze_profile)
    def spearman(x, y):
        x, y = torch.tensor(x).double(), torch.tensor(y).double()
        rx, ry = x.argsort().argsort().double(), y.argsort().argsort().double()
        rx, ry = rx - rx.mean(), ry - ry.mean()
        return float((rx * ry).sum() / (rx.norm() * ry.norm()))
    res["spearman_pred_vs_actual_single"] = {"attn": spearman(pred_att, pb["kl_attn"]), "mlp": spearman(pred_mlp, pb["kl_mlp"])}
    for m in (int(x) for x in args.budgets.split(",")):
        plan = [(k_, l) for k_, l, _ in comps[:m]]
        prefix_full = [("attn", l) for l in range(m // 2)] + [("mlp", l) for l in range(m // 2)]
        prefix_attn = [("attn", l) for l in range(m)]
        out = {}
        for name, pl in (("planner", plan), ("prefix_blocks", prefix_full), ("prefix_attn", prefix_attn)):
            rep = make_rep(pl)
            r = kl_eval(model, rep, wins, dev)
            r["predicted_kl"] = predict(pl)
            r["plan"] = pl
            out[name] = r
            print(f"m={m} {name:13s} KL {r['kl']:.4f} (предск. {r['predicted_kl']:.4f}) dPPL {r['dppl_pct']:.1f}%  план: {pl[:8]}{'…' if len(pl) > 8 else ''}", flush=True)
        res["budgets"][m] = out
        save_json(res, ROOT / "results" / args.test / "criterion" / "planner.json")
    print("saved")


if __name__ == "__main__":
    main()
