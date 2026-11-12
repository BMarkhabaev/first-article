#!/usr/bin/env python
"""Нейроны стока. На слое L = k_sink − 1 (создаёт массивную активацию позиции 0) находим нейроны с наибольшим
вкладом |z_i(pos0)|·||w_d,i|| (набор, покрывающий 90% вклада, не более 64) и смотрим их ранг в статистиках отбора
по всем позициям (как A3/SVD-LLM) и по позициям ≥1 (робастно). Абляции на 8 окнах: удалить нейроны стока
(а) везде, (б) только на позициях ≥1 (позиция 0 точно), (в) столько же случайных нейронов везде.
Результат: results/<model>/sink/neurons.json."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detlayers.common import calibration_windows, embed, load_model, pick_device, run_layers, save_json, seed_all, wikitext_windows  # noqa: E402
from detlayers.methods import Replacement  # noqa: E402
from detlayers.mlp import ExactPos0, SelectedMLP, _z, iterate_mlp_inputs  # noqa: E402
from sweep import k_sink  # noqa: E402

ROOT = Path(__file__).resolve().parent


@torch.no_grad()
def kl_eval(model, rep, wins, dev, L):
    nll_t = nll_s = kl = 0.0
    n = 0
    pos0 = []
    for s in range(wins.shape[0]):
        ids = wins[s:s + 1].to(dev)
        rep.disable()
        lt = model(ids).logits[:, :-1].float()
        rep.enable()
        ls = model(ids).logits[:, :-1].float()
        h, pos, pe = embed(model, ids)
        h = run_layers(model, model.model.layers, h, pos, pe, 0, L + 1)
        pos0.append(h[0, 0].float().norm().item())
        rep.disable()
        tgt = ids[:, 1:]
        lpt, lps = F.log_softmax(lt, -1), F.log_softmax(ls, -1)
        nll_t += -lpt.gather(-1, tgt[..., None]).sum().item()
        nll_s += -lps.gather(-1, tgt[..., None]).sum().item()
        kl += (lpt.exp() * (lpt - lps)).sum().item()
        n += tgt.numel()
    return {"kl": kl / n, "dppl_pct": 100 * (math.exp(nll_s / n) / math.exp(nll_t / n) - 1), "pos0_norm_after_L": sum(pos0) / len(pos0)}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="0.6b")
    ap.add_argument("--windows", type=int, default=8)
    args = ap.parse_args()
    seed_all(0)
    dev = pick_device(None)
    model, tok = load_model(args.model, dev, "fp32")
    L = k_sink(args.model) - 1
    mlp = model.model.layers[L].mlp
    I = model.config.intermediate_size
    wd = mlp.down_proj.weight.float().norm(dim=0)
    calib = calibration_windows(tok, 131_072, 512, seed=0)[:64]
    c0 = torch.zeros(I, device=dev)
    s_all = torch.zeros(I, device=dev)
    s_ge1 = torch.zeros(I, device=dev)
    nb = 0
    for i, x, _ in iterate_mlp_inputs(model, calib, L + 1, 8, dev):
        if i != L:
            continue
        z = _z(mlp, x).float()
        c0 += (z[:, 0].abs() * wd).sum(0)
        s_all += (z ** 2).sum((0, 1))
        s_ge1 += (z[:, 1:] ** 2).sum((0, 1))
        nb += x.shape[0]
    c0 /= nb
    order = c0.argsort(descending=True)
    cum = c0[order].cumsum(0) / c0.sum()
    m = int(min(64, (cum < 0.9).sum().item() + 1))
    sink = order[:m]
    rank_all = s_all.argsort(descending=True).argsort()[sink]
    rank_ge1 = s_ge1.argsort(descending=True).argsort()[sink]
    res = {"model": args.model, "layer": L, "n_sink_neurons": m, "sink_neurons": sink.tolist(),
           "share_of_pos0_contribution": float(cum[m - 1]),
           "rank_in_all_positions_stat": rank_all.tolist(), "rank_in_positions_ge1_stat": rank_ge1.tolist(),
           "in_top25pct_all": int((rank_all < 0.25 * I).sum()), "in_top25pct_ge1": int((rank_ge1 < 0.25 * I).sum()),
           "in_top50pct_all": int((rank_all < 0.5 * I).sum()), "in_top50pct_ge1": int((rank_ge1 < 0.5 * I).sum()),
           "ablations": {}}
    print(f"слой {L}: нейронов стока {m} (покрывают {100 * res['share_of_pos0_contribution']:.0f}% вклада позиции 0); "
          f"в топ-50% по всем позициям: {res['in_top50pct_all']}/{m}, по позициям ≥1: {res['in_top50pct_ge1']}/{m}", flush=True)
    wins = wikitext_windows(tok, "wikitext-2-raw-v1", "test", 1024, max_windows=args.windows)
    keep_wo_sink = torch.tensor(sorted(set(range(I)) - set(sink.tolist())), device=dev)
    g = torch.Generator().manual_seed(0)
    rand = torch.tensor(sorted(set(torch.randperm(I, generator=g)[: 2 * m].tolist()) - set(sink.tolist()))[:m], device=dev)
    keep_wo_rand = torch.tensor(sorted(set(range(I)) - set(rand.tolist())), device=dev)
    for name, mod in (("remove_sink_all_positions", SelectedMLP(mlp, keep_wo_sink)),
                      ("remove_sink_positions_ge1_only", ExactPos0(mlp, SelectedMLP(mlp, keep_wo_sink))),
                      ("remove_random_all_positions", SelectedMLP(mlp, keep_wo_rand))):
        rep = Replacement(model)
        rep.add(L, "mlp", mod)
        res["ablations"][name] = kl_eval(model, rep, wins, dev, L)
        print(name, json.dumps({a: round(b, 4) for a, b in res["ablations"][name].items()}), flush=True)
    save_json(res, ROOT / "results" / args.model / "sink" / "neurons.json")


if __name__ == "__main__":
    main()
