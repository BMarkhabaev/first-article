#!/usr/bin/env python
"""Пункт 5: почему MLP ранних слоёв незаменим отбором нейронов, а поздних — заменим.
На состояниях учителя по всем слоям ℓ считаем (позиции ≥1):
  share      — ||MLP_ℓ(x)|| / ||h_ℓ|| : доля выхода MLP относительно остаточного потока (насколько слой «пишет»);
  conc25     — доля вклада топ-25% нейронов (концентрация);
  eff_rank   — эффективный ранг ковариации выхода MLP (participation ratio);
  active     — среднее число нейронов с |act(gate)·up| выше 1% от максимума по токену (плотность активаций);
  read_sink  — доля энергии выхода MLP_ℓ, попадающая в подпространство, которое читают ключи/запросы слоя-стока
               (сингулярные направления W_k и W_q слоя k_sink после RMSNorm), против случайного подпространства той же размерности;
  cont_vs_start — ошибка отбора 25% на продолжениях слов против начал слов (сборка слов).
Сопоставляем с kl_mlp из criterion/profile.json (ранговая корреляция). Результат: results/<model>/criterion/mlp_mechanism.json."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detlayers.common import embed, load_model, pick_device, save_json, seed_all, wikitext_windows  # noqa: E402
from detlayers.mlp import TopKMLP, _z  # noqa: E402
from sweep import k_sink  # noqa: E402

ROOT = Path(__file__).resolve().parent


def spearman(x, y):
    x, y = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    rx, ry = x.argsort().argsort().double(), y.argsort().argsort().double()
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx * ry).sum() / (rx.norm() * ry.norm()).clamp_min(1e-12))


def tok_is_continuation(tok, t):
    s = tok.convert_ids_to_tokens(int(t)).replace("Ġ", " ")
    return (not s.startswith(" ")) and s.strip().isalpha()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="0.6b")
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--ratio", type=float, default=0.25)
    args = ap.parse_args()
    seed_all(0)
    dev = pick_device(None)
    model, tok = load_model(args.model, dev, "fp32")
    layers = model.model.layers
    L = len(layers)
    I = model.config.intermediate_size
    d = model.config.hidden_size
    kap = int(args.ratio * I)
    ks = k_sink(args.model)
    wins = wikitext_windows(tok, "wikitext-2-raw-v1", "test", 1024, max_windows=args.windows)
    # подпространство чтения слоя-стока: старшие правые сингулярные векторы [W_q; W_k] (после RMSNorm вход нормирован)
    Ls = layers[ks]
    Wqk = torch.cat([Ls.self_attn.q_proj.weight, Ls.self_attn.k_proj.weight], 0).float()
    U, S, Vh = torch.linalg.svd(Wqk, full_matrices=False)
    r_read = int((S.cumsum(0) / S.sum() < 0.9).sum().item()) + 1  # 90% энергии
    P_read = Vh[:r_read].T  # (d, r)
    g = torch.Generator().manual_seed(0)
    P_rand = torch.linalg.qr(torch.randn(d, r_read, generator=g))[0].to(dev)
    acc = {k: torch.zeros(L, dtype=torch.float64) for k in ("share", "conc25", "active", "read_sink", "read_rand", "err_cont", "err_start")}
    cnt_cont = torch.zeros(L); cnt_start = torch.zeros(L)
    cov = [torch.zeros(d, d, dtype=torch.float64) for _ in range(L)]
    mean = [torch.zeros(d, dtype=torch.float64) for _ in range(L)]
    n_tok = 0
    for s_ in range(wins.shape[0]):
        ids = wins[s_:s_ + 1].to(dev)
        h, pos, pe = embed(model, ids)
        cont = torch.tensor([tok_is_continuation(tok, t) for t in ids[0, 1:].tolist()], device=dev)
        for l in range(L):
            Lr = layers[l]
            a, _ = Lr.self_attn(Lr.input_layernorm(h), position_embeddings=pe, attention_mask=None)
            mid = h + a
            xm = Lr.post_attention_layernorm(mid)
            m = Lr.mlp(xm)
            m1, mid1 = m[0, 1:].float(), mid[0, 1:].float()
            acc["share"][l] += (m1.norm(dim=-1) / mid1.norm(dim=-1).clamp_min(1e-6)).mean().item()
            wd = Lr.mlp.down_proj.weight.norm(dim=0)
            c = _z(Lr.mlp, xm)[0, 1:].float().abs() * wd  # (T-1, I) вклад нейронов
            acc["conc25"][l] += (c.topk(kap, dim=-1).values.sum(-1) / c.sum(-1).clamp_min(1e-9)).mean().item()
            acc["active"][l] += (c > 0.01 * c.max(dim=-1, keepdim=True).values).float().sum(-1).mean().item()
            e_read = ((m1 @ P_read.to(dev)) ** 2).sum(-1) / (m1 ** 2).sum(-1).clamp_min(1e-9)
            e_rand = ((m1 @ P_rand) ** 2).sum(-1) / (m1 ** 2).sum(-1).clamp_min(1e-9)
            acc["read_sink"][l] += e_read.mean().item(); acc["read_rand"][l] += e_rand.mean().item()
            mk = TopKMLP(Lr.mlp, kap)(xm)[0, 1:].float()
            err = (mk - m1).norm(dim=-1) / mid1.norm(dim=-1).clamp_min(1e-6)
            acc["err_cont"][l] += err[cont].sum().item(); acc["err_start"][l] += err[~cont].sum().item()
            cnt_cont[l] += cont.sum().item(); cnt_start[l] += (~cont).sum().item()
            mc = m1.cpu().double()  # MPS не умеет float64 — сначала на CPU
            cov[l] += mc.T @ mc; mean[l] += mc.sum(0)
            h = mid + m
        n_tok += ids.shape[1] - 1
    res = {"model": args.model, "k_sink": ks, "read_rank": r_read, "layers": {}}
    prof = json.loads((ROOT / "results" / args.model / "criterion" / "profile.json").read_text())["profile"]
    eff = []
    for l in range(L):
        C = cov[l] / n_tok - torch.outer(mean[l] / n_tok, mean[l] / n_tok)
        ev = torch.linalg.eigvalsh(C).clamp_min(0)
        eff.append(float(ev.sum() ** 2 / (ev ** 2).sum().clamp_min(1e-12)))
    nW = wins.shape[0]
    out = {k: (acc[k] / nW).tolist() for k in ("share", "conc25", "active", "read_sink", "read_rand")}
    out["eff_rank"] = eff
    out["err_cont"] = (acc["err_cont"] / cnt_cont.clamp_min(1)).tolist()
    out["err_start"] = (acc["err_start"] / cnt_start.clamp_min(1)).tolist()
    out["kl_mlp"] = prof["kl_mlp"]
    out["inj_mlp"] = prof["inj_mlp"]
    res["layers"] = out
    res["spearman_with_kl_mlp"] = {k: spearman(out[k], out["kl_mlp"]) for k in ("share", "conc25", "active", "eff_rank", "read_sink")}
    res["spearman_with_kl_mlp"]["read_sink_over_rand"] = spearman([a / max(b, 1e-9) for a, b in zip(out["read_sink"], out["read_rand"])], out["kl_mlp"])
    for l in range(L):
        print(f"слой {l:2d}: share {out['share'][l]:.3f} conc25 {out['conc25'][l]:.3f} active {out['active'][l]:6.0f} eff_rank {eff[l]:6.1f} "
              f"read_sink {out['read_sink'][l]:.3f} (rand {out['read_rand'][l]:.3f}) err cont/start {out['err_cont'][l]:.3f}/{out['err_start'][l]:.3f} kl_mlp {out['kl_mlp'][l]:.4f}", flush=True)
    print("Спирмен с kl_mlp:", {k: round(v, 2) for k, v in res["spearman_with_kl_mlp"].items()})
    save_json(res, ROOT / "results" / args.model / "criterion" / "mlp_mechanism.json")


if __name__ == "__main__":
    main()
