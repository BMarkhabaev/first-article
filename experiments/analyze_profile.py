#!/usr/bin/env python
"""Априорный критерий заменяемости слоя (карта по всей глубине).

Для каждого слоя ℓ = 0..L−1 на состояниях учителя (без калибровки) считает:
  статистики:  mass_out — масса внимания вне окна 64 (без стока); mass_sink — масса на позиции 0;
               e_attn_local — относит. ошибка выхода внимания при окне 64 + сток;
               inj_attn — та же ошибка относительно нормы остаточного потока;
               top25_share — доля вклада 25% самых сильных нейронов MLP; e_mlp_local, inj_mlp — аналогично
               для отбора 25% нейронов по готовым активациям (оракул);
  измерения:   kl_attn — KL на выходе модели, если заменить ТОЛЬКО внимание слоя ℓ окном;
               kl_mlp — то же для MLP (25% нейронов); kl_noise — гауссово возмущение входа слоя ℓ
               с относительной нормой eps (изотропная чувствительность).
Проверяет, предсказывают ли локальные статистики измеренный KL (ранговая корреляция), и сравнивает
сумму одиночных KL первых k слоёв с измеренными кривыми howmany (аддитивность).
Результат: results/<model>/criterion/profile.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detlayers.attention import DetAttention, SparseAttention, exact_attention_weights  # noqa: E402
from detlayers.common import embed, load_model, pick_device, run_layers, save_json, seed_all, wikitext_windows  # noqa: E402
from detlayers.methods import Replacement  # noqa: E402
from detlayers.mlp import ExactPos0, TopKMLP, _z  # noqa: E402

ROOT = Path(__file__).resolve().parent


def rel(a, b):
    a, b = a[:, 1:].float(), b[:, 1:].float()
    return ((a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-6)).mean().item()


def rel_to(a, b, ref):
    """||a−b|| относительно нормы ref (по позициям ≥1)."""
    return ((a[:, 1:].float() - b[:, 1:].float()).norm(dim=-1) / ref[:, 1:].float().norm(dim=-1).clamp_min(1e-6)).mean().item()


class Perturbed(nn.Module):
    """Гауссово возмущение входа слоя: относительная норма eps на токен, позиция 0 не трогается."""

    def __init__(self, layer, eps):
        super().__init__()
        self.layer, self.eps, self.seed = layer, eps, 0

    def forward(self, h, *a, **kw):
        g = torch.Generator().manual_seed(self.seed)
        noise = torch.randn(h.shape, generator=g).to(h.device)
        noise = noise / noise.norm(dim=-1, keepdim=True) * h.float().norm(dim=-1, keepdim=True) * self.eps
        noise[:, 0] = 0
        return self.layer(h + noise.to(h.dtype), *a, **kw)


@torch.no_grad()
def states(model, ids):
    h, pos, pe = embed(model, ids)
    hs = [h]
    for i in range(len(model.model.layers)):
        h = run_layers(model, model.model.layers, h, pos, pe, i, i + 1)
        hs.append(h)
    return hs, pos, pe


def spearman(x, y):
    x, y = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    rx, ry = x.argsort().argsort().double(), y.argsort().argsort().double()
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx * ry).sum() / (rx.norm() * ry.norm()).clamp_min(1e-12))


def load_howmany(model, name, ks):
    out = {}
    for k in ks:
        p = ROOT / "results" / model / "howmany" / f"{name}_k{k}.json"
        if p.exists():
            r = json.loads(p.read_text())["lm"]
            out[k] = {"kl": r["kl"], "dppl_pct": 100 * (r["ppl"] / r["ppl_teacher"] - 1)}
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="0.6b")
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--ratio", type=float, default=0.25)
    ap.add_argument("--eps", type=float, default=0.02)
    args = ap.parse_args()
    seed_all(0)
    device = pick_device(None)
    model, tok = load_model(args.model, device, "fp32")
    layers = model.model.layers
    L, I = len(layers), model.config.intermediate_size
    kap = int(args.ratio * I)
    wins = wikitext_windows(tok, "wikitext-2-raw-v1", "test", 1024, max_windows=args.windows)
    variants = {}
    for l in range(L):
        variants[f"attn{l}"] = (l, "self_attn", SparseAttention(layers[l].self_attn, window=args.window, n_sink=1))
        variants[f"mlp{l}"] = (l, "mlp", ExactPos0(layers[l].mlp, TopKMLP(layers[l].mlp, kap)))
        variants[f"noise{l}"] = (l, "block", Perturbed(layers[l], args.eps))
    keys = ["mass_out", "mass_sink", "e_attn_local", "inj_attn", "top25_share", "e_mlp_local", "inj_mlp",
            "kl_attn", "kl_mlp", "kl_noise", "top1_attn", "top1_mlp", "top1_noise", "relfinal_attn", "relfinal_mlp", "relfinal_noise"]
    acc = {k: torch.zeros(L, dtype=torch.float64) for k in keys}
    t0 = time.time()
    for s in range(wins.shape[0]):
        ids = wins[s:s + 1].to(device)
        hT, pos, pe = states(model, ids)
        outT = model(ids, output_hidden_states=False)
        lt = outT.logits[:, :-1].float()
        lpt = F.log_softmax(lt, -1)
        pt = lpt.exp()
        ntok = lt.shape[1]
        hfinal_T = hT[-1]
        # --- измерения: одиночные замены и возмущения
        for name, (l, attr, mod) in variants.items():
            if attr == "block":
                mod.seed = 1000 * l + s
            rep = Replacement(model)
            rep.add(l, attr, mod)
            rep.enable()
            hS, _, _ = states(model, ids)
            ls = model(ids).logits[:, :-1].float()
            rep.disable()
            kind = name.rstrip("0123456789")
            acc[f"kl_{kind}"][l] += (pt * (lpt - F.log_softmax(ls, -1))).sum().item() / ntok
            acc[f"top1_{kind}"][l] += (lt.argmax(-1) == ls.argmax(-1)).float().mean().item()
            acc[f"relfinal_{kind}"][l] += rel(hS[-1], hfinal_T)
        # --- локальные статистики
        for l in range(L):
            layer = layers[l]
            x = hT[l]
            xn = layer.input_layernorm(x)
            det = DetAttention(layer.self_attn)
            q, k, _ = det.qkv(xn, pe)
            P = exact_attention_weights(q, k, layer.self_attn.scaling)[0]  # (H,T,T)
            T = P.shape[-1]
            t = torch.arange(T, device=device).unsqueeze(1)
            j = torch.arange(T, device=device).unsqueeze(0)
            far = ((t - j) >= args.window) & (j != 0)
            acc["mass_out"][l] += (P * far).sum(-1)[:, 1:].mean().item()
            acc["mass_sink"][l] += P[:, 1:, 0].mean().item()
            aT, _ = layer.self_attn(xn, position_embeddings=pe, attention_mask=None)
            aW, _ = variants[f"attn{l}"][2](xn, position_embeddings=pe, attention_mask=None)
            acc["e_attn_local"][l] += rel(aW, aT)
            acc["inj_attn"][l] += rel_to(aW, aT, x)
            mid = x + aT
            xm = layer.post_attention_layernorm(mid)
            mT = layer.mlp(xm)
            mK = variants[f"mlp{l}"][2](xm)
            acc["e_mlp_local"][l] += rel(mK, mT)
            acc["inj_mlp"][l] += rel_to(mK, mT, mid)
            wd = layer.mlp.down_proj.weight.norm(dim=0)
            contrib = _z(layer.mlp, xm)[0, 1:].float().abs() * wd
            acc["top25_share"][l] += (contrib.topk(kap, dim=-1).values.sum(-1) / contrib.sum(-1).clamp_min(1e-9)).mean().item()
        print(f"окно {s + 1}/{wins.shape[0]}  {time.time() - t0:.0f}s", flush=True)
    n = wins.shape[0]
    prof = {k: [round(float(v) / n, 6) for v in acc[k]] for k in keys}
    # --- критерий: ранговые корреляции предикторов с измеренным KL
    corr = {
        "attn": {"mass_out→kl": spearman(prof["mass_out"], prof["kl_attn"]),
                 "e_attn_local→kl": spearman(prof["e_attn_local"], prof["kl_attn"]),
                 "inj_attn→kl": spearman(prof["inj_attn"], prof["kl_attn"]),
                 "inj_attn²·sens→kl": spearman([a * a * g / (args.eps ** 2) for a, g in zip(prof["inj_attn"], prof["kl_noise"])], prof["kl_attn"])},
        "mlp": {"1−top25_share→kl": spearman([1 - v for v in prof["top25_share"]], prof["kl_mlp"]),
                "e_mlp_local→kl": spearman(prof["e_mlp_local"], prof["kl_mlp"]),
                "inj_mlp→kl": spearman(prof["inj_mlp"], prof["kl_mlp"]),
                "inj_mlp²·sens→kl": spearman([a * a * g / (args.eps ** 2) for a, g in zip(prof["inj_mlp"], prof["kl_noise"])], prof["kl_mlp"])},
    }
    # направленная чувствительность: KL на единицу энергии возмущения, замена vs изотропный шум
    dir_ratio_attn = [(ka / max(ia, 1e-9) ** 2) / (kn / args.eps ** 2) if kn > 0 else None
                      for ka, ia, kn in zip(prof["kl_attn"], prof["inj_attn"], prof["kl_noise"])]
    dir_ratio_mlp = [(km / max(im, 1e-9) ** 2) / (kn / args.eps ** 2) if kn > 0 else None
                     for km, im, kn in zip(prof["kl_mlp"], prof["inj_mlp"], prof["kl_noise"])]
    # аддитивность: сумма одиночных KL первых k слоёв против howmany
    ks = [1, 2, 3, 4, 6, 8, 12]
    additive = {"attn_window": {}, "mlp_topk": {}}
    hm_w, hm_o = load_howmany(args.model, "L1-window64", ks), load_howmany(args.model, "ML2-topk25-dejavu-oracle", ks)
    for k in ks:
        additive["attn_window"][k] = {"sum_single_kl": sum(prof["kl_attn"][:k]), **hm_w.get(k, {})}
        additive["mlp_topk"][k] = {"sum_single_kl": sum(prof["kl_mlp"][:k]), **hm_o.get(k, {})}
    res = {"model": args.model, "windows": n, "window": args.window, "ratio": args.ratio, "eps": args.eps,
           "profile": prof, "spearman": corr,
           "directional_sensitivity_ratio": {"attn": dir_ratio_attn, "mlp": dir_ratio_mlp},
           "additivity": additive, "seconds": round(time.time() - t0)}
    out = ROOT / "results" / args.model / "criterion" / "profile.json"
    save_json(res, out)
    print(json.dumps({"spearman": corr}, ensure_ascii=False, indent=1))
    for l in range(L):
        print(f"слой {l:2d}: mass_out {prof['mass_out'][l]:.3f} sink {prof['mass_sink'][l]:.3f} "
              f"e_attn {prof['e_attn_local'][l]:.3f} kl_attn {prof['kl_attn'][l]:.4f} | "
              f"share25 {prof['top25_share'][l]:.3f} e_mlp {prof['e_mlp_local'][l]:.3f} kl_mlp {prof['kl_mlp'][l]:.4f} | "
              f"kl_noise {prof['kl_noise'][l]:.4f}")
    print("saved", out)


if __name__ == "__main__":
    main()
