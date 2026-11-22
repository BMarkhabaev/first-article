"""Механизмы голов внимания слоя 0 по весам (архитектура Qwen3: MNN-Pocket-0.3B, Qwen3-0.6B, Qwen3-1.7B).

Вход слоя 0 — эмбеддинг токена, поэтому q, k, v каждой головы — точные функции токена (таблицы из весов):
    z^q_{h,m}(x) = q_h(x)[m] + i·q_h(x)[m+D/2],  z^k_{g,m}(y) аналогично,  θ_m = base^(-2m/D),
    s_h(x_i, y_j, Δ) = scale · Re Σ_m z^q_{h,m}(x_i) · conj(z^k_{g,m}(y_j)) · e^{iθ_m Δ},   Δ = i − j ≥ 0.
Логит раскладывается точно (черта — среднее по токенам с частотами p(x) калибровки, тильда — отклонение):
    s = q̄Rk̄ (позиция: ядро P_h(Δ)) + q̄Rk̃(y) (заметность ключа) + q̃(x)Rk̄ (запрос меняет ядро) + q̃(x)Rk̃(y) (пара токенов).
Скрипт считает по весам, без прогона текста через модель:
  1) выбросы весов нормализации (input_layernorm, q_norm, k_norm, post_attention_layernorm) и какой частоте RoPE
     соответствует каждая компонента q_norm/k_norm;
  2) для каждой головы: позиционное ядро P_h(Δ) и его частотный состав (амплитуда, фаза, сдвиг пика Δ* = −ψ_m/θ_m),
     разброс каждого из четырёх слагаемых в логитах, заметность ключей (списки токенов), совпадение токенов
     («тот же токен»: диагональ парного слагаемого), главные компоненты парного слагаемого (SVD) с токенами;
  3) OV-цепь: что голова пишет в остаточный поток (W_O·v(x)): норма, доля постоянного вектора, эффективный ранг,
     копирование токена (ранг самого токена в прямом вкладе в логиты), что продвигает постоянная часть;
  4) роль главной компоненты k_norm: распределение k_c(x) по токенам, q_c по головам, размах логита.
Затем проверка на тексте (внимание считается из тех же таблиц, то есть точно): масса по расстояниям, масса на
повторах токена, какие токены притягивают внимание и согласуется ли это с заметностью из весов.

  python head_mechanisms.py --model pocket0.3b
Результат: results/<m>/heads/layer0.json и layer0.md.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from detlayers.common import calibration_windows, load_model, wikitext_windows  # noqa: E402
from table_attention import dims, project  # noqa: E402

DELTAS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
BUCKETS = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 10 ** 9)]
BNAMES = ["0", "1", "2", "3-4", "5-8", "9-16", "17-32", "33-64", "65+"]
PROBES = [" New", " United", " of", " the", ".", ",", " in", " was", " (", " he", " 1", "\n"]


def cplx(t):
    M = t.shape[-1] // 2
    return torch.complex(t[..., :M].double(), t[..., M:].double())


def realvec(z):
    return torch.cat([z.real, z.imag], -1)


def wmean(x, p):
    return (x * p.view(-1, *[1] * (x.dim() - 1))).sum(0)


def show(tok, i):
    return repr(tok.decode([int(i)]))[1:-1]


def toks(tok, ids, vals=None, n=12):
    return [show(tok, i) for i in ids[:n]] if vals is None else [f"{show(tok, i)} ({v:+.1f})" for i, v in zip(ids[:n], vals[:n])]


def outliers(w, k=6):
    a = w.abs()
    t = a.topk(min(k, a.numel()))
    return {"n": int(w.numel()), "median_abs": float(a.median()), "max_abs": float(a.max()),
            "max_over_median": float(a.max() / a.median().clamp_min(1e-12)),
            "top_dims": t.indices.tolist(), "top_vals": [float(w[i]) for i in t.indices]}


@torch.no_grad()
def norm_weights(model, theta, n_layers):
    """Выбросы весов RMSNorm по слоям; для q_norm/k_norm — частота RoPE и период каждой большой компоненты."""
    out = []
    M = theta.numel()
    for l in range(n_layers):
        L = model.model.layers[l]
        row = {"layer": l}
        for name, mod in (("input_layernorm", L.input_layernorm), ("post_attention_layernorm", L.post_attention_layernorm),
                          ("q_norm", L.self_attn.q_norm), ("k_norm", L.self_attn.k_norm)):
            w = mod.weight.detach().float().cpu()
            o = outliers(w)
            if name in ("q_norm", "k_norm"):
                o["top_freq"] = [d % M for d in o["top_dims"]]
                o["top_part"] = ["cos" if d < M else "sin" for d in o["top_dims"]]
                o["top_period_tokens"] = [float(2 * math.pi / theta[d % M]) for d in o["top_dims"]]
            row[name] = o
        out.append(row)
    return out


@torch.no_grad()
def analyze(args):
    t0 = time.time()
    dev = torch.device("cpu")
    model, tok = load_model(args.model, dev, "fp32")
    H, G, D = dims(model)
    M, rep = D // 2, H // G
    layer = model.model.layers[0]
    att = layer.self_attn
    scale = float(att.scaling)
    theta = model.model.rotary_emb.inv_freq.double().cpu()  # (M,)
    period = 2 * math.pi / theta
    E = model.model.embed_tokens.weight.detach()
    V, d = E.shape
    Wu = model.get_output_embeddings().weight.detach()
    res = {"model": args.model, "layer": 0, "H": H, "G": G, "D": D, "d": d, "vocab": V, "scale": scale,
           "rope_period_tokens": period.tolist(), "tied_unembedding": bool(Wu.data_ptr() == E.data_ptr()),
           "norm_weights": norm_weights(model, theta, model.config.num_hidden_layers)}

    # --- частоты токенов (только как веса усреднения и фильтр редких токенов)
    calib = calibration_windows(tok, args.calib_tokens, 1024)
    cnt = torch.bincount(calib.flatten(), minlength=V).double()
    sel = (cnt >= args.min_count).nonzero().flatten()
    p = (cnt[sel] / cnt[sel].sum())
    N = sel.numel()
    common = cnt[sel] >= args.list_min_count  # токены, которые показываем в списках
    res["tokens_used"] = int(N)
    res["calib_tokens"] = int(calib.numel())
    res["token_mass_covered"] = float(cnt[sel].sum() / cnt.sum())

    q, k, v = project(att, layer.input_layernorm(E[sel]))  # (N,H,D), (N,G,D), (N,G,D)
    zq, zk = cplx(q), cplx(k)  # (N,H,M), (N,G,M)
    zqm, zkm = wmean(zq, p), wmean(zk, p)  # (H,M), (G,M)
    zqt, zkt = zq - zqm, zk - zkm
    slow = theta * args.ctx < 0.5  # частоты, у которых поворот на длине контекста пренебрежим (cos > 0.88)
    res["slow_freqs"] = slow.nonzero().flatten().tolist()
    res["ctx"] = args.ctx
    Tk = args.ctx
    dl = torch.arange(Tk, dtype=torch.float64)
    rot_all = torch.exp(1j * dl[:, None] * theta[None, :])  # (Tk, M)
    rot_d = torch.exp(1j * torch.tensor(DELTAS, dtype=torch.float64)[:, None] * theta[None, :])  # (nΔ, M)

    # --- главная компонента k_norm и q_norm слоя 0
    gk = att.k_norm.weight.detach().double().cpu()
    gq = att.q_norm.weight.detach().double().cpu()
    ck, cq = int(gk.abs().argmax()), int(gq.abs().argmax())
    kc = k.double()[:, :, ck]  # (N,G)
    chan = {"k_norm_dim": ck, "k_norm_gamma": float(gk[ck]), "k_norm_median": float(gk.abs().median()), "freq": ck % M,
            "part": "cos" if ck < M else "sin", "period_tokens": float(period[ck % M]),
            "q_norm_dim": cq, "q_norm_gamma": float(gq[cq]), "q_norm_median": float(gq.abs().median()),
            "q_freq": cq % M, "q_period_tokens": float(period[cq % M]), "groups": []}
    for g in range(G):
        x = kc[:, g]
        mu = float((x * p).sum())
        sd = float((((x - mu) ** 2) * p).sum().sqrt())
        order = torch.argsort(x.masked_fill(~common, float("nan")).nan_to_num(-1e9), descending=True)
        order_lo = torch.argsort(x.masked_fill(~common, float("nan")).nan_to_num(1e9))
        chan["groups"].append({"group": g, "mean": mu, "std": sd, "frac_positive": float((p * (x > 0)).sum()),
                               "top_tokens": toks(tok, sel[order], x[order]), "bottom_tokens": toks(tok, sel[order_lo], x[order_lo])})
    res["dominant_channel"] = chan

    # --- текст для проверки: внимание из таблиц
    wins = wikitext_windows(tok, "wikitext-2-raw-v1", "test", args.text_len, max_windows=args.text_windows)
    Tt = wins.shape[1]
    it = torch.arange(Tt)
    dist = it[:, None] - it[None, :]
    causal = dist >= 0
    rot_t = torch.exp(1j * it.double()[:, None] * theta[None, :])  # (Tt, M)
    bm = torch.stack([(dist >= a) & (dist <= b) for a, b in BUCKETS]).double()  # (nb,Tt,Tt)
    emp = {h: {"mass": torch.zeros(len(BUCKETS), dtype=torch.float64), "dup": 0.0, "dup_base": 0.0, "dup_n": 0,
               "pos0": 0.0, "ent": 0.0, "recv": torch.zeros(V, dtype=torch.float64)} for h in range(H)}
    occ = torch.zeros(V, dtype=torch.float64)
    for w in wins:
        qw, kw, _ = project(att, layer.input_layernorm(E[w]))
        zqw = cplx(qw) * rot_t[:, None, :]
        zkw = cplx(kw) * rot_t[:, None, :]
        same = (w[:, None] == w[None, :]) & (dist > 0)
        has = same.any(1)
        occ += torch.bincount(w, minlength=V).double()
        for h in range(H):
            a, b = realvec(zqw[:, h]), realvec(zkw[:, h // rep])
            s = (a @ b.T) * scale
            pr = torch.softmax(s.masked_fill(~causal, float("-inf")), -1)
            e = emp[h]
            valid = it >= 64  # запросы, у которых есть дальние ключи
            e["mass"] += (bm * pr[None]).sum(-1)[:, valid].mean(-1)
            e["pos0"] += float(pr[valid, 0].mean())
            e["ent"] += float(torch.exp(-(pr.clamp_min(1e-30).log() * pr).sum(-1))[valid].mean())
            if has.any():
                e["dup"] += float((pr * same).sum(-1)[has].sum())
                e["dup_base"] += float((same.sum(-1).double() / (it + 1).double())[has].sum())
                e["dup_n"] += int(has.sum())
            e["recv"] += torch.zeros(V, dtype=torch.float64).index_add_(0, w, (pr * (Tt / pr.sum(0).sum())).sum(0))
    nW = wins.shape[0]

    # --- OV: общие величины
    Wo = att.o_proj.weight.detach().double()  # (d, H*D)
    Ed = E[sel].double()
    Em = wmean(Ed, p)
    Et = Ed - Em
    e_rms = float(((Ed ** 2).sum(-1) * p).sum().sqrt())
    gpost = layer.post_attention_layernorm.weight.detach().double().abs()
    big_dims = gpost.topk(8).indices
    probe = [(s, tok.encode(s, add_special_tokens=False)) for s in PROBES]
    probe = [(s, ids[0]) for s, ids in probe if len(ids) == 1 and (sel == ids[0]).any()]
    samp = torch.argsort(p, descending=True)[: args.copy_tokens]
    Wu_sel = Wu[sel].double()

    heads = []
    for h in range(H):
        g = h // rep
        r = {"head": h, "kv_group": g}
        # 1) позиционное ядро
        cm = zqm[h] * zkm[g].conj()  # (M,)
        P = scale * (cm[None, :] * rot_all).real.sum(-1)  # (Tk,)
        amp = scale * cm.abs()
        psi = torch.angle(cm)
        peak = ((-psi) % (2 * math.pi)) / theta
        top = amp.topk(6).indices
        r["kernel"] = {"P": {str(dv): float(P[dv]) for dv in [0, 1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512] if dv < Tk},
                       "argmax_delta": int(P.argmax()), "range_logits": float(P.max() - P.min()),
                       "slow_share_of_amp": float(amp[slow].sum() / amp.sum()),
                       "top_freqs": [{"m": int(m), "amp_logits": float(amp[m]), "period": float(period[m]),
                                      "phase": float(psi[m]), "peak_delta": float(peak[m])} for m in top]}
        sm = torch.softmax(P[:256], 0)  # внимание «типичного» запроса на 256 ключей, если все токены средние
        r["kernel"]["softmax_mass_T256"] = {n: float(sm[a:min(b, 255) + 1].sum()) for n, (a, b) in zip(BNAMES, BUCKETS)}
        # 2) разброс слагаемых (в логитах)
        b_xd = scale * torch.einsum("m,nm,dm->nd", zqm[h], zkt[:, g].conj(), rot_d).real  # (N,nΔ) заметность ключа
        c_xd = scale * torch.einsum("nm,m,dm->nd", zqt[:, h], zkm[g].conj(), rot_d).real  # запрос меняет ядро
        Cb = torch.einsum("n,ni,nj->ij", p, realvec(zkt[:, g]), realvec(zkt[:, g]))
        sd_pair, dup_z = [], []
        for di in range(len(DELTAS)):
            a = realvec(zqt[:, h] * rot_d[di][None, :])
            Ca = torch.einsum("n,ni,nj->ij", p, a, a)
            sdp = scale * float(torch.trace(Ca @ Cb).clamp_min(0).sqrt())
            diag = scale * float((p * (a * realvec(zkt[:, g])).sum(-1)).sum())
            sd_pair.append(sdp)
            dup_z.append(diag / max(sdp, 1e-12))
        Pd = P[torch.tensor([dv for dv in DELTAS if dv < Tk])]
        r["terms_std_logits"] = {"position_over_delta": float(P[1:].std()), "position_over_listed_deltas": float(Pd.std()),
                                 "key_salience": float(((b_xd ** 2) * p[:, None]).sum(0).sqrt().mean()),
                                 "query_modulation": float(((c_xd ** 2) * p[:, None]).sum(0).sqrt().mean()),
                                 "pair": float(sum(sd_pair) / len(sd_pair)),
                                 "pair_by_delta": dict(zip(map(str, DELTAS), sd_pair))}
        r["same_token_z_by_delta"] = dict(zip(map(str, DELTAS), dup_z))
        # 3) заметность ключей (медленные частоты: не зависит от расстояния)
        sal = scale * (zqm[h][slow][None, :] * zkt[:, g][:, slow].conj()).real.sum(-1)  # (N,)
        sal_sd = float(((sal ** 2) * p).sum().sqrt())
        sal_f = sal.masked_fill(~common, float("nan"))
        hi = torch.argsort(sal_f.nan_to_num(-1e9), descending=True)
        lo = torch.argsort(sal_f.nan_to_num(1e9))
        via = scale * zqm[h][ck % M].real * (kc[:, g] - (kc[:, g] * p).sum()) if ck < M else \
            scale * zqm[h][ck % M].imag * (kc[:, g] - (kc[:, g] * p).sum())
        r["salience"] = {"std_logits": sal_sd, "top_tokens": toks(tok, sel[hi], sal[hi]), "bottom_tokens": toks(tok, sel[lo], sal[lo]),
                         "share_from_dominant_k_channel": float(1 - (((sal - via) ** 2) * p).sum() / ((sal ** 2) * p).sum().clamp_min(1e-30)),
                         "corr_with_delta1_salience": float(torch.corrcoef(torch.stack([sal, b_xd[:, 0]]))[0, 1]),
                         "mean_q_on_channel": float(zqm[h][ck % M].real if ck < M else zqm[h][ck % M].imag),
                         "std_q_on_channel": float((((q.double()[:, h, ck] - (q.double()[:, h, ck] * p).sum()) ** 2) * p).sum().sqrt())}
        # 4) парное слагаемое на медленных частотах: SVD
        A = realvec(zqt[:, h][:, slow]) * p.sqrt()[:, None]
        B = realvec(zkt[:, g][:, slow]) * p.sqrt()[:, None]
        if A.shape[1] > 0:
            Ua, Sa, Vah = torch.linalg.svd(A, full_matrices=False)
            Ub, Sb, Vbh = torch.linalg.svd(B, full_matrices=False)
            U, C, Vh = torch.linalg.svd((Sa[:, None] * (Vah @ Vbh.T)) * Sb[None, :])
            Lq = (Ua @ U) / p.sqrt()[:, None]
            Lk = (Ub @ Vh.T) / p.sqrt()[:, None]
            comps = []
            for i in range(min(3, C.numel())):
                lq, lk = Lq[:, i].masked_fill(~common, float("nan")), Lk[:, i].masked_fill(~common, float("nan"))
                comps.append({"share": float(C[i] ** 2 / (C ** 2).sum()), "sv_logits": float(scale * C[i]),
                              "query_plus": toks(tok, sel[torch.argsort(lq.nan_to_num(-1e9), descending=True)], n=10),
                              "query_minus": toks(tok, sel[torch.argsort(lq.nan_to_num(1e9))], n=10),
                              "key_plus": toks(tok, sel[torch.argsort(lk.nan_to_num(-1e9), descending=True)], n=10),
                              "key_minus": toks(tok, sel[torch.argsort(lk.nan_to_num(1e9))], n=10)})
            r["pair_slow_svd"] = comps
        # 5) OV-цепь
        Woh = Wo[:, h * D:(h + 1) * D]  # (d, D)
        vv = v.double()[:, g]  # (N, D)
        vm = wmean(vv, p)
        o = vv @ Woh.T  # (N, d)
        om = Woh @ vm
        ot = o - om
        o_rms = float(((o ** 2).sum(-1) * p).sum().sqrt())
        Cv = torch.einsum("n,ni,nj->ij", p, vv - vm, vv - vm)
        ev = torch.linalg.eigvalsh(Woh.T @ Woh @ Cv + 1e-12 * torch.eye(D, dtype=torch.float64)).clamp_min(0) if False else \
            torch.linalg.svdvals(torch.linalg.cholesky(Cv + 1e-9 * torch.eye(D, dtype=torch.float64)).T @ Woh.T) ** 2
        cosE = float((p * torch.nn.functional.cosine_similarity(ot, Et, dim=-1)).sum())
        on = model.model.norm(o[samp].float()).double()
        lg = on @ Wu_sel.T  # (S, N) прямой вклад в логиты (среди используемых токенов)
        selfl = lg[torch.arange(samp.numel()), samp]
        rank = (lg > selfl[:, None]).sum(-1)
        lm = model.model.norm(om.float()[None]).double()[0] @ Wu_sel.T
        lm_f = lm.masked_fill(~common, float("nan"))
        ex = {}
        for s, tid in probe[:8]:
            row = (sel == tid).nonzero()[0, 0]
            l1 = (model.model.norm(ot[row].float()[None]).double()[0] @ Wu_sel.T).masked_fill(~common, float("nan"))
            ex[s] = toks(tok, sel[torch.argsort(l1.nan_to_num(-1e9), descending=True)], n=6)
        td = om.abs().topk(5).indices
        r["ov"] = {"rms_out_over_rms_embed": o_rms / e_rms, "const_share": float((om ** 2).sum() / (o_rms ** 2)),
                   "eff_rank": float(ev.sum() ** 2 / (ev ** 2).sum().clamp_min(1e-30)),
                   "cos_with_own_embedding": cosE, "self_rank_median": float(rank.double().median()),
                   "self_top1": float((rank == 0).double().mean()), "self_top10": float((rank < 10).double().mean()),
                   "const_top_dims": td.tolist(), "const_top_dims_share": float((om[td] ** 2).sum() / (om ** 2).sum().clamp_min(1e-30)),
                   "const_share_on_big_postnorm_dims": float((om[big_dims] ** 2).sum() / (om ** 2).sum().clamp_min(1e-30)),
                   "const_promotes": toks(tok, sel[torch.argsort(lm_f.nan_to_num(-1e9), descending=True)], n=10),
                   "const_suppresses": toks(tok, sel[torch.argsort(lm_f.nan_to_num(1e9))], n=10),
                   "token_part_promotes": ex}
        # 6) проверка на тексте
        e = emp[h]
        recv = (e["recv"] / occ.clamp_min(1))[sel]  # во сколько раз токен получает больше внимания, чем в среднем
        seen = (occ[sel] >= 5)
        rf = recv.masked_fill(~seen, float("nan"))
        okc = seen & (recv > 0)
        rho = float(torch.corrcoef(torch.stack([sal[okc].argsort().argsort().double(), recv[okc].argsort().argsort().double()]))[0, 1]) \
            if okc.sum() > 10 else float("nan")
        r["text"] = {"mass_by_distance": dict(zip(BNAMES, (e["mass"] / nW).tolist())), "pos0": e["pos0"] / nW,
                     "effective_keys": e["ent"] / nW,
                     "dup_mass": e["dup"] / max(e["dup_n"], 1), "dup_baseline": e["dup_base"] / max(e["dup_n"], 1),
                     "attractors": toks(tok, sel[torch.argsort(rf.nan_to_num(-1e9), descending=True)],
                                        recv[torch.argsort(rf.nan_to_num(-1e9), descending=True)]),
                     "spearman_salience_vs_received": rho}
        heads.append(r)
        k_, t_ = r["kernel"], r["text"]
        print(f"[{args.model}] голова {h:2d}: ядро argmax Δ={k_['argmax_delta']}, размах {k_['range_logits']:.1f}; "
              f"σ поз/ключ/запрос/пара = {r['terms_std_logits']['position_over_delta']:.1f}/{r['terms_std_logits']['key_salience']:.1f}/"
              f"{r['terms_std_logits']['query_modulation']:.1f}/{r['terms_std_logits']['pair']:.1f}; "
              f"текст: Δ=1 {t_['mass_by_distance']['1']:.2f}, 65+ {t_['mass_by_distance']['65+']:.2f}, "
              f"повтор {t_['dup_mass']:.2f} (база {t_['dup_baseline']:.2f})", flush=True)
    res["heads"] = heads
    res["seconds"] = round(time.time() - t0, 1)
    out = ROOT / "results" / args.model / "heads"
    out.mkdir(parents=True, exist_ok=True)
    (out / "layer0.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    (out / "layer0.md").write_text(report(res))
    print(f"сохранено {out / 'layer0.json'} и layer0.md; {res['seconds']}s")


def report(res):
    L = [f"# Головы слоя 0: {res['model']}", "",
         f"H={res['H']} голов запросов, G={res['G']} групп ключей/значений, D={res['D']}, d={res['d']}; масштаб логита {res['scale']:.4f}; "
         f"токенов в анализе {res['tokens_used']} ({100 * res['token_mass_covered']:.1f} % массы текста); медленные частоты "
         f"(поворот на {res['ctx']} токенах < 0.5 рад): m ≥ {min(res['slow_freqs']) if res['slow_freqs'] else '—'}.", ""]
    c = res["dominant_channel"]
    L += ["## Веса нормализации", "",
          f"k_norm слоя 0: компонента {c['k_norm_dim']} = {c['k_norm_gamma']:.1f} (медиана {c['k_norm_median']:.2f}); это {c['part']}-часть "
          f"частоты m={c['freq']} с периодом {c['period_tokens']:.0f} токенов. q_norm: компонента {c['q_norm_dim']} = {c['q_norm_gamma']:.1f} "
          f"(медиана {c['q_norm_median']:.2f}), частота m={c['q_freq']}, период {c['q_period_tokens']:.0f}.", ""]
    for g in c["groups"]:
        L += [f"- группа ключей {g['group']}: k_c(x) среднее {g['mean']:+.1f}, σ {g['std']:.1f}, доля положительных {g['frac_positive']:.2f}; "
              f"верх: {', '.join(g['top_tokens'][:8])}; низ: {', '.join(g['bottom_tokens'][:8])}"]
    L += ["", "| слой | input_ln макс/мед | post_attn_ln макс/мед | q_norm макс (dim, m, период) | k_norm макс (dim, m, период) |", "|---|---|---|---|---|"]
    for r in res["norm_weights"]:
        def qk(o):
            return f"{o['top_vals'][0]:.1f} / мед {o['median_abs']:.1f} (dim {o['top_dims'][0]}, m={o['top_freq'][0]}, {o['top_period_tokens'][0]:.0f})"
        L.append(f"| {r['layer']} | {r['input_layernorm']['max_abs']:.2f} / {r['input_layernorm']['median_abs']:.2f} (dims {r['input_layernorm']['top_dims'][:3]}) "
                 f"| {r['post_attention_layernorm']['max_abs']:.2f} / {r['post_attention_layernorm']['median_abs']:.2f} (dims {r['post_attention_layernorm']['top_dims'][:3]}) "
                 f"| {qk(r['q_norm'])} | {qk(r['k_norm'])} |")
    L += ["", "## Сводка по головам", "",
          "| h | g | ядро: argmax Δ | размах | σ поз | σ ключ | σ запрос | σ пара | тот же токен z(Δ=1/64) | текст Δ=0 | Δ=1 | Δ=2 | 3–8 | 9–64 | 65+ | повтор/база | эфф. ключей | OV: норма | пост. доля | копия top-10 |",
          "|" + "---|" * 20]
    for r in res["heads"]:
        k, t, s, o = r["kernel"], r["text"], r["terms_std_logits"], r["ov"]
        m = t["mass_by_distance"]
        L.append(f"| {r['head']} | {r['kv_group']} | {k['argmax_delta']} | {k['range_logits']:.1f} | {s['position_over_delta']:.1f} | {s['key_salience']:.1f} "
                 f"| {s['query_modulation']:.1f} | {s['pair']:.1f} | {r['same_token_z_by_delta']['1']:+.2f}/{r['same_token_z_by_delta']['64']:+.2f} "
                 f"| {m['0']:.2f} | {m['1']:.2f} | {m['2']:.2f} | {m['3-4'] + m['5-8']:.2f} | {m['9-16'] + m['17-32'] + m['33-64']:.2f} | {m['65+']:.2f} "
                 f"| {t['dup_mass']:.2f}/{t['dup_baseline']:.2f} | {t['effective_keys']:.0f} | {o['rms_out_over_rms_embed']:.2f} | {o['const_share']:.2f} | {o['self_top10']:.2f} |")
    for r in res["heads"]:
        k, t, s, o, sa = r["kernel"], r["text"], r["terms_std_logits"], r["ov"], r["salience"]
        L += ["", f"### Голова {r['head']} (группа ключей {r['kv_group']})", "",
              "**Позиционное ядро P(Δ), логиты:** " + ", ".join(f"Δ={a}: {b:+.1f}" for a, b in k["P"].items()) +
              f". Доля медленных частот в амплитуде {k['slow_share_of_amp']:.2f}.", "",
              "Главные частоты: " + "; ".join(f"m={f['m']} (период {f['period']:.0f}, амплитуда {f['amp_logits']:.1f}, пик на Δ*={f['peak_delta']:.1f})" for f in k["top_freqs"]) + ".", "",
              "Внимание «среднего» токена на 256 ключей по ядру: " + ", ".join(f"{a}: {b:.2f}" for a, b in k["softmax_mass_T256"].items()) + ".", "",
              f"**Заметность ключей** (σ {sa['std_logits']:.1f} логита; доля от главной компоненты k_norm {sa['share_from_dominant_k_channel']:.2f}; "
              f"q на этой компоненте: среднее {sa['mean_q_on_channel']:+.2f}, σ {sa['std_q_on_channel']:.2f}; согласие с Δ=1: {sa['corr_with_delta1_salience']:.2f}).",
              "- притягивают: " + ", ".join(sa["top_tokens"]), "- отталкивают: " + ", ".join(sa["bottom_tokens"]), ""]
        for i, cmp in enumerate(r.get("pair_slow_svd", [])):
            L += [f"**Пара токенов, компонента {i + 1}** (доля {cmp['share']:.2f}, {cmp['sv_logits']:.1f} логита): запросы [+] {', '.join(cmp['query_plus'])} / [−] {', '.join(cmp['query_minus'])}; "
                  f"ключи [+] {', '.join(cmp['key_plus'])} / [−] {', '.join(cmp['key_minus'])}", ""]
        L += [f"**OV:** норма записи {o['rms_out_over_rms_embed']:.2f} от нормы эмбеддинга; постоянная часть {o['const_share']:.2f} энергии "
              f"(её главные координаты {o['const_top_dims']}, доля {o['const_top_dims_share']:.2f}; на 8 самых больших координатах post_attention_layernorm {o['const_share_on_big_postnorm_dims']:.2f}); "
              f"эффективный ранг токенной части {o['eff_rank']:.1f}; косинус с эмбеддингом своего токена {o['cos_with_own_embedding']:+.2f}; "
              f"сам токен в top-1 / top-10 прямого вклада: {o['self_top1']:.2f} / {o['self_top10']:.2f} (медианный ранг {o['self_rank_median']:.0f}).",
              "- постоянная часть продвигает: " + ", ".join(o["const_promotes"]),
              "- постоянная часть подавляет: " + ", ".join(o["const_suppresses"]),
              "- токенная часть (ключ → что продвигает): " + "; ".join(f"{repr(a)[1:-1]} → {', '.join(b)}" for a, b in o["token_part_promotes"].items()), "",
              "**На тексте:** масса по расстояниям " + ", ".join(f"{a}: {b:.2f}" for a, b in t["mass_by_distance"].items()) +
              f"; на позицию 0: {t['pos0']:.3f}; эффективное число ключей {t['effective_keys']:.1f}; на повторы токена {t['dup_mass']:.2f} "
              f"(при равномерном внимании {t['dup_baseline']:.2f}); согласие заметности из весов с полученным вниманием (Спирмен) {t['spearman_salience_vs_received']:.2f}.",
              "- больше всего внимания получают: " + ", ".join(t["attractors"])]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pocket0.3b")
    ap.add_argument("--calib-tokens", type=int, default=262_144)
    ap.add_argument("--min-count", type=int, default=3, help="токен входит в анализ, если встретился в калибровке столько раз")
    ap.add_argument("--list-min-count", type=int, default=20, help="токен показывается в списках, если встретился столько раз")
    ap.add_argument("--ctx", type=int, default=1024, help="длина контекста для ядра и определения медленных частот")
    ap.add_argument("--text-windows", type=int, default=8)
    ap.add_argument("--text-len", type=int, default=512)
    ap.add_argument("--copy-tokens", type=int, default=1000)
    analyze(ap.parse_args())


if __name__ == "__main__":
    main()
