"""Функции и механизмы голов внимания во всех слоях (архитектура Qwen3), продолжение head_mechanisms.py.

Логит внимания любой головы точно равен
    s_h(i, j) = scale · Re Σ_m z^q_{h,m}(i) · conj(z^k_{g,m}(j)) · e^{iθ_m (i−j)},
где z^q, z^k — запрос и ключ после QK-norm (до RoPE) в комплексной форме по частотам RoPE. В слое 0 они — функции
токена; в слоях ℓ ≥ 1 зависят от контекста, поэтому берутся из прогона текста. Механизм ищется в том, из чего они
собраны. Вход слоя — сумма записей в остаточный поток: r = эмбеддинг + Σ (головы внимания) + Σ (MLP) предыдущих
слоёв. RMSNorm и проекции линейны с точностью до скаляра на позицию, поэтому
    q_h(i) = γ_q ⊙ W_q (γ_in ⊙ Σ_c comp_c(i)) / (rms_r(i) · rms_q(i))
раскладывается по источникам c ТОЧНО. Для главных частот каждой головы скрипт выдаёт доли источников (эмбеддинг,
каждая предыдущая голова, каждый предыдущий MLP) в среднем запросе и среднем ключе — это и есть цепь в весах.

Для каждой головы каждого слоя:
  ядро      — позиционное ядро P(Δ) из средних q̄, k̄; главные частоты (амплитуда, период, сдвиг пика), спад на 512 токенах;
  слагаемые — разброс в логитах: позиция / заметность ключа / запрос меняет ядро / пара; доля q, k, объяснимая токеном;
  токены    — заметность ключей по токенам; z-оценка «тот же токен» по токенным средним;
  OV        — доля постоянного вектора, эффективный ранг, норма записи относительно остаточного потока;
  текст     — масса внимания по расстояниям, на позицию 0, на повторы токена, на индукционные цели (ключ после
              прошлого вхождения текущего токена), эффективное число ключей; внимание пересчитывается из q, k и
              сверяется с выходом модели;
  источники — доли компонент остаточного потока в z̄^q и z̄^k на трёх главных частотах;
  метка     — автоматическая по порогам (предыдущий токен, локальное окно, сток, тот же токен, индукция, …).

  python head_mechanisms_deep.py --model pocket0.3b
Результат: results/<m>/heads/all_layers.json и all_layers.md.
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

from detlayers.common import load_model, wikitext_windows  # noqa: E402
from head_mechanisms import BNAMES, BUCKETS, DELTAS, cplx, norm_weights, realvec, show, toks  # noqa: E402
from table_attention import dims, project  # noqa: E402


def label(t, k, tz):
    """Автоматическая метка по измеренному вниманию (пороги грубые; уточняется вручную)."""
    m = t["mass_by_distance"]
    near8 = m["0"] + m["1"] + m["2"] + m["3-4"] + m["5-8"]
    out = []
    if t["pos0"] > 0.5:
        out.append("сток (позиция 0)")
    elif t["pos0"] > 0.2:
        out.append("частично сток")
    if m["1"] > 0.4:
        out.append("предыдущий токен")
    elif m["0"] > 0.4:
        out.append("сам токен")
    elif m["2"] > 0.3:
        out.append("токен −2")
    elif near8 > 0.6:
        out.append("локальное окно ≤8")
    elif near8 + m["9-16"] + m["17-32"] + m["33-64"] > 0.75 and t["pos0"] <= 0.2:
        out.append("окно ≤64")
    if t["ind_mass"] > 0.25 and t["ind_mass"] > 5 * t["ind_baseline"]:
        out.append("индукция")
    if t["dup_mass"] > 0.25 and t["dup_mass"] > 5 * t["dup_baseline"]:
        out.append("тот же токен")
    if not out:
        out.append("дальний контекст" if m["65+"] > 0.3 else "смешанная")
    return ", ".join(out)


@torch.no_grad()
def run(args):
    t0 = time.time()
    dev = torch.device("cpu")
    model, tok = load_model(args.model, dev, "fp32")
    H, G, D = dims(model)
    M, rep = D // 2, H // G
    layers = model.model.layers
    NL = len(layers) if args.layers <= 0 else min(args.layers, len(layers))
    theta = model.model.rotary_emb.inv_freq.double().cpu()
    period = 2 * math.pi / theta
    d = model.config.hidden_size
    wins = wikitext_windows(tok, "wikitext-2-raw-v1", "test", args.text_len, max_windows=args.text_windows)
    nW, T = wins.shape
    it = torch.arange(T)
    dist = it[:, None] - it[None, :]
    causal = dist >= 0
    rot_t = torch.exp(1j * it.double()[:, None] * theta[None, :])
    bm = torch.stack([(dist >= a) & (dist <= b) for a, b in BUCKETS]).float()
    valid = it >= 64

    # ---- перехват: эмбеддинг, нормализованный вход внимания, выходы внимания и MLP, вход o_proj (по головам)
    cap = {}
    hooks = [model.model.embed_tokens.register_forward_hook(lambda m_, i, o: cap.__setitem__("emb", o[0].detach()))]
    for l in range(NL):
        L = layers[l]
        hooks.append(L.self_attn.register_forward_pre_hook(
            lambda m_, a, kw, l=l: cap.__setitem__(("xn", l), (kw.get("hidden_states") if "hidden_states" in kw else a[0])[0].detach()),
            with_kwargs=True))
        hooks.append(L.self_attn.register_forward_hook(lambda m_, i, o, l=l: cap.__setitem__(("attn", l), o[0][0].detach())))
        hooks.append(L.mlp.register_forward_hook(lambda m_, i, o, l=l: cap.__setitem__(("mlp", l), o[0].detach())))
        hooks.append(L.self_attn.o_proj.register_forward_pre_hook(lambda m_, i, l=l: cap.__setitem__(("oin", l), i[0][0].detach())))

    S = nW * T
    Q = [torch.zeros(S, H, D) for _ in range(NL)]
    K = [torch.zeros(S, G, D) for _ in range(NL)]
    Vv = [torch.zeros(S, G, D) for _ in range(NL)]
    ids_all = wins.flatten()
    emp = [[{"mass": torch.zeros(len(BUCKETS)), "pos0": 0.0, "ent": 0.0, "dup": 0.0, "dup_base": 0.0, "dup_n": 0,
             "ind": 0.0, "ind_base": 0.0, "ind_n": 0} for _ in range(H)] for _ in range(NL)]
    # источники: компоненты [emb] + [голова (l,h) для l<ℓ] + [mlp l для l<ℓ]
    accq = [torch.zeros(1 + l * (H + 1), H, D, dtype=torch.float64) for l in range(NL)]
    acck = [torch.zeros(1 + l * (H + 1), G, D, dtype=torch.float64) for l in range(NL)]
    rms_res = torch.zeros(NL, dtype=torch.float64)
    recon_err = torch.zeros(NL)
    attr_err = torch.zeros(NL)
    Wo = [layers[l].self_attn.o_proj.weight.detach() for l in range(NL)]  # (d, H·D)

    for wi, w in enumerate(wins):
        model(w[None])
        same = (w[:, None] == w[None, :]) & (dist > 0)
        ind = torch.zeros(T, T, dtype=torch.bool)
        ind[:, 1:] = (w[:, None] == w[None, :-1]) & (dist[:, 1:] > 0)  # ключ j: x_{j-1} = x_i, j < i
        has_s, has_i = same.any(1), ind.any(1)
        for l in range(NL):
            L = layers[l]
            att = L.self_attn
            xn = cap[("xn", l)]
            q, k, v = project(att, xn)
            sl = slice(wi * T, (wi + 1) * T)
            Q[l][sl], K[l][sl], Vv[l][sl] = q, k, v
            zq = cplx(q) * rot_t[:, None, :]
            zk = cplx(k) * rot_t[:, None, :]
            outs = []
            for h in range(H):
                s = (realvec(zq[:, h]) @ realvec(zk[:, h // rep]).T) * float(att.scaling)
                pr = torch.softmax(s.masked_fill(~causal, float("-inf")), -1).float()
                outs.append(pr @ v[:, h // rep])
                e = emp[l][h]
                e["mass"] += (bm * pr[None]).sum(-1)[:, valid].mean(-1)
                e["pos0"] += float(pr[valid, 0].mean())
                e["ent"] += float(torch.exp(-(pr.clamp_min(1e-30).log() * pr).sum(-1))[valid].mean())
                if has_s.any():
                    e["dup"] += float((pr * same).sum(-1)[has_s].sum())
                    e["dup_base"] += float((same.sum(-1).float() / (it + 1).float())[has_s].sum())
                    e["dup_n"] += int(has_s.sum())
                if has_i.any():
                    e["ind"] += float((pr * ind).sum(-1)[has_i].sum())
                    e["ind_base"] += float((ind.sum(-1).float() / (it + 1).float())[has_i].sum())
                    e["ind_n"] += int(has_i.sum())
            mine = torch.cat(outs, -1) @ Wo[l].T  # не через модуль: его перехват затёр бы вход o_proj
            recon_err[l] = max(recon_err[l], float((mine - cap[("attn", l)]).abs().max() / cap[("attn", l)].abs().max()))
            # ---- точное разложение q и k по источникам
            r = cap["emb"].clone()
            for j in range(l):
                r += cap[("attn", j)] + cap[("mlp", j)]
            g_in = L.input_layernorm.weight.detach()
            inv_r = ((xn / g_in) * r).sum(-1) / (r * r).sum(-1)  # (T,) = 1/rms_r
            rms_res[l] += float((r * r).mean(-1).sqrt().mean())
            Wq, Wk = att.q_proj.weight.detach(), att.k_proj.weight.detach()
            q_raw = (xn @ Wq.T).view(T, H, D)
            k_raw = (xn @ Wk.T).view(T, G, D)
            gq, gk = att.q_norm.weight.detach(), att.k_norm.weight.detach()
            inv_q = ((q / gq) * q_raw).sum(-1) / (q_raw * q_raw).sum(-1)  # (T,H)
            inv_k = ((k / gk) * k_raw).sum(-1) / (k_raw * k_raw).sum(-1)  # (T,G)
            Aq = (Wq * g_in[None, :]).T  # (d, H·D): остаточный поток → q до QK-norm
            Ak = (Wk * g_in[None, :]).T

            def put(ci, cq_raw, ck_raw):  # вклад компоненты до нормировок (T,H·D), (T,G·D)
                accq[l][ci] += ((cq_raw.view(T, H, D) * inv_r[:, None, None] * inv_q[:, :, None]) * gq).double().mean(0)
                acck[l][ci] += ((ck_raw.view(T, G, D) * inv_r[:, None, None] * inv_k[:, :, None]) * gk).double().mean(0)

            put(0, cap["emb"] @ Aq, cap["emb"] @ Ak)
            ci = 1
            for j in range(l):
                Bq, Bk = Wo[j].T @ Aq, Wo[j].T @ Ak  # (H·D, H·D), (H·D, G·D)
                oin = cap[("oin", j)]
                for hh in range(H):
                    z = oin[:, hh * D:(hh + 1) * D]
                    put(ci, z @ Bq[hh * D:(hh + 1) * D], z @ Bk[hh * D:(hh + 1) * D])
                    ci += 1
                put(ci, cap[("mlp", j)] @ Aq, cap[("mlp", j)] @ Ak)
                ci += 1
            attr_err[l] = max(attr_err[l], float((accq[l].sum(0) / (wi + 1) - Q[l][: (wi + 1) * T].double().mean(0)).abs().max()))
        print(f"[{args.model}] окно {wi + 1}/{nW}: {time.time() - t0:.0f}s", flush=True)
    for h_ in hooks:
        h_.remove()

    # ---- токенные средние (по выборке позиций)
    cnt = torch.bincount(ids_all, minlength=model.config.vocab_size)
    uniq = (cnt >= args.token_min).nonzero().flatten()
    remap = torch.full((model.config.vocab_size,), -1, dtype=torch.long)
    remap[uniq] = torch.arange(uniq.numel())
    rows = remap[ids_all]
    okr = rows >= 0
    pt = cnt[uniq].double() / cnt[uniq].sum()
    listed = cnt[uniq] >= args.list_min

    def tok_mean(x):  # (S, ...) → (U, ...)
        flat = x[okr].double().flatten(1)
        out = torch.zeros(uniq.numel(), flat.shape[1], dtype=torch.float64).index_add_(0, rows[okr], flat)
        return (out / cnt[uniq].double()[:, None]).view(uniq.numel(), *x.shape[1:])

    def comp_name(l, ci):
        if ci == 0:
            return "эмб"
        j, rr = divmod(ci - 1, H + 1)
        return f"MLP{j}" if rr == H else f"L{j}.H{rr}"

    res = {"model": args.model, "H": H, "G": G, "D": D, "d": d, "layers": NL, "text_tokens": int(S),
           "rope_period_tokens": period.tolist(), "recon_err_max": float(recon_err.max()), "attr_err_max": float(attr_err.max()),
           "norm_weights": norm_weights(model, theta, NL), "heads": []}
    rot_all = torch.exp(1j * torch.arange(args.ctx, dtype=torch.float64)[:, None] * theta[None, :])
    rot_d = torch.exp(1j * torch.tensor(DELTAS, dtype=torch.float64)[:, None] * theta[None, :])
    slow = theta * args.ctx < 0.5
    for l in range(NL):
        att = layers[l].self_attn
        scale = float(att.scaling)
        zq, zk = cplx(Q[l]), cplx(K[l])
        zqm, zkm = zq.mean(0), zk.mean(0)
        zqt, zkt = zq - zqm, zk - zkm
        tq, tk = tok_mean(Q[l]), tok_mean(K[l])  # токенные средние
        ztq, ztk = cplx(tq) - zqm, cplx(tk) - zkm
        rr = float(rms_res[l] / nW)
        for h in range(H):
            g = h // rep
            cm = zqm[h] * zkm[g].conj()
            P = scale * (cm[None, :] * rot_all).real.sum(-1)
            amp, psi = scale * cm.abs(), torch.angle(cm)
            peak = ((-psi) % (2 * math.pi)) / theta
            top = amp.topk(5).indices
            m0 = int(top[0])
            ramp512 = float(amp[m0] * (math.cos(theta[m0] * 512 + psi[m0]) - math.cos(psi[m0])))
            sm = torch.softmax(P[:256], 0)
            kern = {"P": {str(dv): float(P[dv]) for dv in [0, 1, 2, 4, 8, 16, 64, 256, 512] if dv < args.ctx},
                    "argmax_delta": int(P.argmax()), "range_logits": float(P.max() - P.min()),
                    "top_freqs": [{"m": int(m), "amp": float(amp[m]), "period": float(period[m]), "peak_delta": float(peak[m])} for m in top],
                    "dominant_freq_change_over_512": ramp512,
                    "softmax_mass_T256": {n: float(sm[a:min(b, 255) + 1].sum()) for n, (a, b) in zip(BNAMES, BUCKETS)}}
            b_xd = scale * torch.einsum("m,nm,dm->nd", zqm[h], zkt[:, g].conj(), rot_d).real
            c_xd = scale * torch.einsum("nm,m,dm->nd", zqt[:, h], zkm[g].conj(), rot_d).real
            bk = realvec(zkt[:, g])
            Cb = bk.T @ bk / bk.shape[0]
            sdp = []
            for di in range(len(DELTAS)):
                a = realvec(zqt[:, h] * rot_d[di][None, :])
                sdp.append(scale * float(torch.trace((a.T @ a / a.shape[0]) @ Cb).clamp_min(0).sqrt()))
            # тот же токен: по токенным средним, Δ-независимая (медленная) и полная части при Δ=64
            a_t = realvec(ztq[:, h] * rot_d[DELTAS.index(64)][None, :])
            b_t = realvec(ztk[:, g])
            diag = float((pt * (a_t * b_t).sum(-1)).sum())
            sd_t = float(torch.trace(torch.einsum("n,ni,nj->ij", pt, a_t, a_t) @ torch.einsum("n,ni,nj->ij", pt, b_t, b_t)).clamp_min(0).sqrt())
            r2q = float((pt[:, None] * (ztq[:, h].abs() ** 2)).sum() / (zqt[:, h].abs() ** 2).mean(0).sum().clamp_min(1e-30))
            r2k = float((pt[:, None] * (ztk[:, g].abs() ** 2)).sum() / (zkt[:, g].abs() ** 2).mean(0).sum().clamp_min(1e-30))
            terms = {"position": float(P[1:].std()), "key_salience": float((b_xd ** 2).mean(0).sqrt().mean()),
                     "query_modulation": float((c_xd ** 2).mean(0).sqrt().mean()), "pair": float(sum(sdp) / len(sdp)),
                     "token_r2_q": min(r2q, 1.0), "token_r2_k": min(r2k, 1.0), "same_token_z": diag / max(sd_t, 1e-12)}
            # заметность ключей по токенам (все частоты, Δ=64)
            sal_tok = scale * (zqm[h][None, :] * ztk[:, g].conj() * rot_d[DELTAS.index(64)][None, :]).real.sum(-1)
            sf = sal_tok.masked_fill(~listed, float("nan"))
            hi, lo = torch.argsort(sf.nan_to_num(-1e9), descending=True), torch.argsort(sf.nan_to_num(1e9))
            sal = {"std_logits": float(((sal_tok ** 2) * pt).sum().sqrt()), "top": toks(tok, uniq[hi], sal_tok[hi], n=10),
                   "bottom": toks(tok, uniq[lo], sal_tok[lo], n=10)}
            # OV
            Woh = Wo[l][:, h * D:(h + 1) * D].double()
            vv = Vv[l][:, g].double()
            vm = vv.mean(0)
            o = vv @ Woh.T
            om = Woh @ vm
            Cv = (vv - vm).T @ (vv - vm) / vv.shape[0]
            ev = torch.linalg.svdvals(torch.linalg.cholesky(Cv + 1e-9 * Cv.diagonal().mean() * torch.eye(D, dtype=torch.float64)).T @ Woh.T) ** 2
            o_rms = float((o ** 2).sum(-1).mean().sqrt())
            td = om.abs().topk(4).indices
            ov = {"rms_out_over_rms_resid": o_rms / (rr * math.sqrt(d)), "const_share": float((om ** 2).sum() / o_rms ** 2),
                  "eff_rank": float(ev.sum() ** 2 / (ev ** 2).sum().clamp_min(1e-30)), "const_top_dims": td.tolist(),
                  "const_top_dims_share": float((om[td] ** 2).sum() / (om ** 2).sum().clamp_min(1e-30))}
            # текст
            e = emp[l][h]
            text = {"mass_by_distance": dict(zip(BNAMES, (e["mass"] / nW).tolist())), "pos0": e["pos0"] / nW, "effective_keys": e["ent"] / nW,
                    "dup_mass": e["dup"] / max(e["dup_n"], 1), "dup_baseline": e["dup_base"] / max(e["dup_n"], 1),
                    "ind_mass": e["ind"] / max(e["ind_n"], 1), "ind_baseline": e["ind_base"] / max(e["ind_n"], 1)}
            # источники главных частот
            src = []
            for m in top[:3].tolist():
                row = {"m": m, "period": float(period[m]), "amp": float(amp[m])}
                for nm, acc, hh, ztot in (("q", accq[l], h, zqm[h][m]), ("k", acck[l], g, zkm[g][m])):
                    zc = torch.complex(acc[:, hh, m], acc[:, hh, m + M]) / nW  # (ncomp,)
                    share = (zc * ztot.conj()).real / (ztot.abs() ** 2).clamp_min(1e-30)
                    agg = {"эмб": float(share[0]), "внимание": 0.0, "MLP": 0.0}
                    for ci in range(1, share.numel()):
                        agg["MLP" if (ci - 1) % (H + 1) == H else "внимание"] += float(share[ci])
                    o5 = share.abs().topk(min(5, share.numel())).indices
                    row[nm] = {"abs": float(ztot.abs()), "agg": agg, "top": [(comp_name(l, int(ci)), float(share[ci])) for ci in o5]}
                src.append(row)
            hd = {"layer": l, "head": h, "kv_group": g, "label": label(text, kern, terms), "kernel": kern, "terms_std_logits": terms,
                  "salience": sal, "ov": ov, "text": text, "sources": src}
            res["heads"].append(hd)
        print(f"[{args.model}] слой {l:2d}: " + "; ".join(f"{h}:{res['heads'][-H + h]['label']}" for h in range(H)), flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    out = ROOT / "results" / args.model / "heads"
    out.mkdir(parents=True, exist_ok=True)
    (out / "all_layers.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    (out / "all_layers.md").write_text(report(res))
    print(f"ошибка пересчёта внимания {res['recon_err_max']:.1e}, разложения по источникам {res['attr_err_max']:.1e}; "
          f"сохранено {out / 'all_layers.json'} и all_layers.md; {res['seconds']}s")


def fsrc(s):
    a = s["agg"]
    return (f"эмб {a['эмб']:+.2f}, внимание {a['внимание']:+.2f}, MLP {a['MLP']:+.2f}; главное: " +
            ", ".join(f"{n} {v:+.2f}" for n, v in s["top"][:4]))


def report(res):
    H = res["H"]
    L = [f"# Головы всех слоёв: {res['model']}", "",
         f"{res['layers']} слоёв × {H} голов; текст: {res['text_tokens']} токенов WikiText-2 (тест). Внимание пересчитано из q, k "
         f"(отн. ошибка ≤ {res['recon_err_max']:.1e}); разложение q, k по источникам точное (ошибка ≤ {res['attr_err_max']:.1e}).", "",
         "Столбцы: масса внимания по расстоянию Δ; «поз.0» — на первый токен; «повтор» и «индукция» — масса / масса при равномерном "
         "внимании; «спад» — изменение логита главной частоты на 512 токенах; σ — разброс слагаемых логита; R² — доля q/k, объяснимая токеном.", ""]
    for l in range(res["layers"]):
        hs = [x for x in res["heads"] if x["layer"] == l]
        nw = res["norm_weights"][l]
        L += [f"## Слой {l}", "",
              f"k_norm: макс {nw['k_norm']['top_vals'][0]:.1f} (dim {nw['k_norm']['top_dims'][0]}, m={nw['k_norm']['top_freq'][0]}, {nw['k_norm']['top_part'][0]}), "
              f"медиана {nw['k_norm']['median_abs']:.1f}; q_norm: макс {nw['q_norm']['top_vals'][0]:.1f} (dim {nw['q_norm']['top_dims'][0]}, "
              f"m={nw['q_norm']['top_freq'][0]}, {nw['q_norm']['top_part'][0]}), медиана {nw['q_norm']['median_abs']:.1f}.", "",
              "| h | метка | Δ=0 | 1 | 2 | 3–8 | 9–64 | 65+ | поз.0 | повтор | индукция | ключей | главная частота (период, ампл.) | спад | σ поз/ключ/запрос/пара | R² q/k | OV пост./ранг |",
              "|" + "---|" * 17]
        for x in hs:
            t, k, s, o = x["text"], x["kernel"], x["terms_std_logits"], x["ov"]
            m = t["mass_by_distance"]
            f = k["top_freqs"][0]
            L.append(f"| {x['head']} | {x['label']} | {m['0']:.2f} | {m['1']:.2f} | {m['2']:.2f} | {m['3-4'] + m['5-8']:.2f} | {m['9-16'] + m['17-32'] + m['33-64']:.2f} "
                     f"| {m['65+']:.2f} | {t['pos0']:.2f} | {t['dup_mass']:.2f}/{t['dup_baseline']:.2f} | {t['ind_mass']:.2f}/{t['ind_baseline']:.2f} | {t['effective_keys']:.0f} "
                     f"| m={f['m']} ({f['period']:.0f}, {f['amp']:.1f}) | {k['dominant_freq_change_over_512']:+.1f} "
                     f"| {s['position']:.1f}/{s['key_salience']:.1f}/{s['query_modulation']:.1f}/{s['pair']:.1f} | {s['token_r2_q']:.2f}/{s['token_r2_k']:.2f} "
                     f"| {o['const_share']:.2f}/{o['eff_rank']:.1f} |")
        L.append("")
        for x in hs:
            k = x["kernel"]
            L += [f"**L{l}.H{x['head']} — {x['label']}.** Ядро: " + ", ".join(f"Δ={a}: {b:+.1f}" for a, b in k["P"].items()) +
                  "; частоты: " + "; ".join(f"m={f['m']} (период {f['period']:.0f}, {f['amp']:.1f}, пик Δ*={f['peak_delta']:.1f})" for f in k["top_freqs"][:4]) + ".",
                  f"- заметность ключей (σ {x['salience']['std_logits']:.1f}): + {', '.join(x['salience']['top'][:8])} / − {', '.join(x['salience']['bottom'][:8])}"]
            for s in x["sources"][:2]:
                L += [f"- источники частоты m={s['m']} (период {s['period']:.0f}, {s['amp']:.1f} логита): запрос |z|={s['q']['abs']:.1f}: {fsrc(s['q'])}",
                      f"  ключ |z|={s['k']['abs']:.1f}: {fsrc(s['k'])}"]
            L.append("")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pocket0.3b")
    ap.add_argument("--layers", type=int, default=0, help="сколько первых слоёв (0 — все)")
    ap.add_argument("--text-windows", type=int, default=16)
    ap.add_argument("--text-len", type=int, default=512)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--token-min", type=int, default=2, help="токен входит в токенные средние, если встретился столько раз")
    ap.add_argument("--list-min", type=int, default=8, help="токен показывается в списках, если встретился столько раз")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
