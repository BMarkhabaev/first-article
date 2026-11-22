"""Откуда берётся выбор головы: точное разложение преимущества «цели» по источникам запроса и ключа.

Для вида головы задаётся цель T(i) запроса i:
  self — сам токен (j = i);  prev — предыдущий (j = i−1);  same — прошлые вхождения того же токена;
  ind  — индукционные цели (x_{j−1} = x_i);  sink — позиция 0.
Преимущество цели A = среднее по запросам [средний логит на целях − средний логит на остальных причинных ключах].
Логит s(i,j) = scale · Re Σ_m z^q(i) conj(z^k(j)) e^{iθ_m(i−j)} линеен и по ключу, и по запросу, а ключ и запрос
точно раскладываются по источникам остаточного потока (эмбеддинг, каждая предыдущая голова, каждый предыдущий MLP;
см. head_mechanisms_deep.py). Поэтому A = Σ_c A^k_c (по источникам ключа) = Σ_c A^q_c (по источникам запроса) —
без приближений. Доли A^k_c / A и A^q_c / A показывают, кто создаёт выбор: например, у индукционной головы ключ
должен приходить из голов «предыдущий токен», а запрос — из самого токена.

Дополнительно:
  — косинус между запросом и ключом одной позиции и разных позиций (для «сам токен»);
  — остаточный поток на позиции 0: во сколько раз его норма больше обычной, в каких координатах и кто их пишет (для стока).

  python head_circuits.py --model pocket0.3b
Результат: results/<m>/heads/circuits.json
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
from head_mechanisms import cplx  # noqa: E402
from table_attention import dims, project  # noqa: E402

TYPES = ["self", "prev", "same", "ind", "sink"]


@torch.no_grad()
def run(args):
    t0 = time.time()
    model, tok = load_model(args.model, torch.device("cpu"), "fp32")
    H, G, D = dims(model)
    M, rep = D // 2, H // G
    layers = model.model.layers
    NL = len(layers)
    d = model.config.hidden_size
    theta = model.model.rotary_emb.inv_freq.double().cpu()
    wins = wikitext_windows(tok, "wikitext-2-raw-v1", "test", args.text_len, max_windows=args.text_windows)
    nW, T = wins.shape
    it = torch.arange(T)
    dist = it[:, None] - it[None, :]
    causal = dist >= 0
    rot = torch.exp(1j * it.double()[:, None] * theta[None, :])  # (T,M)
    g_of_h = torch.arange(H) // rep

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
    Wo = [layers[l].self_attn.o_proj.weight.detach() for l in range(NL)]

    nT = len(TYPES)
    A_tot = torch.zeros(NL, nT, H, dtype=torch.float64)
    A_k = [torch.zeros(1 + l * (H + 1), nT, H, dtype=torch.float64) for l in range(NL)]
    A_q = [torch.zeros(1 + l * (H + 1), nT, H, dtype=torch.float64) for l in range(NL)]
    cos_self = torch.zeros(NL, H, dtype=torch.float64)
    cos_other = torch.zeros(NL, H, dtype=torch.float64)
    n_valid = torch.zeros(nT)
    pos0 = [{"ratio": 0.0, "dims": torch.zeros(d, dtype=torch.float64), "src": torch.zeros(1 + l * 2, dtype=torch.float64)} for l in range(NL)]

    for wi, w in enumerate(wins):
        model(w[None])
        same = (w[:, None] == w[None, :]) & (dist > 0)
        ind = torch.zeros(T, T, dtype=torch.bool)
        ind[:, 1:] = (w[:, None] == w[None, :-1]) & (dist[:, 1:] > 0)
        tg = {"self": dist == 0, "prev": dist == 1, "same": same, "ind": ind, "sink": (it[None, :] == 0) & (dist > 0)}
        lo = {"self": 1, "prev": 2, "same": 1, "ind": 1, "sink": 64}
        Wt = torch.zeros(nT, T, T, dtype=torch.float64)
        valid = torch.zeros(nT, T, dtype=torch.bool)
        for ti, name in enumerate(TYPES):
            tgt = tg[name] & causal
            oth = causal & ~tgt
            nt, no = tgt.sum(1), oth.sum(1)
            ok = (nt > 0) & (no > 0) & (it >= lo[name])
            Wt[ti] = (tgt.double() / nt.clamp_min(1)[:, None] - oth.double() / no.clamp_min(1)[:, None]) * ok[:, None]
            valid[ti] = ok
        nv = valid.sum(1).clamp_min(1).double()  # (nT,)
        n_valid += valid.sum(1)
        Wc = Wt.to(torch.complex128)
        for l in range(NL):
            L = layers[l]
            att = L.self_attn
            scale = float(att.scaling)
            xn = cap[("xn", l)]
            q, k, _ = project(att, xn)
            r = cap["emb"].clone()
            comps = [("emb", cap["emb"])]
            for j in range(l):
                r += cap[("attn", j)] + cap[("mlp", j)]
            g_in = L.input_layernorm.weight.detach()
            inv_r = ((xn / g_in) * r).sum(-1) / (r * r).sum(-1)
            Wq, Wk = att.q_proj.weight.detach(), att.k_proj.weight.detach()
            q_raw, k_raw = (xn @ Wq.T).view(T, H, D), (xn @ Wk.T).view(T, G, D)
            gq, gk = att.q_norm.weight.detach(), att.k_norm.weight.detach()
            inv_q = ((q / gq) * q_raw).sum(-1) / (q_raw * q_raw).sum(-1)
            inv_k = ((k / gk) * k_raw).sum(-1) / (k_raw * k_raw).sum(-1)
            Aq, Ak = (Wq * g_in[None, :]).T, (Wk * g_in[None, :]).T
            zq = cplx(q) * rot[:, None, :]  # (T,H,M)
            zk = cplx(k) * rot[:, None, :]  # (T,G,M)
            kap = torch.einsum("tij,jgm->tigm", Wc, zk)  # (nT,T,G,M): цель минус остальные, полный ключ
            kap_h = kap[:, :, g_of_h, :]  # (nT,T,H,M)
            A_tot[l] += scale * (zq[None] * kap_h.conj()).real.sum(-1).sum(1) / nv[:, None] / nW

            def put(ci, cq_raw, ck_raw):
                zqc = cplx((cq_raw.view(T, H, D) * inv_r[:, None, None] * inv_q[:, :, None]) * gq) * rot[:, None, :]
                zkc = cplx((ck_raw.view(T, G, D) * inv_r[:, None, None] * inv_k[:, :, None]) * gk) * rot[:, None, :]
                A_q[l][ci] += scale * (zqc[None] * kap_h.conj()).real.sum(-1).sum(1) / nv[:, None] / nW
                kc = torch.einsum("tij,jgm->tigm", Wc, zkc)[:, :, g_of_h, :]
                A_k[l][ci] += scale * (zq[None] * kc.conj()).real.sum(-1).sum(1) / nv[:, None] / nW

            put(0, cap["emb"] @ Aq, cap["emb"] @ Ak)
            ci = 1
            for j in range(l):
                Bq, Bk = Wo[j].T @ Aq, Wo[j].T @ Ak
                oin = cap[("oin", j)]
                for hh in range(H):
                    z = oin[:, hh * D:(hh + 1) * D]
                    put(ci, z @ Bq[hh * D:(hh + 1) * D], z @ Bk[hh * D:(hh + 1) * D])
                    ci += 1
                put(ci, cap[("mlp", j)] @ Aq, cap[("mlp", j)] @ Ak)
                ci += 1
            # косинус запроса и ключа: та же позиция и соседние позиции (без поворота)
            qn = torch.nn.functional.normalize(q.double(), dim=-1)
            kn = torch.nn.functional.normalize(k.double(), dim=-1)[:, g_of_h]
            cos_self[l] += (qn * kn).sum(-1).mean(0) / nW
            cos_other[l] += (qn[8:] * kn[:-8]).sum(-1).mean(0) / nW
            # остаточный поток на позиции 0
            rms = (r * r).mean(-1).sqrt()
            pos0[l]["ratio"] += float(rms[0] / rms[1:].median()) / nW
            r0 = r[0].double()
            pos0[l]["dims"] += (r0 ** 2) / (r0 ** 2).sum() / nW
            top = (r0 ** 2).topk(4).indices
            den = (r0[top] ** 2).sum()
            src = [float((cap["emb"][0].double()[top] * r0[top]).sum() / den)]
            for j in range(l):
                src += [float((cap[("attn", j)][0].double()[top] * r0[top]).sum() / den),
                        float((cap[("mlp", j)][0].double()[top] * r0[top]).sum() / den)]
            pos0[l]["src"] += torch.tensor(src, dtype=torch.float64) / nW
        print(f"[{args.model}] окно {wi + 1}/{nW}: {time.time() - t0:.0f}s", flush=True)
    for h_ in hooks:
        h_.remove()

    def cname(ci):
        if ci == 0:
            return "эмб"
        j, rr = divmod(ci - 1, H + 1)
        return f"MLP{j}" if rr == H else f"L{j}.H{rr}"

    def summarize(vec, total):
        share = vec / total if abs(total) > 1e-9 else vec * 0
        agg = {"эмб": float(share[0]), "внимание": 0.0, "MLP": 0.0}
        for ci in range(1, share.numel()):
            agg["MLP" if (ci - 1) % (H + 1) == H else "внимание"] += float(share[ci])
        o = share.abs().topk(min(6, share.numel())).indices
        return {"agg": agg, "top": [(cname(int(c)), float(share[c])) for c in o]}

    err = 0.0
    out = {"model": args.model, "H": H, "layers": NL, "types": TYPES, "text_tokens": int(nW * T),
           "queries_per_type": dict(zip(TYPES, n_valid.tolist())), "heads": [], "pos0": []}
    for l in range(NL):
        err = max(err, float((A_k[l].sum(0) - A_tot[l]).abs().max()), float((A_q[l].sum(0) - A_tot[l]).abs().max()))
        for h in range(H):
            row = {"layer": l, "head": h, "cos_qk_same_pos": float(cos_self[l, h]), "cos_qk_other_pos": float(cos_other[l, h]), "adv": {}}
            for ti, name in enumerate(TYPES):
                tot = float(A_tot[l, ti, h])
                row["adv"][name] = {"logits": tot, "key": summarize(A_k[l][:, ti, h], tot), "query": summarize(A_q[l][:, ti, h], tot)}
            out["heads"].append(row)
        p0 = pos0[l]
        td = p0["dims"].topk(4)
        src = p0["src"]
        names = ["эмб"] + [n for j in range(l) for n in (f"внимание{j}", f"MLP{j}")]
        o = src.abs().topk(min(4, src.numel())).indices
        out["pos0"].append({"layer": l, "rms_ratio": p0["ratio"], "top_dims": td.indices.tolist(), "top_dims_share": float(td.values.sum()),
                            "sources_of_top_dims": [(names[int(c)], float(src[c])) for c in o]})
    out["decomposition_err_logits"] = err
    out["seconds"] = round(time.time() - t0, 1)
    dst = ROOT / "results" / args.model / "heads"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "circuits.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"ошибка разложения {err:.1e} логита; сохранено {dst / 'circuits.json'}; {out['seconds']}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pocket0.3b")
    ap.add_argument("--text-windows", type=int, default=6)
    ap.add_argument("--text-len", type=int, default=512)
    run(ap.parse_args())
