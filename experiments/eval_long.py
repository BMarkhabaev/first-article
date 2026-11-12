#!/usr/bin/env python
"""Окно + сток: число стоков и длина контекста (решение пользователя 2026-11-08).

  --study sinks   : T=1024 (и 2048), первые k слоёв, окно 64; стоки 1, 2, 4 и «1 сток + все переводы строки»
  --study length  : k=3, окно 64, 1 и 4 стока; T = 1024, 2048, 4096, 8192 при фиксированном бюджете токенов
Ленточная реализация (память O(T·w)); логиты учителя и ученика считаются кусками по 1024 позиции.
Результат: results/<model>/long/<study>_k<k>.json."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_time import BandedAttention  # noqa: E402
from detlayers.attention import DetAttention, sparse_pattern  # noqa: E402
from detlayers.common import load_model, pick_device, save_json, seed_all, wikitext_windows  # noqa: E402
from detlayers.methods import InputIdsTap, Replacement  # noqa: E402

ROOT = Path(__file__).resolve().parent


class NewlineSinkAttention(DetAttention):
    """Окно + n_sink первых токенов + все токены-переводы строки (плотная маска, T ≤ 2048)."""

    def __init__(self, orig, window, n_sink, newline_ids, tap):
        super().__init__(orig)
        self.window, self.n_sink, self.tap = window, n_sink, tap
        self.register_buffer("nl", torch.tensor(sorted(newline_ids)))

    def mix(self, q, k, v):
        T = q.shape[-2]
        allowed = sparse_pattern(T, self.window, self.n_sink, None, q.device)  # (T,T)
        ids = self.tap.ids[:, :T].to(q.device)
        is_nl = torch.isin(ids, self.nl.to(q.device))  # (B,T)
        causal = torch.ones(T, T, device=q.device, dtype=torch.bool).tril()
        allowed = allowed[None] | (is_nl[:, None, :] & causal[None])  # (B,T,T)
        s = torch.matmul(q.float(), k.float().transpose(-1, -2)) * self.scaling
        a = torch.softmax(s.masked_fill(~allowed[:, None], float("-inf")), dim=-1)
        return torch.matmul(a, v.float())


@torch.no_grad()
def eval_pair(model, enable, disable, ids, chunk=1024):
    disable()
    hT = model.model(ids).last_hidden_state
    enable()
    hS = model.model(ids).last_hidden_state
    disable()
    tgt = ids[:, 1:]
    nll_t = nll_s = kl = agree = 0.0
    n = tgt.numel()
    for s in range(0, ids.shape[1] - 1, chunk):
        e = min(ids.shape[1] - 1, s + chunk)
        lpt = F.log_softmax(model.lm_head(hT[:, s:e]).float(), -1)
        lps = F.log_softmax(model.lm_head(hS[:, s:e]).float(), -1)
        tg = tgt[:, s:e]
        nll_t += -lpt.gather(-1, tg[..., None]).sum().item()
        nll_s += -lps.gather(-1, tg[..., None]).sum().item()
        kl += (lpt.exp() * (lpt - lps)).sum().item()
        agree += (lpt.argmax(-1) == lps.argmax(-1)).sum().item()
    return {"nll_t": nll_t, "nll_s": nll_s, "kl": kl, "agree": agree, "n": n}


def summarize(acc):
    n = acc["n"]
    pt, ps = math.exp(acc["nll_t"] / n), math.exp(acc["nll_s"] / n)
    return {"ppl_teacher": pt, "ppl": ps, "dppl_pct": 100 * (ps / pt - 1), "kl": acc["kl"] / n,
            "top1_agree": acc["agree"] / n, "tokens": n}


def newline_ids(tok):
    out = []
    for i in range(len(tok)):
        d = tok.decode([i])
        if d and set(d) <= {"\n", "\r"}:
            out.append(i)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="0.6b")
    ap.add_argument("--study", choices=["sinks", "length"], default="sinks")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--token-budget", type=int, default=65_536)
    ap.add_argument("--dtype", default="fp32")
    args = ap.parse_args()
    seed_all(0)
    dev = pick_device(None)
    model, tok = load_model(args.model, dev, args.dtype)
    layers = model.model.layers
    out = ROOT / "results" / args.model / "long" / f"{args.study}_k{args.k}.json"
    res = {"model": args.model, "study": args.study, "k": args.k, "window": args.window, "dtype": args.dtype,
           "token_budget": args.token_budget, "results": {}}
    tap = InputIdsTap(model)
    nl = newline_ids(tok)
    res["newline_ids"] = nl

    def variants(T):
        v = {}
        for ns in (1, 2, 4):
            v[f"band_sink{ns}"] = lambda L, ns=ns: BandedAttention(L.self_attn, args.window, ns)
        if args.study == "sinks" and T <= 2048:
            v["dense_sink1+newlines"] = lambda L: NewlineSinkAttention(L.self_attn, args.window, 1, nl, tap)
        if args.study == "length":
            v = {k_: f for k_, f in v.items() if k_ in ("band_sink1", "band_sink4")}
        return v

    lengths = [1024, 2048] if args.study == "sinks" else [1024, 2048, 4096, 8192]
    for T in lengths:
        nwin = max(1, args.token_budget // T)
        wins = wikitext_windows(tok, "wikitext-2-raw-v1", "test", T, max_windows=nwin)
        vs = variants(T)
        accs = {name: {"nll_t": 0.0, "nll_s": 0.0, "kl": 0.0, "agree": 0.0, "n": 0} for name in vs}
        t0 = time.time()
        try:
            for s in range(wins.shape[0]):
                ids = wins[s:s + 1].to(dev)
                for name, mk in vs.items():
                    rep = Replacement(model)
                    for i in range(args.k):
                        rep.add(i, "self_attn", mk(layers[i]))
                    r = eval_pair(model, rep.enable, rep.disable, ids)
                    for a in accs[name]:
                        accs[name][a] += r[a]
            res["results"][T] = {name: summarize(a) for name, a in accs.items()}
            res["results"][T]["windows"] = int(wins.shape[0])
            res["results"][T]["seconds"] = round(time.time() - t0, 1)
        except RuntimeError as e:  # нехватка памяти на длинном контексте
            res["results"][T] = {"error": str(e)[:200]}
        print(T, json.dumps({n_: ({a: round(b, 4) for a, b in v_.items()} if isinstance(v_, dict) else v_)
                             for n_, v_ in res["results"][T].items()}, ensure_ascii=False), flush=True)
        save_json(res, out)
    print("saved", out)


if __name__ == "__main__":
    main()
