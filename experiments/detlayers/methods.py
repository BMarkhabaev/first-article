"""Сборка методов: калибровка (замкнутая форма, без SGD) и подмена модулей в Qwen3.

Терминология: k — число первых слоёв, которые заменяются (слои 0..k-1).
* attention-методы (sparse, strided, performer, linear_elu, relpos, head_typed, a3_attn,
  ridge_conv) заменяют только подслой внимания, MLP остаётся точным;
* a3 дополнительно сжимает MLP (как в статье A3);
* block-методы (drop, ngram) заменяют блоки целиком.
"""
from __future__ import annotations

import math
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import (DetAttention, HeadTypedAttention, LinearEluAttention, PerformerAttention,
                        RelPosAttention, RelPosStats, SalienceAttention, SparseAttention,
                        WindowTailAttention, exact_attention_weights, head_fit_scores, relpos_matrix,
                        salience_matrix)
from .common import embed, run_layers

ATTN_METHODS = {"sparse", "strided", "performer", "linear_elu", "relpos", "head_typed", "a3",
                "ridge_conv", "salience", "window_tail"}
BLOCK_METHODS = {"drop", "ngram"}


# ============================================================ подмена модулей

class Replacement:
    """Набор подмен {(layer_idx, attr): new_module}; можно включать/выключать,
    чтобы в одном процессе считать и исходную модель (учитель), и изменённую."""

    def __init__(self, model):
        self.model = model
        self.new: dict[tuple[int, str], nn.Module] = {}
        self.old: dict[tuple[int, str], nn.Module] = {}
        self.active = False

    def add(self, i: int, attr: str, module: nn.Module):
        layer = self.model.model.layers[i]
        self.old[(i, attr)] = getattr(layer, attr) if attr != "block" else layer
        self.new[(i, attr)] = module

    def _set(self, which):
        for (i, attr), mod in which.items():
            if attr == "block":
                self.model.model.layers[i] = mod
            else:
                setattr(self.model.model.layers[i], attr, mod)

    def enable(self):
        if not self.active:
            self._set(self.new)
            self.active = True

    def disable(self):
        if self.active:
            self._set(self.old)
            self.active = False

    @contextmanager
    def teacher(self):
        was = self.active
        self.disable()
        try:
            yield
        finally:
            if was:
                self.enable()


# ============================================================ калибровочный проход

def _batches(ids: torch.Tensor, bs: int):
    for s in range(0, ids.shape[0], bs):
        yield ids[s:s + bs]


@torch.no_grad()
def iterate_layer_inputs(model, calib, upto: int, bs: int, device, layers=None):
    """Для каждой калибровочной партии отдаёт (i, h_i, pos, pe, input_ids) для i < upto,
    где h_i — вход i-го слоя (в исходной модели или по списку `layers`)."""
    layers = layers if layers is not None else model.model.layers
    for ids in _batches(calib, bs):
        ids = ids.to(device)
        h, pos, pe = embed(model, ids)
        for i in range(upto):
            yield i, h, pos, pe, ids
            h = run_layers(model, layers, h, pos, pe, i, i + 1)


def qkv_of(attn, x_normed, pe):
    """q, k, v исходного слоя внимания (q,k — после norm и RoPE; kv повторены по группам)."""
    return DetAttention(attn).qkv(x_normed, pe)


# ============================================================ L5/O1: статистика по сдвигам

@torch.no_grad()
def calib_relpos(model, calib, k, bs, device, max_offset):
    H = model.config.num_attention_heads
    stats = [RelPosStats(H, max_offset) for _ in range(k)]
    for i, h, pos, pe, _ in iterate_layer_inputs(model, calib, k, bs, device):
        layer = model.model.layers[i]
        q, kk, _ = qkv_of(layer.self_attn, layer.input_layernorm(h), pe)
        stats[i].update(exact_attention_weights(q, kk, layer.self_attn.scaling))
    return [s.finalize() for s in stats]


@torch.no_grad()
def calib_head_fit(model, calib, k, bs, device, kernels):
    """TV-дистанция каждой головы до её фиксированного ядра (меньше — позиционнее)."""
    acc = [torch.zeros(model.config.num_attention_heads) for _ in range(k)]
    n = 0
    for i, h, pos, pe, _ in iterate_layer_inputs(model, calib, k, bs, device):
        layer = model.model.layers[i]
        q, kk, _ = qkv_of(layer.self_attn, layer.input_layernorm(h), pe)
        kern, sink = kernels[i]
        acc[i] += head_fit_scores(q, kk, kern, sink, layer.self_attn.scaling) * q.shape[0]
        if i == 0:
            n += q.shape[0]
    return [a / n for a in acc]


# ============================================================ L6: A3 (адаптация к Qwen3)

class A3Attention(DetAttention):
    """A3 (Wong et al., 2025), перенесённый на Qwen3 (GQA + QK-norm + RoPE с парами (i, i+D/2)).
    QK: оставляем r_qk/2 пар RoPE на KV-группу по критерию sqrt(E q_i²)·sqrt(E k_i²).
    VO: совместное SVD [W_vᵀW_o,hᵀ]_{h∈группа}, взвешенное R_xx^{1/2}, ранг r_vo."""

    method = "a3"

    def __init__(self, orig, qk_idx, wv_new, wo_new):
        super().__init__(orig)
        self.register_buffer("qk_idx", qk_idx)  # (n_kv, r_qk) индексы измерений головы
        self.register_buffer("wv_new", wv_new)  # (n_kv, r_vo, d)
        self.register_buffer("wo_new", wo_new)  # (n_heads, r_vo, d)

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_values=None, **kwargs):
        q, k, _ = self.qkv(hidden_states, position_embeddings)
        idx = self.qk_idx.repeat_interleave(self.groups, dim=0)  # (H, r)
        gi = idx[None, :, None, :].expand(q.shape[0], -1, q.shape[2], -1)
        q = torch.gather(q, -1, gi)
        k = torch.gather(k, -1, gi)
        a = exact_attention_weights(q, k, self.scaling)  # масштаб исходный, как в A3
        x = hidden_states.float()
        v = torch.einsum("btd,grd->bgtr", x, self.wv_new.float())  # (B, n_kv, T, r)
        v = v.repeat_interleave(self.groups, dim=1)
        o = torch.matmul(a, v)  # (B,H,T,r)
        out = torch.einsum("bhtr,hrd->btd", o, self.wo_new.float())
        return out.to(hidden_states.dtype), None

    def extra_cost(self):
        return {"r_qk": self.qk_idx.shape[1], "r_vo": self.wv_new.shape[1]}


class A3MLP(nn.Module):
    """A3 FFN: оставляем r нейронов по критерию sqrt(E z_i²)·||w_down_i||."""

    def __init__(self, orig_mlp, keep_idx):
        super().__init__()
        self.gate = nn.Parameter(orig_mlp.gate_proj.weight[keep_idx].clone(), requires_grad=False)
        self.up = nn.Parameter(orig_mlp.up_proj.weight[keep_idx].clone(), requires_grad=False)
        self.down = nn.Parameter(orig_mlp.down_proj.weight[:, keep_idx].clone(), requires_grad=False)
        self.act = orig_mlp.act_fn

    def forward(self, x):
        return F.linear(self.act(F.linear(x, self.gate)) * F.linear(x, self.up), self.down)


def _top_right_singular(A: torch.Tensor, r: int) -> torch.Tensor:
    """Первые r правых сингулярных векторов A (строками), с проверкой результата.
    SVD в float64 из LAPACK/Accelerate на macOS иногда возвращает NaN или конечный, но
    неверный ответ. Поэтому считаем несколько кандидатов и берём тот, у которого
    векторы ортонормированы, а ошибка ||A − A VᵀV|| минимальна."""
    cands = []
    try:
        cands.append(torch.linalg.svd(A, full_matrices=False)[2][:r])
    except RuntimeError:
        pass
    try:
        cands.append(torch.linalg.svd(A.T, full_matrices=False)[0][:, :r].T)
    except RuntimeError:
        pass
    ev, U = torch.linalg.eigh(A @ A.T)  # левые векторы; V = Uᵀ A / s
    idx = ev.argsort(descending=True)[:r]
    Ur = U[:, idx]
    Vr = Ur.T @ A
    cands.append(Vr / Vr.norm(dim=1, keepdim=True).clamp_min(1e-300))
    best, best_err = None, float("inf")
    eye = torch.eye(r, dtype=A.dtype, device=A.device)
    for V in cands:
        if V.shape[0] != r or not torch.isfinite(V).all():
            continue
        if (V @ V.T - eye).abs().max() > 1e-6:
            continue
        err = (A - (A @ V.T) @ V).norm().item()
        if err < best_err:
            best, best_err = V, err
    if best is None:
        raise RuntimeError("не удалось получить устойчивое SVD")
    return best


def _sqrtm_psd(r: torch.Tensor, eps: float = 1e-10):
    evals, evecs = torch.linalg.eigh(r)
    evals = evals.clamp_min(eps * evals.max())
    return evecs, evals


@torch.no_grad()
def build_a3(model, calib, k, bs, device, ratio_qk=0.5, ratio_vo=0.5, ratio_ffn=0.5, mlp=True, robust_tau=None):
    cfg = model.config
    H, Hkv, D, d = cfg.num_attention_heads, cfg.num_key_value_heads, model.model.layers[0].self_attn.head_dim, cfg.hidden_size
    g = H // Hkv
    I = cfg.intermediate_size
    q2 = [torch.zeros(H, D, dtype=torch.float64) for _ in range(k)]
    k2 = [torch.zeros(H, D, dtype=torch.float64) for _ in range(k)]
    rxx = [torch.zeros(d, d, dtype=torch.float64) for _ in range(k)]
    z2 = [torch.zeros(I, dtype=torch.float64) for _ in range(k)]
    ntok = [0] * k
    from .mlp import find_sink_neurons
    sink = find_sink_neurons(model, calib, k, bs, device) if mlp else {}
    for i, h, pos, pe, _ in iterate_layer_inputs(model, calib, k, bs, device):
        layer = model.model.layers[i]
        x = layer.input_layernorm(h)
        q, kk, _ = qkv_of(layer.self_attn, x, pe)
        q2[i] += (q.float() ** 2).sum(dim=(0, 2)).cpu().double()
        k2[i] += (kk.float() ** 2).sum(dim=(0, 2)).cpu().double()
        xf = x.float().reshape(-1, d)
        rxx[i] += (xf.T @ xf).cpu().double()
        # вход MLP — после внимания исходного слоя
        a_out, _ = layer.self_attn(x, position_embeddings=pe, attention_mask=None)
        hm = layer.post_attention_layernorm(h + a_out)
        mm = layer.mlp
        z = mm.act_fn(mm.gate_proj(hm)) * mm.up_proj(hm)
        if robust_tau:  # робастный отбор нейронов: без позиции 0 и токенов с массивной активацией
            from .mlp import robust_rows_z
            z = z.float() * robust_rows_z(z, robust_tau)[..., None]
        z2[i] += (z.float() ** 2).sum(dim=(0, 1)).cpu().double()
        ntok[i] += xf.shape[0]

    rep = Replacement(model)
    info = {}
    r_qk = max(2, int(D * ratio_qk) // 2 * 2)
    r_vo = max(1, int(D * ratio_vo))
    r_ffn = max(1, int(I * ratio_ffn))
    for i in range(k):
        layer = model.model.layers[i]
        attn = layer.self_attn
        n = ntok[i]
        # --- QK: пары (j, j+D/2), оценка на KV-группу (сумма по q-головам группы)
        sq = (q2[i] / n).sqrt().view(Hkv, g, D)  # (Hkv, g, D)
        sk = (k2[i] / n).sqrt().view(Hkv, g, D)[:, 0]  # k одинаков в группе
        score = (sq * sk[:, None, :]).sum(1)  # (Hkv, D)
        pair = score[:, : D // 2] + score[:, D // 2:]
        top = pair.argsort(dim=-1, descending=True)[:, : r_qk // 2].sort(dim=-1).values
        qk_idx = torch.cat([top, top + D // 2], dim=-1)  # (Hkv, r_qk)
        # --- VO
        R = (rxx[i] / n).to(torch.float64)
        U, S = _sqrtm_psd(R)
        Rh = (U * S.sqrt()) @ U.T
        Wv = attn.v_proj.weight.cpu().double().view(Hkv, D, d)
        Wo = attn.o_proj.weight.cpu().double().T.reshape(H, D, d)  # Wo_h^T: (D, d)
        wv_new = torch.zeros(Hkv, r_vo, d, dtype=torch.float64)
        wo_new = torch.zeros(H, r_vo, d, dtype=torch.float64)
        for gidx in range(Hkv):
            heads = range(gidx * g, (gidx + 1) * g)
            M = torch.cat([Wv[gidx].T @ Wo[h] for h in heads], dim=1)  # (d, g*d)
            # A3: усечённое SVD R^{1/2}M = U S Vᵀ. Без R^{-1/2}: W_v'ᵀ = M V_r, W_o' = V_rᵀ,
            # произведение M V_r V_rᵀ совпадает с решением A3 при обратимой R.
            Vr = _top_right_singular(Rh @ M, r_vo)  # (r, g*d)
            wv_new[gidx] = (M @ Vr.T).T  # (r, d)
            Vr = Vr.view(r_vo, g, d)
            for j, h in enumerate(heads):
                wo_new[h] = Vr[:, j]
        dt = attn.o_proj.weight.dtype
        dev = attn.o_proj.weight.device
        rep.add(i, "self_attn", A3Attention(attn, qk_idx.to(dev), wv_new.to(dev, torch.float32),
                                            wo_new.to(dev, torch.float32)))
        if mlp:
            wd = layer.mlp.down_proj.weight.cpu().double()  # (d, I)
            sc = (z2[i] / n).sqrt() * wd.norm(dim=0)
            keep = sc.argsort(descending=True)[:r_ffn].sort().values.to(dev)
            if i in sink:  # нейроны стока считаются точно (правило 2)
                keep = torch.unique(torch.cat([keep, sink[i]]))
            from .mlp import ExactPos0  # позиция 0 (сток) считается точно, как у всех MLP-замен в работе:
            rep.add(i, "mlp", ExactPos0(layer.mlp, A3MLP(layer.mlp, keep).to(dev)))  # иначе робастный отбор
            # нейронов (без позиции 0 в ∑z²) выбрасывает нейроны, создающие массивную активацию, и сток исчезает
        info[f"layer{i}"] = {"r_qk": r_qk, "r_vo": r_vo, "r_ffn": r_ffn if mlp else I}
    _ = dt
    return rep, info


# ============================================================ O2: ridge-свёртка

class RidgeConvAttention(nn.Module):
    """O2 (наш): выход подслоя внимания ≈ Σ_{δ<w} W_δ·P x̂_{t-δ} + W_s·P x̂_0 + b.
    Позиция 0 считается точно: внимание на одном ключе = W_o W_v x̂_0."""

    def __init__(self, orig_attn, P, W, b, window):
        super().__init__()
        self.orig = orig_attn
        self.register_buffer("P", P)  # (d, p)
        self.register_buffer("W", W)  # ((w+1)*p, d)
        self.register_buffer("b", b)  # (d,)
        self.window = window

    @staticmethod
    def features(xp: torch.Tensor, window: int) -> torch.Tensor:
        """xp: (B,T,p) -> (B,T,(w+1)p): сдвиги 0..w-1 (нули за началом) и x̂_0."""
        feats = [xp]
        for dlt in range(1, window):
            feats.append(F.pad(xp, (0, 0, dlt, 0))[:, : xp.shape[1]])
        feats.append(xp[:, :1].expand_as(xp))
        return torch.cat(feats, dim=-1)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                past_key_values=None, **kwargs):
        x = hidden_states.float()
        phi = self.features(x @ self.P.float(), self.window)
        out = phi @ self.W.float() + self.b.float()
        o = self.orig
        v0 = o.v_proj(hidden_states[:, :1])  # (B,1,n_kv*D)
        v0 = v0.view(v0.shape[0], 1, -1, o.head_dim).repeat_interleave(o.num_key_value_groups, dim=2)
        out0 = o.o_proj(v0.reshape(v0.shape[0], 1, -1)).float()
        out = torch.cat([out0, out[:, 1:]], dim=1)
        return out.to(hidden_states.dtype), None


@torch.no_grad()
def build_ridge_conv(model, calib, k, bs, device, window=8, pca_dim=256, lam=1e-2):
    """Последовательная подгонка слоёв 0..k-1. Признаки берутся по пути «ученика»
    (с уже заменёнными слоями), цель — остаточный поток учителя после внимания:
    y = (h_T + a_T) − h_S, т.е. замена компенсирует накопленный дрейф."""
    d = model.config.hidden_size
    p = min(pca_dim, d)
    rep = Replacement(model)
    info = {}
    student_layers = list(model.model.layers)
    for i in range(k):
        layer = model.model.layers[i]
        # PCA x̂ учителя
        C = torch.zeros(d, d, dtype=torch.float64)
        for ids in _batches(calib, bs):
            ids = ids.to(device)
            h, pos, pe = embed(model, ids)
            h = run_layers(model, model.model.layers, h, pos, pe, 0, i)
            x = layer.input_layernorm(h).float().reshape(-1, d)
            C += (x.T @ x).cpu().double()
        evals, evecs = torch.linalg.eigh(C)
        P = evecs[:, -p:].flip(-1).float().to(device)
        Fdim = (window + 1) * p + 1
        G = torch.zeros(Fdim, Fdim, dtype=torch.float64)
        Bm = torch.zeros(Fdim, d, dtype=torch.float64)
        for ids in _batches(calib, bs):
            ids = ids.to(device)
            h0, pos, pe = embed(model, ids)
            hT = run_layers(model, model.model.layers, h0, pos, pe, 0, i)
            aT, _ = layer.self_attn(layer.input_layernorm(hT), position_embeddings=pe, attention_mask=None)
            hS = run_layers(model, student_layers, h0, pos, pe, 0, i)
            xS = layer.input_layernorm(hS).float()
            phi = RidgeConvAttention.features(xS @ P, window)[:, 1:]  # t ≥ 1
            phi = torch.cat([phi, torch.ones_like(phi[..., :1])], -1).reshape(-1, Fdim)
            y = ((hT + aT).float() - hS.float())[:, 1:].reshape(-1, d)
            G += (phi.T @ phi).cpu().double()
            Bm += (phi.T @ y).cpu().double()
        reg = lam * G.diagonal().mean()
        Wfull = torch.linalg.solve(G + reg * torch.eye(Fdim, dtype=torch.float64), Bm)
        W, b = Wfull[:-1].float().to(device), Wfull[-1].float().to(device)
        mod = RidgeConvAttention(layer.self_attn, P, W, b, window)
        rep.add(i, "self_attn", mod)
        # слой ученика i: тот же блок, но с новым вниманием
        student_layers[i] = _LayerWithAttn(layer, mod)
        info[f"layer{i}"] = {"features": Fdim, "params": int(W.numel() + P.numel())}
    return rep, info


class _LayerWithAttn(nn.Module):
    """Копия-обёртка декодерного слоя с другим подслоем внимания (только для калибровки)."""

    def __init__(self, layer, attn):
        super().__init__()
        self.layer, self.attn = layer, attn

    def forward(self, h, attention_mask=None, position_embeddings=None, position_ids=None, **kw):
        L = self.layer
        a, _ = self.attn(L.input_layernorm(h), position_embeddings=position_embeddings)
        h = h + a
        return h + L.mlp(L.post_attention_layernorm(h))


# ============================================================ B1 и O3: замена блоков

def manual_block(L, h, pe):
    """Декодерный блок через подмодули (без __call__ слоя, чтобы не попасть в хуки
    output_hidden_states transformers 5)."""
    a, _ = L.self_attn(L.input_layernorm(h), position_embeddings=pe, attention_mask=None)
    h = h + a
    return h + L.mlp(L.post_attention_layernorm(h))


class IdentityBlock(nn.Module):
    def forward(self, hidden_states, *a, **kw):
        return hidden_states


class InputIdsTap:
    """Сохраняет input_ids последнего вызова модели (нужно блочным таблицам)."""

    def __init__(self, model):
        self.ids = None
        self.handle = model.model.register_forward_pre_hook(self._hook, with_kwargs=True)

    def _hook(self, module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        self.ids = ids


class NgramBlock(nn.Module):
    """O3 (наш): выход первых k блоков на позиции t ≥ 1 ≈
        e(x_t) + Σ_{δ<m} T_δ[x_{t-δ}] + P[min(t, Pmax)].
    Позиция 0 — точное значение исходных k блоков (функция одного токена; эквивалентно
    таблице по словарю, вычисляем лениво). Остальные k-1 блоков — identity."""

    def __init__(self, model, k, tables, pos_table, vocab_map, m, tap, orig_layers):
        super().__init__()
        self.k, self.m = k, m
        self.register_buffer("tables", tables)  # (m, V'+1, d), последняя строка — неизвестный токен
        self.register_buffer("pos_table", pos_table)  # (Pmax+1, d)
        self.register_buffer("vocab_map", vocab_map)  # (V,) -> индекс строки
        self.tap = tap
        self.orig_layers = orig_layers  # исходные блоки 0..k-1 (только для позиции 0)
        self.model_ref = [model]

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None, position_ids=None, **kw):
        ids = self.tap.ids.to(hidden_states.device)
        B, T = ids.shape
        rows = self.vocab_map[ids]  # (B,T)
        out = hidden_states.float().clone()
        for dlt in range(self.m):
            r = F.pad(rows, (dlt, 0), value=self.tables.shape[1] - 1)[:, :T]
            out += self.tables[dlt][r]
        t = torch.arange(T, device=ids.device).clamp_max(self.pos_table.shape[0] - 1)
        out += self.pos_table[t]
        # точная позиция 0
        model = self.model_ref[0]
        h0, pos, pe = embed(model, ids[:, :1])
        for L in self.orig_layers:
            h0 = manual_block(L, h0, pe)
        out[:, 0] = h0[:, 0].float()
        return out.to(hidden_states.dtype)


@torch.no_grad()
def build_ngram(model, calib, k, bs, device, m=3, pos_max=16, sweeps=2, shrink=1.0):
    """Аддитивная модель по токенам окна, подгонка backfitting'ом (Гаусс–Зейдель для
    наименьших квадратов): каждая компонента = среднее остатка по её ключу
    (со сжатием: sum / (count + shrink)). Детерминированно, без SGD."""
    d = model.config.hidden_size
    orig_layers = list(model.model.layers[:k])
    uniq = torch.unique(calib)
    V = model.config.vocab_size
    vocab_map = torch.full((V,), uniq.numel(), dtype=torch.long)
    vocab_map[uniq] = torch.arange(uniq.numel())
    vocab_map = vocab_map.to(device)
    nrows = uniq.numel() + 1
    tables = torch.zeros(m, nrows, d, device=device)
    pos_table = torch.zeros(pos_max + 1, d, device=device)
    unk = nrows - 1

    def target_and_rows(ids):
        h0, pos, pe = embed(model, ids)
        hk = run_layers(model, orig_layers, h0, pos, pe, 0, k)
        delta = (hk - h0).float()  # (B,T,d)
        rows = vocab_map[ids]
        shifted = [F.pad(rows, (dl, 0), value=unk)[:, : ids.shape[1]] for dl in range(m)]
        tpos = torch.arange(ids.shape[1], device=device).clamp_max(pos_max)
        return delta, shifted, tpos

    def predict(shifted, tpos, skip=None):
        pred = 0
        for dl in range(m):
            if skip != ("tab", dl):
                pred = pred + tables[dl][shifted[dl]]
        if skip != ("pos",):
            pred = pred + pos_table[tpos]
        return pred

    comps = [("pos",)] + [("tab", dl) for dl in range(m)]
    for sweep in range(sweeps):
        for comp in comps:
            n_rows = pos_max + 1 if comp[0] == "pos" else nrows
            s = torch.zeros(n_rows, d, device=device)
            c = torch.zeros(n_rows, device=device)
            for ids in _batches(calib, bs):
                ids = ids.to(device)
                delta, shifted, tpos = target_and_rows(ids)
                resid = (delta - predict(shifted, tpos, skip=comp))[:, 1:].reshape(-1, d)
                key = (tpos.expand(ids.shape[0], -1) if comp[0] == "pos" else shifted[comp[1]])[:, 1:].reshape(-1)
                s.index_add_(0, key, resid)
                c.index_add_(0, key, torch.ones_like(key, dtype=torch.float))
            val = s / (c[:, None] + shrink)
            if comp[0] == "pos":
                pos_table.copy_(val)
            else:
                tables[comp[1]].copy_(val)
                tables[comp[1]][unk] = 0  # неизвестный/за началом — без вклада
    tap = InputIdsTap(model)
    rep = Replacement(model)
    rep.add(0, "block", NgramBlock(model, k, tables, pos_table, vocab_map, m, tap, orig_layers))
    for i in range(1, k):
        rep.add(i, "block", IdentityBlock())
    info = {"m": m, "rows": nrows, "table_mb": round(tables.numel() * 2 / 2 ** 20, 1)}
    return rep, info


# ============================================================ O4: ядро × значимость

@torch.no_grad()
def build_salience(model, calib, k, bs, device, max_offset=64, iters=3, alpha=1.0, exact_frac=0.0,
                   window=None):
    """β_h[v] подгоняется итеративно (IPF / согласование маргиналов):
        β ← β + log((A_true[v] + α) / (A_model[v] + α)),
    где A[v] — суммарная масса внимания, полученная ключами-токенами v (позиции j ≥ 1).
    Головы с наибольшей остаточной TV-дистанцией (доля exact_frac) остаются точными."""
    H = model.config.num_attention_heads
    kernels = calib_relpos(model, calib, k, bs, device, max_offset)
    uniq = torch.unique(calib)
    V = model.config.vocab_size
    vocab_map = torch.full((V,), uniq.numel(), dtype=torch.long)
    vocab_map[uniq] = torch.arange(uniq.numel())
    vocab_map = vocab_map.to(device)
    nrows = uniq.numel() + 1
    betas = [torch.zeros(H, nrows, device=device) for _ in range(k)]
    tvs = [torch.zeros(H) for _ in range(k)]
    for it in range(iters + 1):
        tgt = [torch.zeros(H, nrows, device=device) for _ in range(k)]
        mod = [torch.zeros(H, nrows, device=device) for _ in range(k)]
        tv = [torch.zeros(H) for _ in range(k)]
        nb = 0
        for i, h, pos, pe, ids in iterate_layer_inputs(model, calib, k, bs, device):
            layer = model.model.layers[i]
            q, kk, _ = qkv_of(layer.self_attn, layer.input_layernorm(h), pe)
            a = exact_attention_weights(q, kk, layer.self_attn.scaling)  # (B,H,T,T)
            rows = vocab_map[ids]
            kern, sink = kernels[i]
            bk = betas[i][:, rows].permute(1, 0, 2)
            w = salience_matrix(kern.to(device), sink.to(device), bk, ids.shape[1])
            recv_t = a[:, :, 1:, 1:].sum(2)  # масса, полученная ключом j (j ≥ 1), (B,H,T-1)
            recv_m = w[:, :, 1:, 1:].sum(2)
            idx = rows[:, 1:].unsqueeze(1).expand(-1, H, -1)  # (B,H,T-1)
            for hh in range(H):
                tgt[i][hh].index_add_(0, idx[:, hh].reshape(-1), recv_t[:, hh].reshape(-1))
                mod[i][hh].index_add_(0, idx[:, hh].reshape(-1), recv_m[:, hh].reshape(-1))
            tv[i] += (0.5 * (a - w).abs().sum(-1)).mean(dim=(0, 2)).cpu() * ids.shape[0]
            if i == 0:
                nb += ids.shape[0]
        for i in range(k):
            tvs[i] = tv[i] / nb
            if it < iters:
                betas[i] += torch.log((tgt[i] + alpha) / (mod[i] + alpha))
                betas[i][:, -1] = 0
    allv = torch.cat(tvs)
    n_ex = int(round(exact_frac * allv.numel()))
    ex = torch.zeros(allv.numel(), dtype=torch.bool)
    ex[allv.argsort(descending=True)[:n_ex]] = True
    ex = ex.view(k, H)
    tap = InputIdsTap(model)
    rep = Replacement(model)
    for i in range(k):
        kern, sink = kernels[i]
        rep.add(i, "self_attn", SalienceAttention(model.model.layers[i].self_attn, kern.to(device),
                                                  sink.to(device), betas[i], vocab_map, tap, ex[i], window))
    info = {"iters": iters, "beta_rows": nrows, "exact_heads": int(ex.sum()), "total_heads": k * H,
            "tv_per_layer": [[round(float(x), 4) for x in t] for t in tvs]}
    return rep, info


# ============================================================ O5: окно + хвост

@torch.no_grad()
def build_window_tail(model, calib, k, bs, device, window=16, iters=3, alpha=1e-4):
    """β_h(v) для хвоста (t−j ≥ w, j ≥ 1). Инициализация — средняя точная оценка s_tj по
    ключам-токенам v (log-mean-exp завышает вес из-за неравенства Йенсена и проигрывает).
    Затем IPF при точном окне: β ← β + log((A_true(v)+α)/(A_model(v)+α)), где A(v) — масса
    внимания, полученная в хвосте ключами-токенами v. Детерминированно, без SGD."""
    H = model.config.num_attention_heads
    uniq = torch.unique(calib)
    V = model.config.vocab_size
    vocab_map = torch.full((V,), uniq.numel(), dtype=torch.long)
    vocab_map[uniq] = torch.arange(uniq.numel())
    vocab_map = vocab_map.to(device)
    nrows = uniq.numel() + 1
    betas = [torch.zeros(H, nrows, device=device) for _ in range(k)]

    def scores(i, h, pe):
        layer = model.model.layers[i]
        q, kk, _ = qkv_of(layer.self_attn, layer.input_layernorm(h), pe)
        return torch.matmul(q.float(), kk.float().transpose(-1, -2)) * layer.self_attn.scaling

    # проход 0: средняя оценка по ключу-токену
    ssum = [torch.zeros(H, nrows, device=device) for _ in range(k)]
    scnt = [torch.zeros(nrows, device=device) for _ in range(k)]
    for i, h, pos, pe, ids in iterate_layer_inputs(model, calib, k, bs, device):
        s = scores(i, h, pe)
        T = s.shape[-1]
        t = torch.arange(T, device=device)
        d = t[:, None] - t[None, :]
        tail = ((d >= window) & (t[None, :] > 0)).float()
        recv = (s * tail).sum(2)  # (B,H,T) по ключу
        nq = tail.sum(0)  # (T,)
        rows = vocab_map[ids]
        for hh in range(H):
            ssum[i][hh].index_add_(0, rows.reshape(-1), recv[:, hh].reshape(-1))
        scnt[i].index_add_(0, rows.reshape(-1), nq.expand(ids.shape[0], -1).reshape(-1))
    for i in range(k):
        head_mean = ssum[i].sum(1) / scnt[i].sum().clamp_min(1)
        betas[i] = (ssum[i] + 8 * head_mean[:, None]) / (scnt[i][None, :] + 8)

    # IPF по массам внимания в хвосте
    for it in range(iters):
        at = [torch.zeros(H, nrows, device=device) for _ in range(k)]
        am = [torch.zeros(H, nrows, device=device) for _ in range(k)]
        for i, h, pos, pe, ids in iterate_layer_inputs(model, calib, k, bs, device):
            s = scores(i, h, pe)
            T = s.shape[-1]
            t = torch.arange(T, device=device)
            d = t[:, None] - t[None, :]
            causal = d >= 0
            exact = causal & ((d < window) | (t[None, :] == 0))
            tail = (d >= window) & (t[None, :] > 0)
            p_true = torch.softmax(s.masked_fill(~causal, float("-inf")), -1)
            rows = vocab_map[ids]
            bk = betas[i][:, rows].permute(1, 0, 2).unsqueeze(2)  # (B,H,1,T)
            logits = torch.where(exact, s, torch.where(tail, bk.expand_as(s), torch.full_like(s, float("-inf"))))
            p_mod = torch.softmax(logits, -1)
            tf = tail.float()
            rt, rm = (p_true * tf).sum(2), (p_mod * tf).sum(2)  # (B,H,T)
            for hh in range(H):
                at[i][hh].index_add_(0, rows.reshape(-1), rt[:, hh].reshape(-1))
                am[i][hh].index_add_(0, rows.reshape(-1), rm[:, hh].reshape(-1))
        for i in range(k):
            betas[i] += torch.log((at[i] + alpha) / (am[i] + alpha))
    tap = InputIdsTap(model)
    rep = Replacement(model)
    for i in range(k):
        rep.add(i, "self_attn", WindowTailAttention(model.model.layers[i].self_attn, window, betas[i],
                                                    vocab_map, tap))
    return rep, {"window": window, "iters": iters, "beta_rows": nrows}


# ============================================================ диспетчер

def build(model, method: str, k: int, calib: torch.Tensor | None, device, bs: int = 4, **p):
    """Возвращает (Replacement, info). Replacement ещё не включён."""
    from .mlp import MLP_METHODS, build_mlp

    if "+" in method:  # комбинация «внимание + MLP»: params = {"attn": {...}, "mlp": {...}}
        am, mm = method.split("+")
        rep, info_a = build(model, am, k, calib, device, bs, **p.get("attn", {}))
        attn_mods = {i: m for (i, a), m in rep.new.items() if a == "self_attn"}
        mods, info_m = build_mlp(model, mm, k, calib, device, bs, attn_mods, **p.get("mlp", {}))
        for i, m in mods.items():
            rep.add(i, "mlp", m)
        return rep, {"attn": info_a, "mlp": info_m}
    if method in MLP_METHODS:
        mods, info = build_mlp(model, method, k, calib, device, bs, None, **p)
        rep = Replacement(model)
        for i, m in mods.items():
            rep.add(i, "mlp", m)
        return rep, info

    rep = Replacement(model)
    info: dict = {}
    layers = model.model.layers
    if method == "orig":
        return rep, info
    if method == "drop":
        for i in range(k):
            rep.add(i, "block", IdentityBlock())
        return rep, info
    if method in ("sparse", "strided"):
        stride = p.get("stride", 64) if method == "strided" else None
        for i in range(k):
            rep.add(i, "self_attn", SparseAttention(layers[i].self_attn, window=p.get("window", 64),
                                                     n_sink=p.get("n_sink", 1), stride=stride,
                                                     sink_ratio=p.get("sink_ratio")))
        return rep, info
    if method == "performer":
        for i in range(k):
            rep.add(i, "self_attn", PerformerAttention(layers[i].self_attn, p.get("features", 256),
                                                       p.get("seed", 0)).to(device))
        return rep, info
    if method == "linear_elu":
        for i in range(k):
            rep.add(i, "self_attn", LinearEluAttention(layers[i].self_attn))
        return rep, info
    if method in ("relpos", "head_typed"):
        D = p.get("max_offset", 64)
        kernels = calib_relpos(model, calib, k, bs, device, D)
        if method == "relpos":
            for i in range(k):
                kern, sink = kernels[i]
                rep.add(i, "self_attn", RelPosAttention(layers[i].self_attn, kern.to(device), sink.to(device)))
            return rep, {"max_offset": D}
        tv = calib_head_fit(model, calib, k, bs, device, kernels)
        frac = p.get("frac", 0.5)
        n_rep = 0
        per_layer = {}
        allv = torch.cat(tv)
        n_sel = int(round(frac * allv.numel()))
        sel = torch.zeros(allv.numel(), dtype=torch.bool)
        sel[allv.argsort()[:n_sel]] = True  # головы с наименьшей TV-дистанцией
        sel = sel.view(k, -1)
        for i in range(k):
            mask = sel[i]
            n_rep += int(mask.sum())
            kern, sink = kernels[i]
            rep.add(i, "self_attn", HeadTypedAttention(layers[i].self_attn, kern.to(device), sink.to(device),
                                                       mask, content=p.get("content", "exact"),
                                                       window=p.get("window", 64)))
            per_layer[f"layer{i}"] = {"tv": [round(float(x), 4) for x in tv[i]], "replaced": mask.nonzero().flatten().tolist()}
        return rep, {"frac": frac, "replaced_heads": n_rep, "total_heads": k * model.config.num_attention_heads,
                     "per_layer": per_layer}
    if method == "a3":
        return build_a3(model, calib, k, bs, device, p.get("ratio_qk", 0.5), p.get("ratio_vo", 0.5),
                        p.get("ratio_ffn", 0.5), p.get("mlp", True), p.get("robust_tau"))
    if method == "ridge_conv":
        return build_ridge_conv(model, calib, k, bs, device, p.get("window", 8), p.get("pca_dim", 256),
                                p.get("lam", 1e-2))
    if method == "window_tail":
        return build_window_tail(model, calib, k, bs, device, p.get("window", 16), p.get("iters", 3),
                                 p.get("alpha", 1e-4))
    if method == "salience":
        return build_salience(model, calib, k, bs, device, p.get("max_offset", 64), p.get("iters", 3),
                              p.get("alpha", 1.0), p.get("exact_frac", 0.0), p.get("window"))
    if method == "ngram":
        return build_ngram(model, calib, k, bs, device, p.get("m", 3), p.get("pos_max", 16),
                           p.get("sweeps", 2), p.get("shrink", 1.0))
    raise ValueError(f"неизвестный метод {method}")
