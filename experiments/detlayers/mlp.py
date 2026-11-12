"""Замены SwiGLU-MLP первых слоёв: down(silu(gate·x) ⊙ up·x).

Литература (калибровка по пути исходной модели, как в оригинальных работах):
  a3_mlp     — A3 FFN [22]: r нейронов по sqrt(E z_i²)·||w_down,i||
  topk_mlp   — контекстная разреженность: на токене top-k нейронов по |z_i| («оракул» Deja Vu [28],
               он же разреженность памяти «ключ–значение» Geva [3])
  asvd_mlp   — ASVD [26]: SVD(W·S), S = diag(E|x|)^α
  svdllm_mlp — SVD-LLM [27]: отбеливание по Холецкому XᵀX + усечённое SVD (без дообучения)
  moe_mlp    — MoEfication [29]: сбалансированный k-means по нейронам + маршрутизация по центроидам

Наши (последовательная калибровка по пути «ученика», цель компенсирует дрейф):
  ridge_mlp   — MLP ≈ W x̂ + b (+ r точных нейронов при keep_ratio > 0)
  table_mlp   — MLP ≈ T[x_t] + W x̂ + b (таблица по токену + линейная поправка, backfitting)
  cluster_mlp — кусочно-линейная: k-means по x̂, в каждом кластере ridge-карта на PCA-признаках

Позиция 0 во всех методах считается исходным MLP (массивная активация первого токена).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import embed, run_layers

MLP_METHODS = {"a3_mlp", "topk_mlp", "asvd_mlp", "svdllm_mlp", "moe_mlp", "ridge_mlp", "table_mlp",
               "cluster_mlp", "routed_mlp", "cats_mlp"}


class ExactPos0(nn.Module):
    """Обёртка: позиция 0 — исходный MLP, остальные — приближение."""

    def __init__(self, orig_mlp, approx: nn.Module):
        super().__init__()
        self.orig = orig_mlp
        self.approx = approx

    def forward(self, x):
        out = self.approx(x)
        out0 = self.orig(x[:, :1])
        return torch.cat([out0.to(out.dtype), out[:, 1:]], dim=1)

    def extra_cost(self):
        f = getattr(self.approx, "extra_cost", None)
        c = dict(f() if f else {})
        g = getattr(self.approx, "cost", None)
        if g:
            full = 3 * self.orig.gate_proj.weight.numel()
            pc, fc = g()
            c.update(param_frac=round(pc / full, 4), flop_frac=round(fc / full, 4))
        return c


def _z(mlp, x):
    return mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x)


# ============================================================ калибровка: входы MLP учителя

@torch.no_grad()
def iterate_mlp_inputs(model, calib, k, bs, device):
    """Отдаёт (i, x̂_mlp, ids) для слоёв i < k по пути исходной модели."""
    layers = model.model.layers
    for s in range(0, calib.shape[0], bs):
        ids = calib[s:s + bs].to(device)
        h, pos, pe = embed(model, ids)
        for i in range(k):
            L = layers[i]
            a, _ = L.self_attn(L.input_layernorm(h), position_embeddings=pe, attention_mask=None)
            hm = h + a
            x = L.post_attention_layernorm(hm)
            yield i, x, ids
            h = hm + L.mlp(x)


# ============================================================ M-L1: A3 FFN

class SelectedMLP(nn.Module):
    def __init__(self, mlp, keep):
        super().__init__()
        self.register_buffer("gate", mlp.gate_proj.weight[keep].detach().clone())
        self.register_buffer("up", mlp.up_proj.weight[keep].detach().clone())
        self.register_buffer("down", mlp.down_proj.weight[:, keep].detach().clone())
        self.act = mlp.act_fn

    def forward(self, x):
        return F.linear(self.act(F.linear(x, self.gate)) * F.linear(x, self.up), self.down)

    def extra_cost(self):
        return {"neurons": self.gate.shape[0]}

    def cost(self):
        n = 3 * self.gate.numel()
        return n, n


def robust_rows_z(z, tau):
    """Маска строк (B,T) для сумм по токенам: без позиции 0 и без выбросов ||z|| > tau·медиана."""
    zn = z.float().norm(dim=-1)
    keep = zn <= tau * zn[:, 1:].median()
    keep[:, 0] = False
    return keep


@torch.no_grad()
def find_sink_neurons(model, calib, k, bs, device, tau=20.0, cover=0.9, max_n=16, n_windows=64):
    """Нейроны стока (правило 2): для слоёв i < k — нейроны, которые на «массивных» токенах
    (||MLP(x)|| > tau·медиана) покрывают долю cover вклада |z_j|·||w_d,j||. Находятся по исходной
    модели на первых n_windows окнах калибровки; таких нейронов единицы (Qwen3: слой 2, SmolLM2: слой 1)."""
    I = model.config.intermediate_size
    votes = [torch.zeros(I, dtype=torch.long) for _ in range(k)]
    for i, x, _ in iterate_mlp_inputs(model, calib[:n_windows], k, bs, device):
        mlp = model.model.layers[i].mlp
        z = _z(mlp, x).float()
        y = mlp.down_proj(z).float()
        n = y.norm(dim=-1)  # (B,T)
        big = n > tau * n[:, 1:].median()
        if not big.any():
            continue
        wd = mlp.down_proj.weight.detach().float().norm(dim=0)
        c = (z[big].abs() * wd)  # (nbig, I)
        c = c / c.sum(-1, keepdim=True).clamp_min(1e-12)
        v, idx = c.sort(-1, descending=True)
        need = (v.cumsum(-1) < cover).sum(-1) + 1  # число нейронов для покрытия доли cover
        for row, m in zip(idx, need.tolist()):
            votes[i].index_add_(0, row[:m].cpu(), torch.ones(m, dtype=torch.long))
    out = {}
    for i in range(k):
        nz = votes[i].nonzero().flatten()
        if nz.numel():
            top = votes[i][nz].argsort(descending=True)[:max_n]
            out[i] = nz[top].sort().values.to(device)
    return out


def _robust_resid(xf, r, tau, stats=None):
    """Строки регрессии с ||r|| > tau·медиана обнуляются (ловушка калибровки в остатке: защищённые
    нейроны стока могут дать массивный выход там, где у учителя его нет, напр. при окне внимания)."""
    if not tau:
        return xf, r
    n = r.norm(dim=-1)
    m = n <= tau * n.median()
    if stats is not None:
        stats["dropped_resid"] = stats.get("dropped_resid", 0) + int((~m).sum())
    return xf * m[:, None], r * m[:, None]


class PlusNeurons(nn.Module):
    """approx(x) + точный вклад защищённых нейронов S (approx построен без них)."""

    def __init__(self, mlp, approx, S):
        super().__init__()
        self.approx = approx
        self.register_buffer("gate", mlp.gate_proj.weight[S].detach().clone())
        self.register_buffer("up", mlp.up_proj.weight[S].detach().clone())
        self.register_buffer("down", mlp.down_proj.weight[:, S].detach().clone())
        self.act = mlp.act_fn

    def forward(self, x):
        return self.approx(x) + F.linear(self.act(F.linear(x, self.gate)) * F.linear(x, self.up), self.down)

    def extra_cost(self):
        f = getattr(self.approx, "extra_cost", None)
        return {**(f() if f else {}), "sink_neurons": self.gate.shape[0]}

    def cost(self):
        pc, fc = self.approx.cost()
        n = 3 * self.gate.numel()
        return pc + n, fc + n


@torch.no_grad()
def build_a3_mlp(model, calib, k, bs, device, ratio=0.5, robust_tau=None, protect=None):
    I = model.config.intermediate_size
    protect = protect or {}
    z2 = [torch.zeros(I, dtype=torch.float64) for _ in range(k)]
    for i, x, _ in iterate_mlp_inputs(model, calib, k, bs, device):
        z = _z(model.model.layers[i].mlp, x).float()
        if robust_tau:  # иначе сумму ∑z² держат позиция 0 и токены с массивной активацией
            z = z * robust_rows_z(z, robust_tau)[..., None]
        z2[i] += (z ** 2).sum((0, 1)).cpu().double()
    out = {}
    r = max(1, int(I * ratio))
    for i in range(k):
        mlp = model.model.layers[i].mlp
        sc = z2[i].sqrt() * mlp.down_proj.weight.detach().cpu().double().norm(dim=0)
        keep = sc.argsort(descending=True)[:r].sort().values.to(device)
        if i in protect:
            keep = torch.unique(torch.cat([keep, protect[i]]))
        out[i] = SelectedMLP(mlp, keep)
    return out, {"neurons": r, "of": I, **({"robust_tau": robust_tau} if robust_tau else {})}


# ============================================================ CATS (Lee et al., 2024): порог по |act(gate)|

class CatsMLP(nn.Module):
    """CATS: gate считается полностью, нейроны с |act(gate)| < thr пропускаются (up и down только для активных).
    Эталонная реализация маскирует; стоимость (1 + 2·доля активных)/3 от MLP."""

    def __init__(self, mlp, thr: float, ratio: float):
        super().__init__()
        self.mlp, self.thr, self.ratio = mlp, float(thr), float(ratio)

    def forward(self, x):
        g = self.mlp.act_fn(self.mlp.gate_proj(x))
        g = g * (g.abs() >= self.thr)
        return self.mlp.down_proj(g * self.mlp.up_proj(x))

    def extra_cost(self):
        return {"threshold": round(self.thr, 5), "target_active": self.ratio, "note": "gate считается полностью"}

    def cost(self):
        d, I = self.mlp.gate_proj.weight.shape[1], self.mlp.gate_proj.weight.shape[0]
        return 3 * d * I, int(d * I * (1 + 2 * self.ratio))


@torch.no_grad()
def build_cats_mlp(model, calib, k, bs, device, ratio=0.25, sample=4_000_000):
    """Порог на слой = квантиль |act(gate)| уровня (1 − ratio) по калибровке (позиции ≥ 1, подвыборка)."""
    g = torch.Generator().manual_seed(0)
    pools = [[] for _ in range(k)]
    for i, x, _ in iterate_mlp_inputs(model, calib, k, bs, device):
        mlp = model.model.layers[i].mlp
        a = mlp.act_fn(mlp.gate_proj(x[:, 1:])).float().abs().reshape(-1)
        idx = torch.randperm(a.numel(), generator=g)[: max(1, sample // (calib.shape[0] // bs + 1))]
        pools[i].append(a[idx.to(a.device)].cpu())
    out, info = {}, {"ratio": ratio}
    for i in range(k):
        vals = torch.cat(pools[i])
        thr = vals.kthvalue(max(1, int((1 - ratio) * vals.numel()))).values.item()
        out[i] = CatsMLP(model.model.layers[i].mlp, thr, ratio)
        info[f"layer{i}"] = {"threshold": round(thr, 5), "measured_active": round(float((vals >= thr).float().mean()), 4)}
    return out, info


# ============================================================ M-L2 / Deja Vu (оракул): top-k

class TopKMLP(nn.Module):
    def __init__(self, mlp, k_neurons):
        super().__init__()
        self.mlp, self.kn = mlp, k_neurons

    def forward(self, x):
        z = _z(self.mlp, x)
        thr = z.abs().topk(self.kn, dim=-1).values[..., -1:]
        return self.mlp.down_proj(z * (z.abs() >= thr))

    def extra_cost(self):
        return {"active_neurons": self.kn, "note": "gate/up считаются полностью (оракул)"}

    def cost(self):
        d, I = self.mlp.gate_proj.weight.shape[1], self.mlp.gate_proj.weight.shape[0]
        return 3 * d * I, 2 * d * I + self.kn * d


# ============================================================ M-L4/M-L5: ASVD и SVD-LLM

class LowRankLinear(nn.Module):
    """W ≈ A·B, A: (out, r), B: (r, in)."""

    def __init__(self, A, B):
        super().__init__()
        self.register_buffer("A", A)
        self.register_buffer("B", B)

    def forward(self, x):
        return F.linear(F.linear(x.float(), self.B.float()), self.A.float()).to(x.dtype)


class LowRankMLP(nn.Module):
    def __init__(self, mlp, gate, up, down):
        super().__init__()
        self.gate, self.up, self.down, self.act = gate, up, down, mlp.act_fn

    def forward(self, x):
        return self.down(self.act(self.gate(x)) * self.up(x))

    def extra_cost(self):
        return {"ranks": [m.A.shape[1] for m in (self.gate, self.up, self.down)]}

    def cost(self):
        n = sum(m.A.numel() + m.B.numel() for m in (self.gate, self.up, self.down))
        return n, n


def _rank_for(out_f, in_f, ratio):
    """Ранг, при котором A·B занимает долю `ratio` параметров исходной матрицы."""
    return max(1, int(ratio * out_f * in_f / (out_f + in_f)))


def _lowrank_asvd(W, mean_abs, alpha, r):
    s = mean_abs.clamp_min(1e-6) ** alpha  # (in,)
    U, S, Vt = torch.linalg.svd(W * s[None, :], full_matrices=False)
    A = U[:, :r] * S[:r]
    B = Vt[:r] / s[None, :]
    return A, B


def _lowrank_svdllm(W, C, r, damp=1e-4):
    C = C + damp * C.diagonal().mean() * torch.eye(C.shape[0], dtype=C.dtype)
    L = torch.linalg.cholesky(C)  # C = L Lᵀ
    U, S, Vt = torch.linalg.svd(W @ L, full_matrices=False)
    A = U[:, :r] * S[:r]
    B = torch.linalg.solve_triangular(L, Vt[:r], upper=False, left=False)  # X·L = Vt_r → Vt_r·L⁻¹
    return A, B


@torch.no_grad()
def build_svd_mlp(model, calib, k, bs, device, kind, ratio=0.5, alpha=0.5, robust_tau=None, protect=None):
    """kind ∈ {'asvd', 'svdllm'}: низкий ранг для gate/up/down первых k слоёв.
    robust_tau: из статистик исключаются позиция 0 и токены с ||z|| > tau·медиана (массивные активации);
    в оригинальных методах статистики копятся по всем токенам."""
    cfg = model.config
    d, I = cfg.hidden_size, cfg.intermediate_size
    protect = protect or {}
    if kind == "asvd":
        sx = [torch.zeros(d, dtype=torch.float64) for _ in range(k)]
        sz = [torch.zeros(I, dtype=torch.float64) for _ in range(k)]
    else:
        sx = [torch.zeros(d, d, dtype=torch.float64) for _ in range(k)]
        sz = [torch.zeros(I, I, dtype=torch.float64) for _ in range(k)]
    n = [0] * k
    for i, x, _ in iterate_mlp_inputs(model, calib, k, bs, device):
        zz = _z(model.model.layers[i].mlp, x).float()
        if robust_tau:
            keep = robust_rows_z(zz, robust_tau)
            z, xf = zz[keep], x.float()[keep]
        else:
            z, xf = zz.reshape(-1, I), x.float().reshape(-1, d)
        if kind == "asvd":
            sx[i] += xf.abs().sum(0).cpu().double()
            sz[i] += z.abs().sum(0).cpu().double()
        else:
            sx[i] += (xf.T @ xf).cpu().double()
            sz[i] += (z.T @ z).cpu().double()
        n[i] += xf.shape[0]
    out = {}
    for i in range(k):
        mlp = model.model.layers[i].mlp
        mods = []
        S = protect.get(i)
        for lin, stat in ((mlp.gate_proj, sx[i]), (mlp.up_proj, sx[i]), (mlp.down_proj, sz[i])):
            W = lin.weight.detach().cpu().double()
            if S is not None:  # нейроны стока исключены из факторизации и добавляются точно (PlusNeurons)
                W = W.clone()
                if lin is mlp.down_proj:
                    W[:, S.cpu()] = 0
                else:
                    W[S.cpu()] = 0
            r = _rank_for(W.shape[0], W.shape[1], ratio)
            if kind == "asvd":
                A, B = _lowrank_asvd(W, stat / n[i], alpha, r)
            else:
                A, B = _lowrank_svdllm(W, stat / n[i], r)
            mods.append(LowRankLinear(A.float().to(device), B.float().to(device)))
        out[i] = LowRankMLP(mlp, *mods)
        if S is not None:
            out[i] = PlusNeurons(mlp, out[i], S)
    return out, {"param_ratio": ratio, "alpha": alpha if kind == "asvd" else None, "robust_tau": robust_tau}


# ============================================================ M-L7: MoEfication

def balanced_kmeans(X: torch.Tensor, n_clusters: int, iters: int = 20, seed: int = 0) -> torch.Tensor:
    """Детерминированный сбалансированный k-means (равные размеры кластеров):
    инициализация k-means++ с фиксированным seed, присваивание — жадно по возрастанию
    расстояния с ограничением ёмкости."""
    g = torch.Generator().manual_seed(seed)
    N = X.shape[0]
    cap = math.ceil(N / n_clusters)
    cent = X[torch.randint(N, (1,), generator=g)]
    for _ in range(1, n_clusters):
        dmin = torch.cdist(X, cent).min(1).values ** 2
        cent = torch.cat([cent, X[torch.multinomial(dmin / dmin.sum(), 1, generator=g)]])
    assign = torch.zeros(N, dtype=torch.long)
    for _ in range(iters):
        D = torch.cdist(X, cent)  # (N, C)
        order = D.flatten().argsort()
        assign.fill_(-1)
        load = torch.zeros(n_clusters, dtype=torch.long)
        for flat in order.tolist():
            j, c = divmod(flat, n_clusters)
            if assign[j] < 0 and load[c] < cap:
                assign[j] = c
                load[c] += 1
            if (assign >= 0).all():
                break
        new = torch.stack([X[assign == c].mean(0) for c in range(n_clusters)])
        if torch.allclose(new, cent):
            break
        cent = new
    return assign


class MoEMLP(nn.Module):
    """Нейроны разбиты на E экспертов; на токене включаем top-m экспертов по скалярному
    произведению входа с центроидом gate-строк эксперта (маршрутизация без обучения)."""

    def __init__(self, mlp, assign, n_experts, top_m):
        super().__init__()
        self.mlp = mlp
        self.register_buffer("assign", assign)
        W = mlp.gate_proj.weight.detach()
        cent = torch.stack([W[assign == e].mean(0) for e in range(n_experts)])
        self.register_buffer("cent", cent)
        self.E, self.m = n_experts, top_m

    def forward(self, x):
        score = F.linear(x, self.cent)  # (B,T,E)
        top = score.topk(self.m, dim=-1).indices
        on = torch.zeros_like(score, dtype=torch.bool).scatter_(-1, top, True)
        mask = on[..., self.assign]  # (B,T,I)
        return self.mlp.down_proj(_z(self.mlp, x) * mask)

    def extra_cost(self):
        return {"experts": self.E, "active": self.m}

    def cost(self):
        full = 3 * self.mlp.gate_proj.weight.numel()
        return full + self.cent.numel(), full * self.m / self.E + self.cent.numel()


@torch.no_grad()
def build_moe_mlp(model, k, device, n_experts=32, active_frac=0.25):
    out = {}
    for i in range(k):
        mlp = model.model.layers[i].mlp
        W = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], 1).detach().float().cpu()
        W = W / W.norm(dim=1, keepdim=True)
        assign = balanced_kmeans(W, n_experts).to(device)
        out[i] = MoEMLP(mlp, assign, n_experts, max(1, int(round(active_frac * n_experts))))
    return out, {"experts": n_experts, "active_frac": active_frac}


# ============================================================ наши: последовательная калибровка

class _StudentLayer(nn.Module):
    """Блок с (возможно) заменённым вниманием и MLP — только для калибровки."""

    def __init__(self, L, attn=None, mlp=None):
        super().__init__()
        self.L, self.attn, self.mlp = L, attn or L.self_attn, mlp or L.mlp

    def forward(self, h, attention_mask=None, position_embeddings=None, position_ids=None, **kw):
        a, _ = self.attn(self.L.input_layernorm(h), position_embeddings=position_embeddings,
                         attention_mask=None)
        h = h + a
        return h + self.mlp(self.L.post_attention_layernorm(h))


@torch.no_grad()
def sequential_pairs(model, calib, i, bs, device, student_layers, attn_mods=None, tap=None):
    """Для слоя i отдаёт (x̂_S, y, ids): x̂_S — вход MLP по пути ученика,
    y = h_T,i+1 − h_S,mid — нужный выход MLP, чтобы догнать учителя после блока i."""
    layers = model.model.layers
    for s in range(0, calib.shape[0], bs):
        ids = calib[s:s + bs].to(device)
        if tap is not None:
            tap.ids = ids
        for m_ in (attn_mods or {}).values():  # заменённое внимание с таблицами по токенам (O4)
            if getattr(m_, "tap", None) is not None:
                m_.tap.ids = ids
        h0, pos, pe = embed(model, ids)
        hT = h0
        for j in range(i + 1):  # учитель: исходные модули
            L = layers[j]
            a, _ = L.self_attn(L.input_layernorm(hT), position_embeddings=pe, attention_mask=None)
            hT = hT + a
            hT = hT + L.mlp(L.post_attention_layernorm(hT))
        hS = run_layers(model, student_layers, h0, pos, pe, 0, i)
        L = layers[i]
        attn = (attn_mods or {}).get(i, L.self_attn)
        a, _ = attn(L.input_layernorm(hS), position_embeddings=pe, attention_mask=None)
        hmid = hS + a
        x = L.post_attention_layernorm(hmid)
        yield x, (hT - hmid), ids


def robust_pairs(gen, tau: float, stats: dict | None = None):
    """Робастная калибровка: строки (позиции), где норма цели ||y|| превышает tau·медиану по партии,
    обнуляются (и вход, и цель), т.е. исключаются из всех сумм наименьших квадратов.
    Причина: токены с массивной активацией (норма ~10³ при медиане ~5) на позициях ≥1 встречаются
    в калибровке единично, но держат почти всю энергию целей и «отравляют» подгонку (обрыв на k=3)."""
    for x, y, ids in gen:
        yn = y[:, 1:].float().norm(dim=-1)  # (B, T-1)
        bad = yn > tau * yn.median()
        if bad.any():
            x = x.clone(); y = y.clone()
            m = torch.cat([torch.zeros_like(bad[:, :1]), bad], 1)
            x[m] = 0; y[m] = 0
            if stats is not None:
                stats["dropped"] = stats.get("dropped", 0) + int(bad.sum())
        yield x, y, ids


def reduced_rank(Wf, Gxx, r):
    """Регрессия пониженного ранга (Reinsel–Velu): МНК-решение W (d_in×d_out) проецируется на r главных
    направлений предсказаний Ŷ = X W, т.е. на старшие собственные векторы Wᵀ Gxx W. Возвращает A (d_in×r), B (r×d_out)."""
    W = Wf[:-1]
    S = W.T @ Gxx @ W
    S = (S + S.T) / 2
    evals, evecs = torch.linalg.eigh(S)
    V = evecs[:, -r:]
    return W @ V, V.T


class RidgeMLP(nn.Module):
    """MLP(x̂) ≈ x̂ Wᵀ + b (+ точные нейроны keep); при A,B — низкоранговая карта x̂ A B + b."""

    def __init__(self, mlp, W, b, keep=None, A=None, B=None):
        super().__init__()
        self.register_buffer("W", W if W is not None else torch.zeros(0))  # (d_out, d_in)
        self.register_buffer("b", b)
        self.register_buffer("A", A if A is not None else torch.zeros(0))  # (d_in, r)
        self.register_buffer("B", B if B is not None else torch.zeros(0))  # (r, d_out)
        self.sel = SelectedMLP(mlp, keep) if keep is not None else None

    def forward(self, x):
        if self.A.numel():
            out = (x.float() @ self.A.float()) @ self.B.float() + self.b.float()
        else:
            out = F.linear(x.float(), self.W.float(), self.b.float())
        if self.sel is not None:
            out = out + self.sel(x).float()
        return out.to(x.dtype)

    def extra_cost(self):
        return {"linear_params": self.W.numel() + self.A.numel() + self.B.numel(),
                "rank": self.A.shape[1] if self.A.numel() else None,
                "exact_neurons": 0 if self.sel is None else self.sel.gate.shape[0]}

    def cost(self):
        n = self.W.numel() + self.A.numel() + self.B.numel() + (0 if self.sel is None else 3 * self.sel.gate.numel())
        return n, n


class TableMLP(nn.Module):
    """MLP ≈ T[x_t] + x̂ Wᵀ + b."""

    def __init__(self, W, b, table, vocab_map, tap):
        super().__init__()
        self.register_buffer("W", W)
        self.register_buffer("b", b)
        self.register_buffer("table", table)
        self.register_buffer("vocab_map", vocab_map)
        self.tap = tap

    def forward(self, x):
        ids = self.tap.ids[:, : x.shape[1]].to(x.device)
        out = F.linear(x.float(), self.W.float(), self.b.float()) + self.table[self.vocab_map[ids]].float()
        return out.to(x.dtype)

    def extra_cost(self):
        return {"table_rows": self.table.shape[0], "linear_params": self.W.numel()}

    def cost(self):  # таблица — память, но не вычисления (один lookup)
        return self.W.numel() + self.table.numel(), self.W.numel() + self.table.shape[1]


class ClusterMLP(nn.Module):
    """Кусочно-линейная: ближайший центроид в пространстве PCA → своя ridge-карта."""

    def __init__(self, P, cent, Ws, bs_):
        super().__init__()
        self.register_buffer("P", P)  # (d, p)
        self.register_buffer("cent", cent)  # (C, p)
        self.register_buffer("Ws", Ws)  # (C, p, d)
        self.register_buffer("bs_", bs_)  # (C, d)

    def forward(self, x):
        xp = x.float() @ self.P.float()
        c = torch.cdist(xp.reshape(-1, xp.shape[-1]), self.cent.float()).argmin(-1).view(xp.shape[:-1])
        out = torch.einsum("btp,btpd->btd", xp, self.Ws[c].float()) + self.bs_[c].float()
        return out.to(x.dtype)

    def extra_cost(self):
        return {"clusters": self.cent.shape[0], "params": self.Ws.numel()}

    def cost(self):
        n = self.P.numel() + self.Ws.numel() + self.cent.numel()
        return n, self.P.numel() + self.Ws[0].numel() + self.cent.numel()


class RoutedMLP(nn.Module):
    """M-O4 (наш): детерминированный предиктор Deja Vu — для токена x_t вычисляются только
    κ нейронов из таблицы N[x_t] (по калибровке: наибольший средний вклад |z_i|·||w_d,i||),
    остаток исходного MLP компенсируется линейной картой W x̂ + b.
    Эталонная реализация считает z целиком и маскирует; стоимость — κ/I от MLP."""

    def __init__(self, mlp, idx_table, vocab_map, tap, W=None, b=None, P=None, A=None, kap=None, Wa=None, Wb=None,
                 protect=None, mode=None, gw=None, Pg=None, Qg=None, Pu=None, Qu=None, wd=None):
        super().__init__()
        self.mlp = mlp
        # режим предиктора: None/"token" — таблица, "linear" — (x P) A, "gate" — точная gate-проекция,
        # "quad" — низкоранговые gate/up: |act((x Pg) Qg) · ((x Pu) Qu)|·||w_d||
        self.mode = mode
        self.register_buffer("gw", gw if gw is not None else torch.zeros(0))
        self.register_buffer("Pg", Pg if Pg is not None else torch.zeros(0))
        self.register_buffer("Qg", Qg if Qg is not None else torch.zeros(0))
        self.register_buffer("Pu", Pu if Pu is not None else torch.zeros(0))
        self.register_buffer("Qu", Qu if Qu is not None else torch.zeros(0))
        self.register_buffer("wd", wd if wd is not None else torch.zeros(0))
        self.register_buffer("protect", protect if protect is not None else torch.zeros(0, dtype=torch.long))
        self.register_buffer("idx_table", idx_table)  # (V'+1, κ) int; пусто для контекстного предиктора
        self.register_buffer("vocab_map", vocab_map)
        self.tap = tap
        self.register_buffer("W", W if W is not None else torch.zeros(0))
        self.register_buffer("b", b if b is not None else torch.zeros(0))
        # контекстный предиктор: score = (x̂ P) A  — гребневая регрессия, без SGD
        self.register_buffer("P", P if P is not None else torch.zeros(0))
        self.register_buffer("A", A if A is not None else torch.zeros(0))
        # низкоранговая линейная поправка x̂ Wa Wb + b (регрессия пониженного ранга) вместо полной W
        self.register_buffer("Wa", Wa if Wa is not None else torch.zeros(0))
        self.register_buffer("Wb", Wb if Wb is not None else torch.zeros(0))
        self.kap = kap if kap is not None else idx_table.shape[1]

    def mode_kwargs(self):
        return dict(mode=self.mode, gw=self.gw if self.gw.numel() else None, Pg=self.Pg if self.Pg.numel() else None,
                    Qg=self.Qg if self.Qg.numel() else None, Pu=self.Pu if self.Pu.numel() else None,
                    Qu=self.Qu if self.Qu.numel() else None, wd=self.wd if self.wd.numel() else None)

    def select(self, x):
        if self.mode == "gate":  # двухступенчатый: gate считается точно, up/down — только для отобранных
            g = self.mlp.act_fn(self.mlp.gate_proj(x)).float()
            return (g.abs() * self.gw.float()).topk(self.kap, dim=-1).indices
        if self.mode == "quad":  # квадратичный предиктор малого ранга
            xf = x.float()
            g = self.mlp.act_fn((xf @ self.Pg.float()) @ self.Qg.float())
            u = (xf @ self.Pu.float()) @ self.Qu.float()
            return ((g * u).abs() * self.wd.float()).topk(self.kap, dim=-1).indices
        if self.P.numel():
            score = (x.float() @ self.P.float()) @ self.A[:-1].float() + self.A[-1].float()
            return score.topk(self.kap, dim=-1).indices
        ids = self.tap.ids[:, : x.shape[1]].to(x.device)
        return self.idx_table[self.vocab_map[ids]].long()

    def forward(self, x):
        z = _z(self.mlp, x)
        mask = torch.zeros_like(z, dtype=torch.bool)
        mask.scatter_(-1, self.select(x), True)
        if self.protect.numel():  # нейроны стока всегда активны (правило 2)
            mask[..., self.protect] = True
        out = self.mlp.down_proj(z * mask).float()
        if self.Wa.numel():
            out = out + (x.float() @ self.Wa.float()) @ self.Wb.float() + self.b.float()
        elif self.W.numel():
            out = out + F.linear(x.float(), self.W.float(), self.b.float())
        return out.to(x.dtype)

    def extra_cost(self):
        pred = self.mode if self.mode in ("gate", "quad") else ("linear" if self.P.numel() else "token")
        return {"active_neurons": self.kap, "predictor": pred,
                "table_rows": self.idx_table.shape[0], "sink_neurons": self.protect.numel(),
                "quad_rank": self.Pg.shape[1] if self.Pg.numel() else None,
                "lin_rank": self.Wa.shape[1] if self.Wa.numel() else ("full" if self.W.numel() else None)}

    def cost(self):
        d, I = self.mlp.gate_proj.weight.shape[1], self.mlp.gate_proj.weight.shape[0]
        lin = self.W.numel() + self.Wa.numel() + self.Wb.numel()
        act = self.kap + self.protect.numel()
        if self.mode == "gate":  # gate полностью (d·I), up и down — только активные
            n = d * I + 2 * d * act + lin
        elif self.mode == "quad":  # два низкоранговых произведения + активные нейроны
            n = self.Pg.numel() + self.Qg.numel() + self.Pu.numel() + self.Qu.numel() + 3 * d * act + lin
        else:
            n = 3 * d * act + lin + self.P.numel() + self.A.numel()
        return 3 * d * I + lin + self.P.numel() + self.A.numel(), n


def _ridge(G, B, lam):
    reg = lam * G.diagonal().mean()
    return torch.linalg.solve(G + reg * torch.eye(G.shape[0], dtype=G.dtype), B)


@torch.no_grad()
def build_ours_mlp(model, calib, k, bs, device, kind, attn_mods=None, lam=1e-3, keep_ratio=0.0,
                   sweeps=2, shrink=1.0, clusters=16, pca_dim=128, kmeans_iters=10,
                   active_ratio=0.25, min_count=8, lam_lin=1e-3, predictor="token", pred_dim=64, quad_rank=256,
                   robust_tau=None, rank=None, lin_rank=None, protect=None):
    """Последовательная подгонка слоёв 0..k-1 (наши методы). robust_tau — см. robust_pairs;
    protect — {слой: нейроны стока}, они считаются точно (ridge/routed)."""
    rstats = {}
    protect = protect or {}
    cfg = model.config
    d, I = cfg.hidden_size, cfg.intermediate_size
    student = list(model.model.layers)
    attn_mods = attn_mods or {}
    tap = None
    out = {}
    info = {}
    if kind in ("table_mlp", "routed_mlp"):
        from .methods import InputIdsTap
        uniq = torch.unique(calib)
        vocab_map = torch.full((cfg.vocab_size,), uniq.numel(), dtype=torch.long)
        vocab_map[uniq] = torch.arange(uniq.numel())
        vocab_map = vocab_map.to(device)
        tap = InputIdsTap(model)
        nrows = uniq.numel() + 1
    for i in range(k):
        mlp = model.model.layers[i].mlp
        pairs = lambda: (robust_pairs(sequential_pairs(model, calib, i, bs, device, student, attn_mods, tap),  # noqa: E731
                                      robust_tau, rstats) if robust_tau
                         else sequential_pairs(model, calib, i, bs, device, student, attn_mods, tap))
        S = protect.get(i)
        if kind == "ridge_mlp":
            keep = None
            if keep_ratio > 0:  # точные нейроны по критерию A3, линейная часть — на остаток
                z2 = torch.zeros(I, dtype=torch.float64)
                for x, y, _ in pairs():
                    z2 += (_z(mlp, x).float() ** 2).sum((0, 1)).cpu().double()
                sc = z2.sqrt() * mlp.down_proj.weight.detach().cpu().double().norm(dim=0)
                keep = sc.argsort(descending=True)[: int(I * keep_ratio)].sort().values.to(device)
            if S is not None:
                keep = S if keep is None else torch.unique(torch.cat([keep, S]))
            if keep is not None:
                sel = SelectedMLP(mlp, keep)
            G = torch.zeros(d + 1, d + 1, dtype=torch.float64)
            Bm = torch.zeros(d + 1, d, dtype=torch.float64)
            for x, y, _ in pairs():
                xf = x[:, 1:].float().reshape(-1, d)
                yf = y[:, 1:].float().reshape(-1, d)
                if keep is not None:
                    yf = yf - sel(x[:, 1:]).float().reshape(-1, d)
                    xf, yf = _robust_resid(xf, yf, robust_tau, rstats)
                phi = torch.cat([xf, torch.ones_like(xf[:, :1])], 1)
                G += (phi.T @ phi).cpu().double()
                Bm += (phi.T @ yf).cpu().double()
            Wf = _ridge(G, Bm, lam)
            if rank:  # регрессия пониженного ранга
                A_, B_ = reduced_rank(Wf, G[:-1, :-1], rank)
                mod = RidgeMLP(mlp, None, Wf[-1].float().to(device), keep, A_.float().to(device), B_.float().to(device))
            else:
                mod = RidgeMLP(mlp, Wf[:-1].T.float().contiguous().to(device), Wf[-1].float().to(device), keep)
        elif kind == "table_mlp":
            W = torch.zeros(d, d, device=device)
            b = torch.zeros(d, device=device)
            table = torch.zeros(nrows, d, device=device)
            for sw in range(sweeps):
                # (а) линейная часть на остаток после таблицы
                G = torch.zeros(d + 1, d + 1, dtype=torch.float64)
                Bm = torch.zeros(d + 1, d, dtype=torch.float64)
                for x, y, ids in pairs():
                    xf = x[:, 1:].float().reshape(-1, d)
                    yf = (y[:, 1:].float() - table[vocab_map[ids[:, 1:]]]).reshape(-1, d)
                    phi = torch.cat([xf, torch.ones_like(xf[:, :1])], 1)
                    G += (phi.T @ phi).cpu().double()
                    Bm += (phi.T @ yf).cpu().double()
                Wf = _ridge(G, Bm, lam)
                W = Wf[:-1].T.float().contiguous().to(device)
                b = Wf[-1].float().to(device)
                # (б) таблица — среднее остатка по токену (со сжатием)
                s_ = torch.zeros(nrows, d, device=device)
                c_ = torch.zeros(nrows, device=device)
                for x, y, ids in pairs():
                    r = (y[:, 1:].float() - F.linear(x[:, 1:].float(), W, b)).reshape(-1, d)
                    key = vocab_map[ids[:, 1:]].reshape(-1)
                    s_.index_add_(0, key, r)
                    c_.index_add_(0, key, torch.ones_like(key, dtype=torch.float))
                table = s_ / (c_[:, None] + shrink)
                table[-1] = 0
            mod = TableMLP(W, b, table, vocab_map, tap)
        elif kind == "routed_mlp":
            kap = max(1, int(I * active_ratio))
            wd = mlp.down_proj.weight.detach().float().norm(dim=0)  # (I,)
            acc = torch.zeros(nrows if predictor != "linear" else 1, I, device=device)
            cnt = torch.zeros(nrows if predictor != "linear" else 1, device=device)
            glob = torch.zeros(I, device=device)
            for x, y, ids in (pairs() if predictor == "token" else ()):  # таблица нужна только без предиктора
                z = _z(mlp, x[:, 1:]).float().abs() * wd  # вклад нейрона
                key = vocab_map[ids[:, 1:]].reshape(-1)
                acc.index_add_(0, key, z.reshape(-1, I))
                cnt.index_add_(0, key, torch.ones_like(key, dtype=torch.float))
                glob += z.reshape(-1, I).sum(0)
            if predictor == "gate":  # двухступенчатый: точная gate-проекция, up/down только для κ нейронов
                gw = (mlp.up_proj.weight.detach().float().norm(dim=1) * wd).to(device)
                mod = RoutedMLP(mlp, torch.zeros(0, kap, dtype=torch.int16, device=device), vocab_map, tap,
                                kap=kap, protect=S, mode="gate", gw=gw)
                idx_table = mod.idx_table
            elif predictor == "quad":  # низкоранговые gate/up по SVD весов (без калибровки), ранг quad_rank
                def lowrank(Wm, r):
                    U, Sv, Vh = torch.linalg.svd(Wm.detach().float().cpu(), full_matrices=False)
                    r = min(r, Sv.numel())
                    return (Vh[:r].T * Sv[:r]).contiguous(), U[:, :r].T.contiguous()  # x (d×r) → (r×I)
                Pg, Qg = lowrank(mlp.gate_proj.weight, quad_rank)
                Pu, Qu = lowrank(mlp.up_proj.weight, quad_rank)
                mod = RoutedMLP(mlp, torch.zeros(0, kap, dtype=torch.int16, device=device), vocab_map, tap,
                                kap=kap, protect=S, mode="quad", Pg=Pg.to(device), Qg=Qg.to(device),
                                Pu=Pu.to(device), Qu=Qu.to(device), wd=wd.to(device))
                idx_table = mod.idx_table
            elif predictor == "linear":  # контекстный: |z|·||w_d|| ≈ (x̂ P) A
                C = torch.zeros(d, d, dtype=torch.float64)
                for x, y, ids in pairs():
                    xf = x[:, 1:].float().reshape(-1, d)
                    C += (xf.T @ xf).cpu().double()
                P = torch.linalg.eigh(C)[1][:, -pred_dim:].flip(-1).float().to(device)
                G = torch.zeros(pred_dim + 1, pred_dim + 1, dtype=torch.float64)
                Bm = torch.zeros(pred_dim + 1, I, dtype=torch.float64)
                for x, y, ids in pairs():
                    xp = x[:, 1:].float().reshape(-1, d) @ P
                    tz = (_z(mlp, x[:, 1:]).float().abs() * wd).reshape(-1, I)
                    phi = torch.cat([xp, torch.ones_like(xp[:, :1])], 1)
                    G += (phi.T @ phi).cpu().double()
                    Bm += (phi.T @ tz).cpu().double()
                A = _ridge(G, Bm, 1e-3).float().to(device)
                mod = RoutedMLP(mlp, torch.zeros(0, kap, dtype=torch.int16, device=device), vocab_map, tap,
                                P=P, A=A, kap=kap, protect=S)
                idx_table = mod.idx_table
            else:
                score = acc / cnt[:, None].clamp_min(1)
                rare = cnt < min_count  # редкие и неизвестные токены — глобальный набор
                score[rare] = glob / glob.sum() * score.sum(1).mean()
                idx_table = score.topk(kap, dim=-1).indices.to(torch.int16 if I < 32768 else torch.int32)
                mod = RoutedMLP(mlp, idx_table, vocab_map, tap, protect=S)
            if lam_lin is not None:  # линейная поправка на остаток
                G = torch.zeros(d + 1, d + 1, dtype=torch.float64)
                Bm = torch.zeros(d + 1, d, dtype=torch.float64)
                for x, y, ids in pairs():
                    tap.ids = ids
                    xf = x[:, 1:].float().reshape(-1, d)
                    r = (y[:, 1:].float() - mod(x)[:, 1:].float()).reshape(-1, d)
                    xf, r = _robust_resid(xf, r, robust_tau, rstats)
                    phi = torch.cat([xf, torch.ones_like(xf[:, :1])], 1)
                    G += (phi.T @ phi).cpu().double()
                    Bm += (phi.T @ r).cpu().double()
                Wf = _ridge(G, Bm, lam_lin)
                if lin_rank:  # низкоранговая поправка
                    A_, B_ = reduced_rank(Wf, G[:-1, :-1], lin_rank)
                    mod = RoutedMLP(mlp, idx_table, vocab_map, tap, None, Wf[-1].float().to(device),
                                    P=mod.P if mod.P.numel() else None, A=mod.A if mod.A.numel() else None, kap=kap,
                                    Wa=A_.float().to(device), Wb=B_.float().to(device), protect=S, **mod.mode_kwargs())
                else:
                    mod = RoutedMLP(mlp, idx_table, vocab_map, tap, Wf[:-1].T.float().contiguous().to(device),
                                    Wf[-1].float().to(device), P=mod.P if mod.P.numel() else None,
                                    A=mod.A if mod.A.numel() else None, kap=kap, protect=S, **mod.mode_kwargs())
        elif kind == "cluster_mlp":
            p = min(pca_dim, d)
            C = torch.zeros(d, d, dtype=torch.float64)
            xs = []
            for x, y, _ in pairs():
                xf = x[:, 1:].float().reshape(-1, d)
                C += (xf.T @ xf).cpu().double()
                xs.append(xf[:: max(1, xf.shape[0] // 512)].cpu())  # подвыборка для k-means
            P = torch.linalg.eigh(C)[1][:, -p:].flip(-1).float()
            Xs = torch.cat(xs) @ P
            g = torch.Generator().manual_seed(0)
            cent = Xs[torch.randperm(Xs.shape[0], generator=g)[:clusters]].clone()
            for _ in range(kmeans_iters):
                a = torch.cdist(Xs, cent).argmin(1)
                cent = torch.stack([Xs[a == c].mean(0) if (a == c).any() else cent[c] for c in range(clusters)])
            P, cent = P.to(device), cent.to(device)
            G = torch.zeros(clusters, p + 1, p + 1, dtype=torch.float64)
            Bm = torch.zeros(clusters, p + 1, d, dtype=torch.float64)
            for x, y, _ in pairs():
                xp = x[:, 1:].float().reshape(-1, d) @ P
                yf = y[:, 1:].float().reshape(-1, d)
                a = torch.cdist(xp, cent).argmin(1)
                phi = torch.cat([xp, torch.ones_like(xp[:, :1])], 1)
                for c in a.unique().tolist():
                    m = a == c
                    G[c] += (phi[m].T @ phi[m]).cpu().double()
                    Bm[c] += (phi[m].T @ yf[m]).cpu().double()
            Ws = torch.stack([_ridge(G[c], Bm[c], lam) for c in range(clusters)])
            mod = ClusterMLP(P, cent, Ws[:, :-1].float().to(device), Ws[:, -1].float().to(device))
        else:
            raise ValueError(kind)
        mod = ExactPos0(mlp, mod)
        out[i] = mod
        student[i] = _StudentLayer(model.model.layers[i], attn_mods.get(i), mod)
        info[f"layer{i}"] = mod.extra_cost()
        if robust_tau:
            info[f"layer{i}"]["robust_dropped_rows"] = rstats.get("dropped", 0)
            info[f"layer{i}"]["robust_dropped_resid_rows"] = rstats.get("dropped_resid", 0)
            rstats.clear()
    if robust_tau:
        info["robust_tau"] = robust_tau
    return out, info


# ============================================================ диспетчер MLP

def build_mlp(model, method, k, calib, device, bs=4, attn_mods=None, **p):
    """Возвращает ({i: модуль-замена MLP}, info). Позиция 0 везде точная (ExactPos0)."""
    layers = model.model.layers
    # нейроны стока (правило 2) считаются точно во всех калиброванных заменах; у Qwen3 они действуют
    # только на позиции 0 (и так точной), у Llama-семейства — на токенах вроде « the» в любой позиции
    protect = find_sink_neurons(model, calib, k, bs, device) if p.get("protect_sinks", True) and calib is not None else {}
    pinfo = {f"sink_neurons_layer{i}": S.tolist() for i, S in protect.items()}
    if method == "a3_mlp":
        mods, info = build_a3_mlp(model, calib, k, bs, device, p.get("ratio", 0.5), p.get("robust_tau"), protect)
    elif method == "topk_mlp":
        kn = max(1, int(model.config.intermediate_size * p.get("ratio", 0.25)))
        mods, info = {i: TopKMLP(layers[i].mlp, kn) for i in range(k)}, {"active_neurons": kn}
    elif method in ("asvd_mlp", "svdllm_mlp"):
        mods, info = build_svd_mlp(model, calib, k, bs, device, method.split("_")[0], p.get("ratio", 0.5),
                                   p.get("alpha", 0.5), p.get("robust_tau"), protect)
    elif method == "moe_mlp":
        mods, info = build_moe_mlp(model, k, device, p.get("experts", 32), p.get("active_frac", 0.25))
    elif method == "cats_mlp":
        mods, info = build_cats_mlp(model, calib, k, bs, device, p.get("ratio", 0.25))
    elif method in ("ridge_mlp", "table_mlp", "cluster_mlp", "routed_mlp"):
        keys = ("lam", "keep_ratio", "sweeps", "shrink", "clusters", "pca_dim", "kmeans_iters",
                "active_ratio", "min_count", "lam_lin", "predictor", "pred_dim", "robust_tau", "rank", "lin_rank",
                "quad_rank")
        mods, info = build_ours_mlp(model, calib, k, bs, device, method, attn_mods, protect=protect,
                                    **{x: p[x] for x in keys if x in p})
        return mods, {**info, **pinfo}
    else:
        raise ValueError(method)
    return {i: ExactPos0(layers[i].mlp, m) for i, m in mods.items()}, {**info, **pinfo}
