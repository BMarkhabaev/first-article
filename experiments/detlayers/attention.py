"""Замены подслоя самовнимания Qwen3.

Все модули сохраняют сигнатуру Qwen3Attention.forward и возвращают (out, None).
Проекции q/k/v/o, q_norm/k_norm и RoPE берутся из исходного слоя — меняется только
способ смешивания токенов (матрица внимания). Реализации эталонные (материализуют T×T),
цель — измерить качество аппроксимации, а не скорость; асимптотику обсуждаем в статье.

Предположение: каузальная LM без левого паддинга (так работает lm-eval loglikelihood
и наши PPL/KL), KV-кэш не используется.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv


class DetAttention(nn.Module):
    """База: вычисляет q, k, v ровно как исходный Qwen3Attention."""

    method = "base"

    def __init__(self, orig):
        super().__init__()
        self.orig = orig  # исходный модуль (его веса не меняются)
        self.layer_idx = orig.layer_idx
        self.head_dim = orig.head_dim
        self.scaling = orig.scaling
        self.groups = orig.num_key_value_groups
        cfg = orig.config
        self.n_heads = cfg.num_attention_heads

    def qkv(self, hidden_states, position_embeddings):
        o = self.orig
        shape = (*hidden_states.shape[:-1], -1, self.head_dim)
        qn, kn = getattr(o, "q_norm", None), getattr(o, "k_norm", None)  # QK-norm есть у Qwen3, нет у Llama/SmolLM
        q = o.q_proj(hidden_states).view(shape)
        k = o.k_proj(hidden_states).view(shape)
        q = (qn(q) if qn is not None else q).transpose(1, 2)
        k = (kn(k) if kn is not None else k).transpose(1, 2)
        v = o.v_proj(hidden_states).view(shape).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        return q, repeat_kv(k, self.groups), repeat_kv(v, self.groups)

    def mix(self, q, k, v):  # (B,H,T,D) -> (B,H,T,D)
        raise NotImplementedError

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_values=None, **kwargs):
        q, k, v = self.qkv(hidden_states, position_embeddings)
        out = self.mix(q, k, v).to(hidden_states.dtype)
        out = out.transpose(1, 2).reshape(*hidden_states.shape[:-1], -1)
        return self.orig.o_proj(out), None

    def extra_cost(self) -> dict:
        """Описание стоимости для таблицы (асимптотика, доп. параметры)."""
        return {}


def causal_scores(q, k, scaling):
    s = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scaling
    T = q.shape[-2]
    mask = torch.ones(T, T, dtype=torch.bool, device=q.device).tril_()
    return s, mask


def exact_attention_weights(q, k, scaling):
    s, mask = causal_scores(q, k, scaling)
    return torch.softmax(s.masked_fill(~mask, float("-inf")), dim=-1)


# ============================================================ L1/L2: разреженные шаблоны

def sparse_pattern(T: int, window: int, n_sink: int, stride: int | None, device) -> torch.Tensor:
    """Bool-маска допустимых (t, j): j ≤ t и (t-j < window или j < n_sink
    или (stride и (t-j) % stride == 0))."""
    t = torch.arange(T, device=device).unsqueeze(1)
    j = torch.arange(T, device=device).unsqueeze(0)
    d = t - j
    allowed = (d >= 0) & ((d < window) | (j < n_sink))
    if stride:
        allowed |= (d >= 0) & (d % stride == 0)
    return allowed


class SparseAttention(DetAttention):
    """L1 (окно + sink; Sparse Transformer / BigBird без случайных рёбер)
    и L2 (strided-шаблон Sparse Transformer). Точный softmax на разреженном носителе."""

    method = "sparse"

    def __init__(self, orig, window: int = 64, n_sink: int = 1, stride: int | None = None,
                 sink_ratio: float | None = None):
        super().__init__(orig)
        self.window, self.n_sink, self.stride, self.sink_ratio = window, n_sink, stride, sink_ratio
        self._dyn = None  # (B,T) ключи-стоки по массивности, см. forward
        self.dyn_frac = []  # доля таких ключей (для отчёта)

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kw):
        if self.sink_ratio:
            # Стоки по массивной активации, а не только по позиции 0: у Llama-семейства массивная
            # активация живёт на токенах вроде « the» в любой позиции. После RMSNorm норма не видна,
            # но отношение пика к среднему по каналам у таких токенов в разы выше медианы по последовательности.
            x = hidden_states.float()
            r = x.abs().amax(-1) / x.abs().mean(-1).clamp_min(1e-6)  # (B,T)
            self._dyn = r > self.sink_ratio * r.median(dim=-1, keepdim=True).values
            self.dyn_frac.append(self._dyn.float().mean().item())
        return super().forward(hidden_states, position_embeddings, attention_mask, past_key_values, **kw)

    def mix(self, q, k, v):
        s = torch.matmul(q.float(), k.float().transpose(-1, -2)) * self.scaling
        allowed = sparse_pattern(q.shape[-2], self.window, self.n_sink, self.stride, q.device)
        if self._dyn is not None:
            causal = torch.ones_like(allowed).tril_()
            allowed = allowed | (self._dyn[:, None, None, :] & causal)
        a = torch.softmax(s.masked_fill(~allowed, float("-inf")), dim=-1)
        return torch.matmul(a, v.float())

    def extra_cost(self):
        c = f"O(T·(w+s))" if not self.stride else "O(T·(w+T/s))"
        d = {"complexity": c, "window": self.window, "n_sink": self.n_sink, "stride": self.stride}
        if self.sink_ratio:
            d.update(sink_ratio=self.sink_ratio, dyn_sink_frac=round(sum(self.dyn_frac) / max(1, len(self.dyn_frac)), 4))
        return d


# ============================================================ L3/L4: ядерные методы

def _kernel_attention(phi_q, phi_k, v):
    """Каузальное ядерное внимание: (φ(Q)φ(K)^T ⊙ M) V / (φ(Q)φ(K)^T ⊙ M) 1.
    Эквивалентно рекуррентной (prefix-sum) форме, но экономнее по памяти при eval."""
    a = torch.matmul(phi_q, phi_k.transpose(-1, -2))
    T = a.shape[-1]
    a = a * torch.ones(T, T, device=a.device).tril_()
    num = torch.matmul(a, v.float())
    den = a.sum(-1, keepdim=True).clamp_min(1e-6)
    return num / den


def orthogonal_gaussian(m: int, d: int, gen: torch.Generator) -> torch.Tensor:
    """Блочно-ортогональные гауссовы направления с χ-нормами (как в FAVOR+)."""
    blocks = []
    for _ in range(math.ceil(m / d)):
        g = torch.randn(d, d, generator=gen)
        qmat, _ = torch.linalg.qr(g)
        blocks.append(qmat.T)
    w = torch.cat(blocks, 0)[:m]
    norms = torch.randn(m, d, generator=gen).norm(dim=1, keepdim=True)
    return w * norms


class PerformerAttention(DetAttention):
    """L3: FAVOR+ (положительные ортогональные случайные признаки), Choromanski et al.
    СТОХАСТИЧЕСКИЙ контроль: признаки зависят от seed."""

    method = "performer"

    def __init__(self, orig, n_features: int = 256, seed: int = 0):
        super().__init__(orig)
        gen = torch.Generator().manual_seed(seed * 1000 + orig.layer_idx)
        w = torch.stack([orthogonal_gaussian(n_features, self.head_dim, gen) for _ in range(self.n_heads)])
        self.register_buffer("w", w)  # (H, m, D)
        self.m = n_features

    def _phi(self, x, is_query):
        x = x.float() * self.scaling ** 0.5  # x'ᵀy' = xᵀy·scaling
        proj = torch.einsum("bhtd,hmd->bhtm", x, self.w.to(x.device).float())
        sq = (x ** 2).sum(-1, keepdim=True) / 2
        z = proj - sq
        # стабилизатор: константа по ключам/признакам сокращается при нормировке
        stab = z.amax(dim=-1, keepdim=True) if is_query else z.amax(dim=(-1, -2), keepdim=True)
        return torch.exp(z - stab) / math.sqrt(self.m)

    def mix(self, q, k, v):
        return _kernel_attention(self._phi(q, True), self._phi(k, False), v)

    def extra_cost(self):
        return {"complexity": "O(T·m·D)", "features": self.m, "stochastic": True}


class LinearEluAttention(DetAttention):
    """L4: линейное внимание φ(x)=elu(x)+1 (Katharopoulos et al.)."""

    method = "linear_elu"

    def mix(self, q, k, v):
        return _kernel_attention(F.elu(q.float()) + 1, F.elu(k.float()) + 1, v)

    def extra_cost(self):
        return {"complexity": "O(T·D²)"}


# ============================================================ L5: фиксированное позиционное внимание

class RelPosStats:
    """Накопитель статистики внимания по относительным сдвигам (калибровка L5/O1)."""

    def __init__(self, n_heads: int, max_offset: int):
        self.D = max_offset
        self.kernel = torch.zeros(n_heads, max_offset, dtype=torch.float64)
        self.sink = torch.zeros(n_heads, dtype=torch.float64)
        self.count = 0

    @torch.no_grad()
    def update(self, a: torch.Tensor):
        """a: (B,H,T,T) точные веса внимания. Берём запросы t ≥ D, где все сдвиги
        < D доступны и не пересекаются с позицией 0."""
        B, H, T, _ = a.shape
        D = self.D
        if T <= D + 1:
            return
        t = torch.arange(D + 1, T, device=a.device)
        self.sink += a[:, :, t, 0].float().sum(dim=(0, 2)).cpu().double()
        for d in range(D):
            self.kernel[:, d] += a[:, :, t, t - d].float().sum(dim=(0, 2)).cpu().double()
        self.count += B * t.numel()

    def finalize(self):
        return (self.kernel / self.count).float(), (self.sink / self.count).float()


def relpos_matrix(kernel: torch.Tensor, sink: torch.Tensor, T: int) -> torch.Tensor:
    """Строит (H,T,T) каузальную матрицу из ядра по сдвигам, веса sink для j=0 и
    равномерного «хвоста» для сдвигов ≥ D. Строки перенормируются.
    Экономно по памяти: один тензор H×T×T, остальное — на месте."""
    H, D = kernel.shape
    dev = kernel.device
    t = torch.arange(T, device=dev)
    d = t[:, None] - t[None, :]  # (T,T)
    tail_mass = (1 - kernel.sum(1) - sink).clamp_min(0)  # (H,)
    n_tail = (t - D).clamp_min(1).float()  # число позиций 1..t-D (≥1, чтобы не делить на 0)
    # таблица значений по сдвигу: 0..D-1 — ядро, D — «хвост» (делится на n_tail по строке), D+1 — ноль
    table = torch.cat([kernel, tail_mass[:, None], torch.zeros(H, 1, device=dev)], dim=1)  # (H, D+2)
    idx = torch.where(d < 0, D + 1, d.clamp(max=D))  # (T,T)
    w = table[:, idx]  # (H,T,T) — единственная большая аллокация
    tail = (d >= D)
    w[:, tail] /= n_tail[:, None].expand(T, T)[tail]
    w[:, :, 0] = sink[:, None]  # j=0 — всегда sink (t ≥ 0)
    w /= w.sum(-1, keepdim=True).clamp_min(1e-9)
    return w


class RelPosAttention(DetAttention):
    """L5: Synthesizer-подобное фиксированное внимание (Tay et al.): веса зависят только
    от относительной позиции и оценены усреднением точного внимания на калибровке.
    q, k не используются; V и O — исходные."""

    method = "relpos"

    def __init__(self, orig, kernel: torch.Tensor, sink: torch.Tensor):
        super().__init__(orig)
        self.register_buffer("kernel", kernel)
        self.register_buffer("sink", sink)
        self._cache = {}

    def weights(self, T, device):
        key = (T, str(device))
        if self.kernel.requires_grad:  # дистилляция: без кэша
            return relpos_matrix(self.kernel.float(), self.sink.float(), T)
        if key not in self._cache:
            self._cache = {key: relpos_matrix(self.kernel.to(device).float(), self.sink.to(device).float(), T)}
        return self._cache[key]

    def mix(self, q, k, v):
        w = self.weights(q.shape[-2], q.device)
        return torch.matmul(w.unsqueeze(0), v.float())

    def extra_cost(self):
        return {"complexity": "O(T·D) (свёртка по позиции)", "max_offset": self.kernel.shape[1]}


# ============================================================ O1: типизированная замена голов

class HeadTypedAttention(DetAttention):
    """O1 (наш): головы с хорошо аппроксимируемым позиционным профилем заменяются
    фиксированным ядром (как в L5), остальные остаются точными (`content='exact'`)
    или ограничиваются окном + sink (`content='window'`)."""

    method = "head_typed"

    def __init__(self, orig, kernel, sink, replace_mask: torch.Tensor, content: str = "exact",
                 window: int = 64):
        super().__init__(orig)
        self.register_buffer("kernel", kernel)
        self.register_buffer("sink", sink)
        self.register_buffer("replace_mask", replace_mask.bool())  # (H,)
        self.content, self.window = content, window
        self._cache = {}

    def mix(self, q, k, v):
        T = q.shape[-2]
        s = torch.matmul(q.float(), k.float().transpose(-1, -2)) * self.scaling
        if self.content == "window":
            allowed = sparse_pattern(T, self.window, 1, None, q.device)
        else:
            allowed = torch.ones(T, T, dtype=torch.bool, device=q.device).tril_()
        a = torch.softmax(s.masked_fill(~allowed, float("-inf")), dim=-1)
        key = (T, str(q.device))
        if self.kernel.requires_grad:
            fixed = relpos_matrix(self.kernel.float(), self.sink.float(), T).unsqueeze(0)
        else:
            if key not in self._cache:
                self._cache = {key: relpos_matrix(self.kernel.to(q.device).float(), self.sink.to(q.device).float(), T)}
            fixed = self._cache[key].unsqueeze(0)
        m = self.replace_mask.to(q.device)[None, :, None, None]
        a = torch.where(m, fixed, a)
        return torch.matmul(a, v.float())

    def extra_cost(self):
        return {"replaced_heads": int(self.replace_mask.sum()), "n_heads": self.n_heads,
                "content": self.content}


@torch.no_grad()
def head_fit_scores(q, k, kernel, sink, scaling) -> torch.Tensor:
    """Средняя TV-дистанция между точными строками внимания и фиксированным ядром, по головам."""
    a = exact_attention_weights(q, k, scaling)  # (B,H,T,T)
    fixed = relpos_matrix(kernel.to(a.device), sink.to(a.device), a.shape[-1]).unsqueeze(0)
    tv = 0.5 * (a - fixed).abs().sum(-1)  # (B,H,T)
    return tv.mean(dim=(0, 2)).cpu()


# ============================================================ O4: ядро × значимость ключевого токена

def salience_matrix(kernel, sink, beta_keys, T):
    """(B,H,T,T): w_tj ∝ K_h(t-j)·exp(β_h[x_j]) при j ≥ 1, w_t0 = sink_h (позиция-сток).
    beta_keys: (B,H,T) — β_h для токена на позиции j."""
    base = relpos_matrix(kernel, sink, T)  # (H,T,T), уже нормирована по строкам
    w = base.unsqueeze(0) * torch.exp(beta_keys).unsqueeze(2)  # масштабируем столбцы j
    j0 = torch.zeros(T, dtype=torch.bool, device=w.device)
    j0[0] = True
    w = torch.where(j0, base.unsqueeze(0).expand_as(w), w)  # sink без β
    return w / w.sum(-1, keepdim=True).clamp_min(1e-12)


class SalienceAttention(DetAttention):
    """O4 (наш): a_tj ∝ K_h(t−j)·exp(β_h(x_j)). Ядро K_h и sink — как в L5, таблица β_h
    по токенам подгоняется в замкнутой форме (итеративное согласование масс внимания).
    q·k не вычисляется. Головы из `exact_mask` остаются точными (гибрид с O1)."""

    method = "salience"

    def __init__(self, orig, kernel, sink, beta, vocab_map, tap, exact_mask=None, window=None):
        super().__init__(orig)
        self.register_buffer("kernel", kernel)
        self.register_buffer("sink", sink)
        self.register_buffer("beta", beta)  # (H, V'+1)
        self.register_buffer("vocab_map", vocab_map)
        self.tap = tap
        em = exact_mask if exact_mask is not None else torch.zeros(self.n_heads, dtype=torch.bool)
        self.register_buffer("exact_mask", em.bool())
        self.window = window

    def fixed_weights(self, ids, T, device):
        rows = self.vocab_map[ids.to(device)]  # (B,T)
        bk = self.beta[:, rows].permute(1, 0, 2).float()  # (B,H,T)
        return salience_matrix(self.kernel.to(device).float(), self.sink.to(device).float(), bk, T)

    def mix(self, q, k, v):
        T = q.shape[-2]
        a = self.fixed_weights(self.tap.ids[:, :T], T, q.device)
        if self.exact_mask.any():
            s = torch.matmul(q.float(), k.float().transpose(-1, -2)) * self.scaling
            if self.window:
                allowed = sparse_pattern(T, self.window, 1, None, q.device)
            else:
                allowed = torch.ones(T, T, dtype=torch.bool, device=q.device).tril_()
            ex = torch.softmax(s.masked_fill(~allowed, float("-inf")), dim=-1)
            a = torch.where(self.exact_mask[None, :, None, None].to(q.device), ex, a)
        return torch.matmul(a, v.float())

    def extra_cost(self):
        return {"complexity": "O(T·D) + таблица β", "beta_rows": self.beta.shape[1],
                "exact_heads": int(self.exact_mask.sum())}


# ============================================================ O5: окно + детерминированный хвост

class WindowTailAttention(DetAttention):
    """O5 (наш): внутри окна (t−j < w) и на позиции-стоке j=0 — точные оценки q·k;
    для дальних ключей оценка заменяется таблицей по токену β_h(x_j) (log-mean-exp точных
    оценок на этот токен по калибровке). Логиты L_tj = s_tj (окно, сток), β_h(x_j) (хвост),
    a = softmax_j(L). Хвост считается префиксными суммами: O(T·w·D + T·D)."""

    method = "window_tail"

    def __init__(self, orig, window, beta, vocab_map, tap):
        super().__init__(orig)
        self.window = window
        self.register_buffer("beta", beta)  # (H, V'+1)
        self.register_buffer("vocab_map", vocab_map)
        self.tap = tap

    def mix(self, q, k, v):
        T = q.shape[-2]
        s = torch.matmul(q.float(), k.float().transpose(-1, -2)) * self.scaling  # (B,H,T,T)
        t = torch.arange(T, device=q.device)
        d = t[:, None] - t[None, :]
        exact = (d >= 0) & ((d < self.window) | (t[None, :] == 0))
        tail = (d >= self.window) & (t[None, :] > 0)
        rows = self.vocab_map[self.tap.ids[:, :T].to(q.device)]  # (B,T)
        bk = self.beta.float()[:, rows].permute(1, 0, 2)  # (B,H,T) — по ключу j
        logits = torch.where(exact, s, torch.where(tail, bk.unsqueeze(2).expand_as(s),
                                                   torch.full_like(s, float("-inf"))))
        return torch.matmul(torch.softmax(logits, dim=-1), v.float())

    def extra_cost(self):
        return {"complexity": "O(T·w·D) + O(T·D)", "window": self.window, "beta_rows": self.beta.shape[1]}
