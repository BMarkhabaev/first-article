"""Дистилляционный трек (НЕ детерминированный, помечается отдельно в таблицах).

Заменённые модули дообучаются AdamW так, чтобы остаточный поток после первых k слоёв
совпадал с учителем: loss = mean_t ||h_S,k − h_T,k||² / ||h_T,k||² (позиция 0 исключена).
Нужен проход только по k слоям, поэтому дёшево.

Обучаемым становится:
* все вещественные буферы модулей-замен (ridge-карты, факторы SVD, таблицы, ядра, β);
* для замен внимания, использующих исходные проекции (DetAttention.orig), — их копия
  (учитель не меняется).
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .attention import DetAttention
from .common import prefix_hidden

_SKIP = {"vocab_map", "qk_idx", "assign", "replace_mask", "exact_mask"}


def _make_trainable(mod: nn.Module) -> list[nn.Parameter]:
    params = []
    for sub in mod.modules():
        for name, buf in list(sub._buffers.items()):
            if buf is None or name in _SKIP or not buf.is_floating_point():
                continue
            del sub._buffers[name]
            p = nn.Parameter(buf.detach().clone().float())
            setattr(sub, name, p)
            params.append(p)
    return params


def prepare(rep) -> list[nn.Parameter]:
    params = []
    for key, mod in rep.new.items():
        if isinstance(mod, DetAttention):
            mod.orig = copy.deepcopy(mod.orig)  # своя копия q/k/v/o
            for p in mod.orig.parameters():
                p.requires_grad_(True)
                params.append(p)
        params += _make_trainable(mod)
    # параметры, общие с учителем (ExactPos0.orig, TopK.mlp и т.п.), остаются замороженными
    return params


def distill(model, rep, calib: torch.Tensor, k: int, device, steps: int = 500, lr: float = 1e-4,
            bs: int = 8, log_every: int = 50) -> dict:
    for p in model.parameters():
        p.requires_grad_(False)
    params = prepare(rep)
    if not params:
        return {"steps": 0, "note": "нет обучаемых параметров"}
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(0)
    hist = []
    n_params = sum(p.numel() for p in params)
    for step in range(steps):
        idx = torch.randint(calib.shape[0], (bs,), generator=g)
        ids = calib[idx].to(device)
        for mod in rep.new.values():  # модули с таблицами по токенам (salience, n-граммы) ждут ids через tap
            if getattr(mod, "tap", None) is not None:
                mod.tap.ids = ids
        with torch.no_grad(), rep.teacher():
            hT = prefix_hidden(model, ids, k)[:, 1:].float()
        rep.enable()
        with torch.enable_grad():
            hS = prefix_hidden.__wrapped__(model, ids, k)[:, 1:].float()
            loss = (((hS - hT) ** 2).sum(-1) / (hT ** 2).sum(-1).clamp_min(1e-6)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
        if step % log_every == 0 or step == steps - 1:
            hist.append((step, round(loss.item(), 5)))
            print(f"distill step {step} loss {loss.item():.5f}", flush=True)
    for p in params:
        p.requires_grad_(False)
    return {"steps": steps, "lr": lr, "bs": bs, "trainable_params": n_params, "loss_curve": hist}
