from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from models.stacey.core.network import StaceyCore
from .objectives import teacher_forcing_loss


@dataclass(frozen=True, slots=True)
class SmokeStepReport:
    loss: float
    gradient_norm: float
    trainable_parameter_count: int
    updated_parameter_count: int


def run_synthetic_smoke_step(
    model: StaceyCore,
    ingress_token_ids: torch.Tensor,
    target_token_ids: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> SmokeStepReport:
    """Run exactly one caller-supplied synthetic step; no dataset or model is loaded."""
    if not isinstance(model, StaceyCore):
        raise TypeError("model must be a StaceyCore")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer")
    if not isinstance(ingress_token_ids, torch.Tensor) or ingress_token_ids.ndim != 2:
        raise ValueError("ingress_token_ids must have shape [batch, input_length]")
    if not isinstance(target_token_ids, torch.Tensor) or target_token_ids.ndim != 2:
        raise ValueError("target_token_ids must have shape [batch, target_length]")
    if ingress_token_ids.shape[0] != target_token_ids.shape[0]:
        raise ValueError("input and target batch sizes must match")
    if target_token_ids.shape[1] < 2:
        raise ValueError("teacher-forcing target requires BOS and at least one next-token target")
    if target_token_ids.dtype != torch.long:
        raise ValueError("target_token_ids must use torch.long")

    optimizer.zero_grad(set_to_none=True)
    decoder_inputs = target_token_ids[:, :-1]
    labels = target_token_ids[:, 1:]
    logits = model(ingress_token_ids, decoder_inputs)
    loss = teacher_forcing_loss(logits, labels)
    loss.backward()

    squared_norm = 0.0
    trainable_count = 0
    gradients_found = False
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        trainable_count += parameter.numel()
        if parameter.grad is None:
            continue
        gradients_found = True
        if not bool(torch.isfinite(parameter.grad).all()):
            optimizer.zero_grad(set_to_none=True)
            raise RuntimeError("smoke step produced non-finite gradients")
        squared_norm += float(torch.sum(parameter.grad.detach() ** 2))
    if not gradients_found:
        optimizer.zero_grad(set_to_none=True)
        raise RuntimeError("smoke step produced no gradients")

    before = [parameter.detach().clone() for parameter in model.parameters() if parameter.requires_grad]
    optimizer.step()
    updated_count = sum(
        1
        for old_value, parameter in zip(before, (p for p in model.parameters() if p.requires_grad))
        if not torch.equal(old_value, parameter.detach())
    )
    gradient_norm = math.sqrt(squared_norm)
    if not math.isfinite(gradient_norm):
        raise RuntimeError("smoke step produced a non-finite gradient norm")
    return SmokeStepReport(
        loss=float(loss.detach()),
        gradient_norm=gradient_norm,
        trainable_parameter_count=trainable_count,
        updated_parameter_count=updated_count,
    )