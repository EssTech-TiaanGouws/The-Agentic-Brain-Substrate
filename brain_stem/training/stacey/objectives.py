from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor

from models.stacey.core.tokens import PAD_TOKEN_ID


class StaceyTrainingError(ValueError):
    pass


def teacher_forcing_loss(logits: Tensor, target_token_ids: Tensor, *, padding_token_id: int = PAD_TOKEN_ID) -> Tensor:
    """Cross-entropy for next-token structured-envelope training targets."""
    if not isinstance(logits, Tensor) or logits.ndim != 3:
        raise StaceyTrainingError("logits must have shape [batch, target_length, vocabulary]")
    if not isinstance(target_token_ids, Tensor) or target_token_ids.ndim != 2:
        raise StaceyTrainingError("target_token_ids must have shape [batch, target_length]")
    if target_token_ids.dtype != torch.long:
        raise StaceyTrainingError("target_token_ids must use torch.long")
    if logits.shape[:2] != target_token_ids.shape:
        raise StaceyTrainingError("logit batch/sequence dimensions must match targets")
    if logits.shape[2] <= 1:
        raise StaceyTrainingError("vocabulary dimension is invalid")
    if isinstance(padding_token_id, bool) or not isinstance(padding_token_id, int):
        raise StaceyTrainingError("padding_token_id must be an integer")
    if not 0 <= padding_token_id < logits.shape[2]:
        raise StaceyTrainingError("padding_token_id is outside the output vocabulary")
    valid_targets = target_token_ids.ne(padding_token_id)
    if not bool(valid_targets.any()):
        raise StaceyTrainingError("batch contains no non-padding target tokens")
    if target_token_ids.numel() and (
        int(target_token_ids.min()) < 0 or int(target_token_ids.max()) >= logits.shape[2]
    ):
        raise StaceyTrainingError("target token ID is outside the output vocabulary")
    if not bool(torch.isfinite(logits).all()):
        raise StaceyTrainingError("logits contain non-finite values")

    loss = functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target_token_ids.reshape(-1),
        ignore_index=padding_token_id,
    )
    if not bool(torch.isfinite(loss)):
        raise StaceyTrainingError("teacher-forcing loss is non-finite")
    return loss