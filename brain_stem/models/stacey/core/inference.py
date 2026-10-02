from __future__ import annotations

import math
import re

import torch

from .inputs import UnifiedContextIngress
from .network import StaceyCore
from .outputs import (
    CoreDecisionEnvelope,
    enforce_clarification_threshold,
    parse_decision_jsonl,
)


class StaceyInferenceError(RuntimeError):
    pass


class StaceyCoreInferenceAdapter:
    """Bridge one Core ingress to a validated decision; performs no dispatch."""

    def __init__(
        self,
        model: StaceyCore,
        *,
        candidate_id: str,
        artifact_sha256: str,
        minimum_confidence: float,
        maximum_new_tokens: int | None = None,
    ) -> None:
        if not isinstance(model, StaceyCore):
            raise TypeError("model must be a StaceyCore")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError("candidate_id must be a non-empty string")
        if not isinstance(artifact_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", artifact_sha256) is None:
            raise ValueError("artifact_sha256 must be a lowercase SHA-256 digest")
        if (
            isinstance(minimum_confidence, bool)
            or not isinstance(minimum_confidence, (int, float))
            or not math.isfinite(minimum_confidence)
            or not 0.0 <= minimum_confidence <= 1.0
        ):
            raise ValueError("minimum_confidence must be a finite value in [0, 1]")
        if maximum_new_tokens is None:
            maximum_new_tokens = model.config.maximum_output_tokens
        if (
            isinstance(maximum_new_tokens, bool)
            or not isinstance(maximum_new_tokens, int)
            or not 1 <= maximum_new_tokens <= model.config.maximum_output_tokens
        ):
            raise ValueError("maximum_new_tokens must fit the configured output limit")

        self.model = model
        self.candidate_id = candidate_id
        self.artifact_sha256 = artifact_sha256
        self.minimum_confidence = float(minimum_confidence)
        self.maximum_new_tokens = maximum_new_tokens

    def decide(self, ingress: UnifiedContextIngress) -> CoreDecisionEnvelope:
        if not isinstance(ingress, UnifiedContextIngress):
            raise StaceyInferenceError("ingress must be a UnifiedContextIngress")
        try:
            model_device = next(self.model.parameters()).device
            source_token_ids = self.model.encode_ingresses((ingress,), device=model_device)
            generated_token_ids = self.model.generate_token_ids(
                source_token_ids,
                maximum_new_tokens=self.maximum_new_tokens,
            )
            if (
                not isinstance(generated_token_ids, torch.Tensor)
                or generated_token_ids.ndim != 2
                or generated_token_ids.shape[0] != 1
                or generated_token_ids.dtype != torch.long
                or not 1 <= generated_token_ids.shape[1] <= self.maximum_new_tokens
            ):
                raise StaceyInferenceError("Core generation returned an invalid token tensor")

            token_ids = tuple(int(token_id) for token_id in generated_token_ids[0].tolist())
            decision_line = self.model.codec.decode(token_ids)
            decision = parse_decision_jsonl(decision_line, ingress)
            enforce_clarification_threshold(
                decision,
                minimum_confidence=self.minimum_confidence,
            )
            return decision
        except StaceyInferenceError:
            raise
        except Exception as error:
            raise StaceyInferenceError("Core generation did not produce an admissible decision") from error