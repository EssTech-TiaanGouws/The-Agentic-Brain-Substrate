from __future__ import annotations

import math
import sys
from dataclasses import asdict, dataclass

from .tokens import BYTE_VOCABULARY_SIZE

STACEY_CORE_ARCHITECTURE_ID = "stacey.byte_transformer_seq2seq.v0"
STACEY_CORE_TOKENIZER_ID = "utf8-byte-v0"


class StaceyCoreConfigurationError(ValueError):
    pass


def _positive_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= sys.maxsize:
        raise StaceyCoreConfigurationError(f"{field_name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class StaceyCoreConfig:
    """Explicit shape/configuration for the untrained Stacey Core v0 hypothesis."""

    architecture_id: str
    tokenizer_id: str
    vocabulary_size: int
    model_dimension: int
    attention_heads: int
    encoder_layers: int
    decoder_layers: int
    feedforward_dimension: int
    maximum_input_tokens: int
    maximum_output_tokens: int
    dropout_probability: float

    def __post_init__(self) -> None:
        for field_name in ("architecture_id", "tokenizer_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise StaceyCoreConfigurationError(f"{field_name} must be a non-empty string")
        if self.architecture_id != STACEY_CORE_ARCHITECTURE_ID:
            raise StaceyCoreConfigurationError("unsupported Stacey Core architecture candidate")
        if self.tokenizer_id != STACEY_CORE_TOKENIZER_ID:
            raise StaceyCoreConfigurationError("unsupported Stacey Core tokenizer protocol")
        for field_name in (
            "vocabulary_size",
            "model_dimension",
            "attention_heads",
            "encoder_layers",
            "decoder_layers",
            "feedforward_dimension",
            "maximum_input_tokens",
            "maximum_output_tokens",
        ):
            _positive_integer(getattr(self, field_name), field_name)
        if self.vocabulary_size != BYTE_VOCABULARY_SIZE:
            raise StaceyCoreConfigurationError(
                f"byte tokenizer vocabulary is fixed at {BYTE_VOCABULARY_SIZE} symbols"
            )
        if self.model_dimension % self.attention_heads:
            raise StaceyCoreConfigurationError("model_dimension must be divisible by attention_heads")
        if self.maximum_input_tokens < 3 or self.maximum_output_tokens < 3:
            raise StaceyCoreConfigurationError("token limits must allow BOS, payload, and EOS")
        if (
            isinstance(self.dropout_probability, bool)
            or not isinstance(self.dropout_probability, (int, float))
            or not math.isfinite(self.dropout_probability)
            or not 0.0 <= self.dropout_probability < 1.0
        ):
            raise StaceyCoreConfigurationError("dropout_probability must be finite and in [0, 1)")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: object) -> StaceyCoreConfig:
        if not isinstance(values, dict):
            raise StaceyCoreConfigurationError("configuration must be a JSON object")
        expected = {
            "architecture_id",
            "tokenizer_id",
            "vocabulary_size",
            "model_dimension",
            "attention_heads",
            "encoder_layers",
            "decoder_layers",
            "feedforward_dimension",
            "maximum_input_tokens",
            "maximum_output_tokens",
            "dropout_probability",
        }
        if set(values) != expected:
            raise StaceyCoreConfigurationError("configuration has missing or unknown fields")
        return cls(**values)