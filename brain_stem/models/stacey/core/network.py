from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .config import StaceyCoreConfig
from .inputs import UnifiedContextIngress
from .tokens import BOS_TOKEN_ID, EOS_TOKEN_ID, PAD_TOKEN_ID, ByteTokenCodec


class StaceyCoreInputError(ValueError):
    pass


class StaceyCore(nn.Module):
    """Untrained encoder-decoder candidate for structured Stacey Core decisions.

    This is a small, configurable architecture hypothesis used to validate tensor
    and training plumbing. It is not the final World Model Core design and has no
    learned capability until trained and evaluated.
    """

    def __init__(self, config: StaceyCoreConfig) -> None:
        super().__init__()
        if not isinstance(config, StaceyCoreConfig):
            raise TypeError("config must be a StaceyCoreConfig")
        self.config = config
        self.codec = ByteTokenCodec()
        self.token_embedding = nn.Embedding(
            config.vocabulary_size,
            config.model_dimension,
            padding_idx=PAD_TOKEN_ID,
        )
        self.encoder_position_embedding = nn.Embedding(config.maximum_input_tokens, config.model_dimension)
        self.decoder_position_embedding = nn.Embedding(config.maximum_output_tokens, config.model_dimension)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model_dimension,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_dimension,
            dropout=config.dropout_probability,
            batch_first=True,
            norm_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.model_dimension,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_dimension,
            dropout=config.dropout_probability,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.encoder_layers,
            enable_nested_tensor=False,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.decoder_layers)
        self.output_projection = nn.Linear(config.model_dimension, config.vocabulary_size)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def raw_weight_storage_bytes(self, *, bits_per_parameter: int) -> int:
        if (
            isinstance(bits_per_parameter, bool)
            or not isinstance(bits_per_parameter, int)
            or bits_per_parameter <= 0
        ):
            raise ValueError("bits_per_parameter must be a positive integer")
        return (self.parameter_count * bits_per_parameter + 7) // 8

    def encode_ingresses(
        self,
        ingresses: tuple[UnifiedContextIngress, ...],
        *,
        device: torch.device | str | None = None,
    ) -> Tensor:
        if not isinstance(ingresses, tuple) or not ingresses:
            raise StaceyCoreInputError("ingresses must be a non-empty tuple")
        token_rows: list[tuple[int, ...]] = []
        for ingress in ingresses:
            if not isinstance(ingress, UnifiedContextIngress):
                raise StaceyCoreInputError("each ingress must be a UnifiedContextIngress")
            tokens = self.codec.encode(ingress.to_canonical_json())
            if len(tokens) > self.config.maximum_input_tokens:
                raise StaceyCoreInputError("serialized ingress exceeds maximum_input_tokens")
            token_rows.append(tokens)

        padded = torch.full(
            (len(token_rows), max(map(len, token_rows))),
            PAD_TOKEN_ID,
            dtype=torch.long,
            device=device,
        )
        for row_index, tokens in enumerate(token_rows):
            padded[row_index, : len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)
        return padded

    def forward(self, source_token_ids: Tensor, decoder_input_ids: Tensor) -> Tensor:
        self._validate_token_tensor(source_token_ids, "source_token_ids", self.config.maximum_input_tokens)
        self._validate_token_tensor(decoder_input_ids, "decoder_input_ids", self.config.maximum_output_tokens)
        if source_token_ids.shape[0] != decoder_input_ids.shape[0]:
            raise StaceyCoreInputError("source and decoder batch dimensions must match")

        source_positions = torch.arange(source_token_ids.shape[1], device=source_token_ids.device)
        decoder_positions = torch.arange(decoder_input_ids.shape[1], device=decoder_input_ids.device)
        source = self.token_embedding(source_token_ids) + self.encoder_position_embedding(source_positions)
        target = self.token_embedding(decoder_input_ids) + self.decoder_position_embedding(decoder_positions)
        source_padding_mask = source_token_ids.eq(PAD_TOKEN_ID)
        target_padding_mask = decoder_input_ids.eq(PAD_TOKEN_ID)
        causal_mask = torch.triu(
            torch.ones(
                (decoder_input_ids.shape[1], decoder_input_ids.shape[1]),
                dtype=torch.bool,
                device=decoder_input_ids.device,
            ),
            diagonal=1,
        )

        memory = self.encoder(source, src_key_padding_mask=source_padding_mask)
        decoded = self.decoder(
            tgt=target,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=target_padding_mask,
            memory_key_padding_mask=source_padding_mask,
        )
        logits = self.output_projection(decoded)
        if not torch.isfinite(logits).all():
            raise StaceyCoreInputError("Core forward pass produced non-finite logits")
        return logits

    @torch.no_grad()
    def generate_token_ids(self, source_token_ids: Tensor, *, maximum_new_tokens: int) -> Tensor:
        self._validate_token_tensor(source_token_ids, "source_token_ids", self.config.maximum_input_tokens)
        if isinstance(maximum_new_tokens, bool) or not isinstance(maximum_new_tokens, int):
            raise StaceyCoreInputError("maximum_new_tokens must be an integer")
        if not 1 <= maximum_new_tokens <= self.config.maximum_output_tokens:
            raise StaceyCoreInputError("maximum_new_tokens exceeds the configured output limit")

        was_training = self.training
        self.eval()
        try:
            generated = torch.full(
                (source_token_ids.shape[0], 1),
                BOS_TOKEN_ID,
                dtype=torch.long,
                device=source_token_ids.device,
            )
            finished = torch.zeros(source_token_ids.shape[0], dtype=torch.bool, device=source_token_ids.device)
            for _ in range(maximum_new_tokens):
                next_token = self.forward(source_token_ids, generated)[:, -1, :].argmax(dim=-1)
                next_token = torch.where(finished, torch.full_like(next_token, PAD_TOKEN_ID), next_token)
                generated = torch.cat((generated, next_token.unsqueeze(1)), dim=1)
                finished |= next_token.eq(EOS_TOKEN_ID)
                if bool(finished.all()):
                    break
            return generated[:, 1:]
        finally:
            self.train(was_training)

    def _validate_token_tensor(self, token_ids: Tensor, name: str, maximum_length: int) -> None:
        if not isinstance(token_ids, Tensor) or token_ids.ndim != 2 or token_ids.shape[0] < 1:
            raise StaceyCoreInputError(f"{name} must be a rank-2 non-empty batch tensor")
        if token_ids.dtype != torch.long:
            raise StaceyCoreInputError(f"{name} must use torch.long token IDs")
        if not 1 <= token_ids.shape[1] <= maximum_length:
            raise StaceyCoreInputError(f"{name} length is outside the configured limit")
        if token_ids.numel() and (int(token_ids.min()) < 0 or int(token_ids.max()) >= self.config.vocabulary_size):
            raise StaceyCoreInputError(f"{name} contains token IDs outside the vocabulary")