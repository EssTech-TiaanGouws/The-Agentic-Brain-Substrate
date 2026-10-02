from __future__ import annotations


PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
BYTE_TOKEN_OFFSET = 3
BYTE_VOCABULARY_SIZE = BYTE_TOKEN_OFFSET + 256


class ByteTokenCodec:
    """Stable UTF-8 byte tokenizer; it has no learned vocabulary or external assets."""

    tokenizer_id = "utf8-byte-v0"
    vocabulary_size = BYTE_VOCABULARY_SIZE
    pad_token_id = PAD_TOKEN_ID
    bos_token_id = BOS_TOKEN_ID
    eos_token_id = EOS_TOKEN_ID

    def encode(self, text: str) -> tuple[int, ...]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text:
            raise ValueError("text must not be empty")
        return (BOS_TOKEN_ID, *(byte + BYTE_TOKEN_OFFSET for byte in text.encode("utf-8")), EOS_TOKEN_ID)

    def decode(self, token_ids: tuple[int, ...] | list[int], *, strip_special: bool = True) -> str:
        if not isinstance(token_ids, (tuple, list)):
            raise TypeError("token_ids must be a tuple or list")
        values = list(token_ids)
        if strip_special:
            while values and values[0] in (PAD_TOKEN_ID, BOS_TOKEN_ID):
                values.pop(0)
            while values and values[-1] in (PAD_TOKEN_ID, EOS_TOKEN_ID):
                values.pop()
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not BYTE_TOKEN_OFFSET <= value < BYTE_VOCABULARY_SIZE
            for value in values
        ):
            raise ValueError("token sequence contains a non-byte token after special-token removal")
        return bytes(value - BYTE_TOKEN_OFFSET for value in values).decode("utf-8", errors="strict")