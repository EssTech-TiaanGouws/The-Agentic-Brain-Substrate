from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping, Protocol


class DocumentFormatError(ValueError):
    pass


class DuplicateFormatHandlerError(DocumentFormatError):
    pass


def _nonempty_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DocumentFormatError(f"{field_name} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class DocumentPayload:
    format_key: str
    content: bytes

    def __post_init__(self) -> None:
        _nonempty_text(self.format_key, "format_key")
        if not isinstance(self.content, bytes):
            raise DocumentFormatError("Document content must be bytes")


@dataclass(frozen=True, slots=True)
class PreparedDocument:
    format_key: str
    content: bytes
    digest: str
    verification_ref: str


class IDocumentFormatHandler(Protocol):
    """In-memory adapter contract; implementations must not perform file I/O."""

    format_key: str

    def validate(self, content: bytes) -> None: ...

    def prepare(self, content: bytes) -> bytes: ...

    def verify(self, content: bytes) -> str: ...


class DocumentFormatRegistry:
    def __init__(self, handlers: Iterable[IDocumentFormatHandler]) -> None:
        registered: dict[str, IDocumentFormatHandler] = {}
        for handler in handlers:
            format_key = _nonempty_text(getattr(handler, "format_key", None), "handler.format_key")
            if format_key in registered:
                raise DuplicateFormatHandlerError(f"Duplicate document format handler: {format_key}")
            for method_name in ("validate", "prepare", "verify"):
                if not callable(getattr(handler, method_name, None)):
                    raise DocumentFormatError(f"Handler for {format_key} lacks {method_name}()")
            registered[format_key] = handler
        self._handlers: Mapping[str, IDocumentFormatHandler] = MappingProxyType(registered)

    @property
    def format_keys(self) -> tuple[str, ...]:
        return tuple(self._handlers)

    def resolve(self, format_key: str) -> IDocumentFormatHandler:
        format_key = _nonempty_text(format_key, "format_key")
        try:
            return self._handlers[format_key]
        except KeyError as error:
            raise DocumentFormatError(f"Unsupported document format: {format_key}") from error

    def prepare(self, document: DocumentPayload) -> PreparedDocument:
        if not isinstance(document, DocumentPayload):
            raise DocumentFormatError("document must be a DocumentPayload")
        handler = self.resolve(document.format_key)
        handler.validate(document.content)
        prepared_content = handler.prepare(document.content)
        if not isinstance(prepared_content, bytes):
            raise DocumentFormatError("Format handler must return prepared bytes")
        verification_ref = _nonempty_text(handler.verify(prepared_content), "verification_ref")
        digest = hashlib.sha256(prepared_content).hexdigest()
        return PreparedDocument(
            format_key=document.format_key,
            content=prepared_content,
            digest=digest,
            verification_ref=verification_ref,
        )