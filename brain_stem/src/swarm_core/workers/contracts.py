from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum


class WorkerLifecycle(str, Enum):
    TRANSIENT = "TRANSIENT"


@dataclass(frozen=True, slots=True)
class MockWorkerRequest:
    content: bytes
    format_key: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.content, bytes):
            raise ValueError("content must be bytes")
        if self.format_key is not None and (
            not isinstance(self.format_key, str) or not self.format_key.strip()
        ):
            raise ValueError("format_key must be a non-empty string when supplied")

    @property
    def content_digest(self) -> str:
        return hashlib.sha256(self.content).hexdigest()


@dataclass(frozen=True, slots=True)
class MockWorkerResult:
    block_id: int
    capability_key: str
    lifecycle: WorkerLifecycle
    status: str
    input_digest: str
    output_digest: str | None = None
    output_content: bytes | None = None
    verification_ref: str | None = None