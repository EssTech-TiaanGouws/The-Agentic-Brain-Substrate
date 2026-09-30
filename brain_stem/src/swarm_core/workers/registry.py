from __future__ import annotations

from types import MappingProxyType
from typing import Iterable, Mapping, Protocol

from ..format_adapters import DocumentFormatRegistry
from .block_01_sensory import SensoryOpticalMockWorker
from .block_09_algorithmic_coder import AlgorithmicCoderMockWorker
from .block_11_document_driver import DeclarativeDocumentDriverMockWorker
from .contracts import WorkerLifecycle


class CapabilityRegistrationError(ValueError):
    pass


class TransientWorker(Protocol):
    block_id: int
    capability_key: str
    lifecycle: WorkerLifecycle


class CapabilityRegistry:
    def __init__(self, workers: Iterable[TransientWorker]) -> None:
        by_capability: dict[str, TransientWorker] = {}
        by_block: dict[int, TransientWorker] = {}
        for worker in workers:
            block_id = getattr(worker, "block_id", None)
            capability_key = getattr(worker, "capability_key", None)
            lifecycle = getattr(worker, "lifecycle", None)
            if isinstance(block_id, bool) or not isinstance(block_id, int) or not 1 <= block_id <= 15:
                raise CapabilityRegistrationError("Transient worker block_id must be in 1..15")
            if not isinstance(capability_key, str) or not capability_key.strip():
                raise CapabilityRegistrationError("Transient worker needs a capability key")
            if lifecycle is not WorkerLifecycle.TRANSIENT:
                raise CapabilityRegistrationError("Capability registry accepts transient workers only")
            if block_id in by_block or capability_key in by_capability:
                raise CapabilityRegistrationError("Duplicate block or capability registration")
            if not callable(getattr(worker, "execute", None)):
                raise CapabilityRegistrationError("Transient worker must implement execute()")
            by_block[block_id] = worker
            by_capability[capability_key] = worker
        self._by_capability: Mapping[str, TransientWorker] = MappingProxyType(by_capability)
        self._by_block: Mapping[int, TransientWorker] = MappingProxyType(by_block)

    @property
    def block_ids(self) -> tuple[int, ...]:
        return tuple(self._by_block)

    def resolve(self, capability_key: str) -> TransientWorker:
        if not isinstance(capability_key, str) or not capability_key.strip():
            raise CapabilityRegistrationError("capability_key must be a non-empty string")
        try:
            return self._by_capability[capability_key]
        except KeyError as error:
            raise CapabilityRegistrationError(f"Unknown capability: {capability_key}") from error


def build_mock_registry(formats: DocumentFormatRegistry) -> CapabilityRegistry:
    return CapabilityRegistry(
        (
            SensoryOpticalMockWorker(),
            AlgorithmicCoderMockWorker(),
            DeclarativeDocumentDriverMockWorker(formats),
        )
    )