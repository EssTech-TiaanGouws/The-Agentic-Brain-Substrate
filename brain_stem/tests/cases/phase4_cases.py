from __future__ import annotations

import hashlib
import unittest

from src.swarm_core.format_adapters import (
    DocumentFormatError,
    DocumentFormatRegistry,
    DocumentPayload,
)
from src.swarm_core.workers import MockWorkerRequest, WorkerLifecycle, build_mock_registry
from src.swarm_core.workers.block_01_sensory import SensoryOpticalMockWorker
from src.swarm_core.workers.block_09_algorithmic_coder import AlgorithmicCoderMockWorker
from src.swarm_core.workers.block_11_document_driver import DeclarativeDocumentDriverMockWorker
from src.swarm_core.workers.registry import CapabilityRegistrationError, CapabilityRegistry


class MemoryTextHandler:
    format_key = "fixture.text"

    def __init__(self) -> None:
        self.events: list[str] = []

    def validate(self, content: bytes) -> None:
        self.events.append("validate")
        content.decode("utf-8")

    def prepare(self, content: bytes) -> bytes:
        self.events.append("prepare")
        return content.strip()

    def verify(self, content: bytes) -> str:
        self.events.append("verify")
        content.decode("utf-8")
        return "fixture-verification"


class InvalidOutputHandler(MemoryTextHandler):
    def prepare(self, content: bytes) -> bytes:
        return "not bytes"


class Phase4AdapterTests(unittest.TestCase):
    def test_format_handler_validates_prepares_and_verifies_in_order(self) -> None:
        handler = MemoryTextHandler()
        registry = DocumentFormatRegistry((handler,))

        prepared = registry.prepare(DocumentPayload("fixture.text", b"  content  "))

        self.assertEqual(handler.events, ["validate", "prepare", "verify"])
        self.assertEqual(prepared.content, b"content")
        self.assertEqual(prepared.digest, hashlib.sha256(b"content").hexdigest())
        self.assertEqual(prepared.verification_ref, "fixture-verification")

    def test_registry_rejects_duplicate_format_keys_and_unknown_formats(self) -> None:
        with self.assertRaises(DocumentFormatError):
            DocumentFormatRegistry((MemoryTextHandler(), MemoryTextHandler()))

        registry = DocumentFormatRegistry((MemoryTextHandler(),))
        with self.assertRaises(DocumentFormatError):
            registry.prepare(DocumentPayload("unknown.format", b"content"))

    def test_registry_rejects_handler_that_returns_non_bytes(self) -> None:
        registry = DocumentFormatRegistry((InvalidOutputHandler(),))

        with self.assertRaises(DocumentFormatError):
            registry.prepare(DocumentPayload("fixture.text", b"content"))

    def test_block_1_and_block_9_are_transient_brand_neutral_mocks(self) -> None:
        request = MockWorkerRequest(content=b"opaque input")
        sensory = SensoryOpticalMockWorker().execute(request)
        coder = AlgorithmicCoderMockWorker().execute(request)

        self.assertEqual((sensory.block_id, coder.block_id), (1, 9))
        self.assertEqual(sensory.lifecycle, WorkerLifecycle.TRANSIENT)
        self.assertEqual(coder.lifecycle, WorkerLifecycle.TRANSIENT)
        self.assertEqual(sensory.status, "MOCKED")
        self.assertEqual(coder.status, "MOCKED")
        self.assertIsNone(coder.output_content)

    def test_block_11_stages_prepared_bytes_in_memory_without_writing_files(self) -> None:
        handler = MemoryTextHandler()
        registry = DocumentFormatRegistry((handler,))
        worker = DeclarativeDocumentDriverMockWorker(registry)

        result = worker.execute(MockWorkerRequest(b"  brief  ", "fixture.text"))

        self.assertEqual(result.block_id, 11)
        self.assertEqual(result.lifecycle, WorkerLifecycle.TRANSIENT)
        self.assertEqual(result.status, "STAGED_IN_MEMORY")
        self.assertEqual(result.output_content, b"brief")
        self.assertEqual(result.output_digest, hashlib.sha256(b"brief").hexdigest())

    def test_block_11_requires_an_explicit_supported_format(self) -> None:
        worker = DeclarativeDocumentDriverMockWorker(DocumentFormatRegistry(()))

        with self.assertRaises(ValueError):
            worker.execute(MockWorkerRequest(b"brief"))
        with self.assertRaises(DocumentFormatError):
            worker.execute(MockWorkerRequest(b"brief", "missing.format"))

    def test_mock_registry_registers_only_the_three_requested_transient_blocks(self) -> None:
        registry = build_mock_registry(DocumentFormatRegistry((MemoryTextHandler(),)))

        self.assertEqual(set(registry.block_ids), {1, 9, 11})
        self.assertEqual(registry.resolve("sensory.inspect").block_id, 1)
        self.assertEqual(registry.resolve("coder.inspect_artifact").block_id, 9)
        self.assertEqual(registry.resolve("document.stage").block_id, 11)

    def test_capability_registry_rejects_unknown_and_duplicate_workers(self) -> None:
        registry = CapabilityRegistry((SensoryOpticalMockWorker(),))
        with self.assertRaises(CapabilityRegistrationError):
            registry.resolve("not.registered")
        with self.assertRaises(CapabilityRegistrationError):
            CapabilityRegistry((SensoryOpticalMockWorker(), SensoryOpticalMockWorker()))


if __name__ == "__main__":
    unittest.main()