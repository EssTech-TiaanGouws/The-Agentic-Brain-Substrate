from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.swarm_core.lease_manager import ResourceSnapshotError
from src.swarm_core.resource_adapters import LiveResourceSnapshotProvider


class FakeCuda:
    def __init__(self, *, available: bool = True, device_count: int = 2, memory=(1234, 5678)) -> None:
        self._available = available
        self._device_count = device_count
        self._memory = memory

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return self._device_count

    def mem_get_info(self, device_index: int):
        return self._memory


class LiveResourceSnapshotTests(unittest.TestCase):
    def test_host_provider_returns_a_live_nonnegative_snapshot(self) -> None:
        provider = LiveResourceSnapshotProvider(
            "host-memory",
            device_id="host",
            clock=lambda: 123,
        )
        snapshot = provider()

        self.assertEqual(snapshot.resource_domain, "host-memory")
        self.assertGreater(snapshot.available_bytes, 0)
        self.assertEqual(snapshot.observed_at_ns, 123)

    def test_cuda_provider_reads_selected_device_memory(self) -> None:
        fake_torch = SimpleNamespace(cuda=FakeCuda(memory=(4096, 8192)))
        provider = LiveResourceSnapshotProvider(
            "device-memory",
            device_id="cuda:1",
            clock=lambda: 456,
        )

        with patch.dict(sys.modules, {"torch": fake_torch}):
            snapshot = provider()

        self.assertEqual(snapshot.available_bytes, 4096)
        self.assertEqual(snapshot.observed_at_ns, 456)

    def test_cuda_provider_rejects_unavailable_or_missing_device(self) -> None:
        unavailable = SimpleNamespace(cuda=FakeCuda(available=False))
        provider = LiveResourceSnapshotProvider("cuda", device_id="cuda:0")
        with patch.dict(sys.modules, {"torch": unavailable}):
            with self.assertRaises(ResourceSnapshotError):
                provider()

        missing = SimpleNamespace(cuda=FakeCuda(device_count=1))
        missing_provider = LiveResourceSnapshotProvider("cuda", device_id="cuda:1")
        with patch.dict(sys.modules, {"torch": missing}):
            with self.assertRaises(ResourceSnapshotError):
                missing_provider()

    def test_provider_rejects_invalid_device_names(self) -> None:
        with self.assertRaises(ValueError):
            LiveResourceSnapshotProvider("memory", device_id="cuda:gpu0")


if __name__ == "__main__":
    unittest.main()
