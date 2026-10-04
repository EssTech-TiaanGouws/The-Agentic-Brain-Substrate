from __future__ import annotations

import os
import re
from time import time_ns
from typing import Callable

from .lease_manager import ResourceSnapshot, ResourceSnapshotError


class LiveResourceSnapshotProvider:
    """Reads host or CUDA availability; the lease manager still reserves in-process only."""

    def __init__(
        self,
        resource_domain: str,
        *,
        device_id: str,
        clock: Callable[[], int] = time_ns,
    ) -> None:
        if not isinstance(resource_domain, str) or not resource_domain.strip():
            raise ValueError("resource_domain must be a non-empty string")
        if not isinstance(device_id, str) or not device_id.strip():
            raise ValueError("device_id must be a non-empty string")
        if device_id != "host" and re.fullmatch(r"cuda:[0-9]+", device_id) is None:
            raise ValueError("device_id must be 'host' or a CUDA device such as 'cuda:0'")
        if not callable(clock):
            raise ValueError("clock must be callable")
        self.resource_domain = resource_domain
        self.device_id = device_id
        self._clock = clock

    def __call__(self) -> ResourceSnapshot:
        available_bytes = self._host_available_bytes() if self.device_id == "host" else self._cuda_available_bytes()
        observed_at_ns = self._clock()
        return ResourceSnapshot(self.resource_domain, available_bytes, observed_at_ns)

    @staticmethod
    def _host_available_bytes() -> int:
        try:
            with open("/proc/meminfo", "r", encoding="ascii") as memory_info:
                for line in memory_info:
                    if line.startswith("MemAvailable:"):
                        fields = line.split()
                        if len(fields) != 3 or fields[2] != "kB":
                            raise ResourceSnapshotError("/proc/meminfo has malformed MemAvailable")
                        value = int(fields[1])
                        if value < 0:
                            raise ResourceSnapshotError("host available memory is negative")
                        return value * 1024
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as error:
            raise ResourceSnapshotError("Could not read host available memory") from error
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            available_pages = os.sysconf("SC_AVPHYS_PAGES")
        except (AttributeError, OSError, ValueError) as error:
            raise ResourceSnapshotError("Host memory measurement is unavailable") from error
        if page_size <= 0 or available_pages < 0:
            raise ResourceSnapshotError("Host memory measurement returned invalid values")
        return page_size * available_pages

    def _cuda_available_bytes(self) -> int:
        try:
            import torch
        except ImportError as error:
            raise ResourceSnapshotError("PyTorch is required to measure CUDA memory") from error
        if not torch.cuda.is_available():
            raise ResourceSnapshotError("CUDA is not visible to the runtime")
        device_index = int(self.device_id.split(":", 1)[1])
        if device_index >= torch.cuda.device_count():
            raise ResourceSnapshotError("Requested CUDA device is not visible to the runtime")
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        except (RuntimeError, ValueError) as error:
            raise ResourceSnapshotError("CUDA memory measurement failed") from error
        if isinstance(free_bytes, bool) or not isinstance(free_bytes, int) or free_bytes < 0:
            raise ResourceSnapshotError("CUDA memory measurement returned invalid free bytes")
        if isinstance(total_bytes, bool) or not isinstance(total_bytes, int) or total_bytes < free_bytes:
            raise ResourceSnapshotError("CUDA memory measurement returned invalid total bytes")
        return free_bytes