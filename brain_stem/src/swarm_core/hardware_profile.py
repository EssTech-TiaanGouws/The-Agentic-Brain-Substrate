from __future__ import annotations

import json
import os
import platform
import sys
import time
from dataclasses import dataclass

from .training_readiness import TrainingEnvironmentEvidence, TrainingMethod

try:
    import torch
except ImportError:  # The probe can still report host-only information.
    torch = None  # type: ignore[assignment]


class HardwareProbeError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AcceleratorSnapshot:
    device_id: str
    backend_id: str
    name: str
    total_memory_bytes: int | None
    available_memory_bytes: int | None
    supports_fp16: bool
    supports_bf16: bool

    def __post_init__(self) -> None:
        for field_name in ("device_id", "backend_id", "name"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise HardwareProbeError(f"{field_name} must be a non-empty string")
        for field_name in ("total_memory_bytes", "available_memory_bytes"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise HardwareProbeError(f"{field_name} must be a non-negative integer or unknown")
        if (
            self.total_memory_bytes is not None
            and self.available_memory_bytes is not None
            and self.available_memory_bytes > self.total_memory_bytes
        ):
            raise HardwareProbeError("available accelerator memory cannot exceed total memory")
        if not isinstance(self.supports_fp16, bool) or not isinstance(self.supports_bf16, bool):
            raise HardwareProbeError("precision support fields must be booleans")


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    observed_at_ns: int
    operating_system: str
    machine_architecture: str
    logical_cpu_count: int
    host_memory_total_bytes: int | None
    host_memory_available_bytes: int | None
    tensor_runtime: str | None
    cuda_runtime: str | None
    accelerators: tuple[AcceleratorSnapshot, ...]

    def __post_init__(self) -> None:
        if isinstance(self.observed_at_ns, bool) or not isinstance(self.observed_at_ns, int) or self.observed_at_ns < 0:
            raise HardwareProbeError("observed_at_ns must be a non-negative integer")
        for field_name in ("operating_system", "machine_architecture"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise HardwareProbeError(f"{field_name} must be a non-empty string")
        if isinstance(self.logical_cpu_count, bool) or not isinstance(self.logical_cpu_count, int) or self.logical_cpu_count < 1:
            raise HardwareProbeError("logical_cpu_count must be a positive integer")
        for field_name in ("host_memory_total_bytes", "host_memory_available_bytes"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise HardwareProbeError(f"{field_name} must be a non-negative integer or unknown")
        if (
            self.host_memory_total_bytes is not None
            and self.host_memory_available_bytes is not None
            and self.host_memory_available_bytes > self.host_memory_total_bytes
        ):
            raise HardwareProbeError("available host memory cannot exceed total memory")
        for field_name in ("tensor_runtime", "cuda_runtime"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise HardwareProbeError(f"{field_name} must be a non-empty string or unknown")
        if not isinstance(self.accelerators, tuple) or any(
            not isinstance(device, AcceleratorSnapshot) for device in self.accelerators
        ):
            raise HardwareProbeError("accelerators must be a tuple of AcceleratorSnapshot values")
        device_ids = tuple(device.device_id for device in self.accelerators)
        if len(set(device_ids)) != len(device_ids):
            raise HardwareProbeError("accelerator device IDs must be unique")

    def to_payload(self) -> dict[str, object]:
        return {
            "observed_at_ns": self.observed_at_ns,
            "operating_system": self.operating_system,
            "machine_architecture": self.machine_architecture,
            "logical_cpu_count": self.logical_cpu_count,
            "host_memory_total_bytes": self.host_memory_total_bytes,
            "host_memory_available_bytes": self.host_memory_available_bytes,
            "tensor_runtime": self.tensor_runtime,
            "cuda_runtime": self.cuda_runtime,
            "accelerators": [
                {
                    "device_id": device.device_id,
                    "backend_id": device.backend_id,
                    "name": device.name,
                    "total_memory_bytes": device.total_memory_bytes,
                    "available_memory_bytes": device.available_memory_bytes,
                    "supports_fp16": device.supports_fp16,
                    "supports_bf16": device.supports_bf16,
                }
                for device in self.accelerators
            ],
        }

    def to_training_environment_evidence(
        self,
        *,
        device_id: str,
        execution_profile_id: str,
        accessible_dataset_ids: tuple[str, ...],
        approved_run_references: tuple[str, ...],
        artifact_store_reference: str,
        available_teacher_candidate_ids: tuple[str, ...] = (),
    ) -> TrainingEnvironmentEvidence:
        if not isinstance(device_id, str) or not device_id.strip():
            raise HardwareProbeError("device_id must be a non-empty string")
        if device_id == "cpu":
            backend_id = "torch.cpu"
            available_memory_bytes = 0
        else:
            matching = next(
                (device for device in self.accelerators if device.device_id == device_id),
                None,
            )
            if matching is None:
                raise HardwareProbeError(f"device {device_id} is not present in this hardware snapshot")
            if matching.available_memory_bytes is None:
                raise HardwareProbeError(f"free memory for {device_id} was not measured")
            backend_id = f"torch.{matching.backend_id}"
            available_memory_bytes = matching.available_memory_bytes
        runtime_version = self.tensor_runtime
        if runtime_version is None:
            raise HardwareProbeError("PyTorch runtime is unavailable; no training backend can be declared")
        return TrainingEnvironmentEvidence(
            backend_id=backend_id,
            backend_version=runtime_version,
            execution_profile_id=execution_profile_id,
            available_device_memory_bytes=available_memory_bytes,
            supported_methods=(TrainingMethod.FROM_SCRATCH,),
            accessible_dataset_ids=accessible_dataset_ids,
            available_teacher_candidate_ids=available_teacher_candidate_ids,
            artifact_store_reference=artifact_store_reference,
            approved_run_references=approved_run_references,
        )


def _host_memory_bytes() -> tuple[int | None, int | None]:
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        total_pages = int(os.sysconf("SC_PHYS_PAGES"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None, None
    if page_size <= 0 or total_pages < 0 or available_pages < 0:
        return None, None
    total = page_size * total_pages
    available = min(page_size * available_pages, total)
    return total, available


def _cuda_accelerators() -> tuple[AcceleratorSnapshot, ...]:
    if torch is None or not torch.cuda.is_available():
        return ()
    snapshots: list[AcceleratorSnapshot] = []
    for index in range(torch.cuda.device_count()):
        try:
            properties = torch.cuda.get_device_properties(index)
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            bf16_support = bool(torch.cuda.is_bf16_supported())
            snapshots.append(
                AcceleratorSnapshot(
                    device_id=f"cuda:{index}",
                    backend_id="cuda",
                    name=str(properties.name),
                    total_memory_bytes=int(total_bytes),
                    available_memory_bytes=int(free_bytes),
                    supports_fp16=True,
                    supports_bf16=bf16_support,
                )
            )
        except Exception as error:
            raise HardwareProbeError(f"failed to measure CUDA device cuda:{index}") from error
    return tuple(snapshots)


def probe_hardware_profile() -> HardwareProfile:
    """Return a one-shot local measurement; it does not reserve memory or set policy."""
    total_memory, available_memory = _host_memory_bytes()
    accelerators = _cuda_accelerators()
    return HardwareProfile(
        observed_at_ns=time.time_ns(),
        operating_system=platform.platform(),
        machine_architecture=platform.machine() or "unknown",
        logical_cpu_count=max(1, os.cpu_count() or 1),
        host_memory_total_bytes=total_memory,
        host_memory_available_bytes=available_memory,
        tensor_runtime=None if torch is None else str(torch.__version__),
        cuda_runtime=None if torch is None else torch.version.cuda,
        accelerators=accelerators,
    )


def main() -> int:
    try:
        profile = probe_hardware_profile()
    except HardwareProbeError as error:
        print(json.dumps({"probe_error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(profile.to_payload(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())