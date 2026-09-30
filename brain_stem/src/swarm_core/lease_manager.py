from __future__ import annotations

import sys
from dataclasses import dataclass
from threading import Lock
from time import time_ns
from typing import Callable


class ResourceConfigurationError(ValueError):
    pass


class ResourceAdmissionDenied(RuntimeError):
    pass


class ResourceSnapshotError(RuntimeError):
    pass


def _validate_nonnegative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ResourceConfigurationError(f"{field_name} must be an integer")
    if value < 0 or value > sys.maxsize:
        raise ResourceConfigurationError(f"{field_name} is outside the supported range")
    return value


def _validate_name(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResourceConfigurationError(f"{field_name} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class ResourcePolicy:
    profile_name: str
    resource_domain: str
    safety_margin_bytes: int
    max_snapshot_age_ns: int

    def __post_init__(self) -> None:
        _validate_name(self.profile_name, "profile_name")
        _validate_name(self.resource_domain, "resource_domain")
        _validate_nonnegative_int(self.safety_margin_bytes, "safety_margin_bytes")
        _validate_nonnegative_int(self.max_snapshot_age_ns, "max_snapshot_age_ns")


@dataclass(frozen=True, slots=True)
class ResourceProfile:
    profile_name: str
    resource_domain: str
    requested_bytes: int

    def __post_init__(self) -> None:
        _validate_name(self.profile_name, "profile_name")
        _validate_name(self.resource_domain, "resource_domain")
        _validate_nonnegative_int(self.requested_bytes, "requested_bytes")


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    resource_domain: str
    available_bytes: int
    observed_at_ns: int

    def __post_init__(self) -> None:
        _validate_name(self.resource_domain, "resource_domain")
        _validate_nonnegative_int(self.available_bytes, "available_bytes")
        _validate_nonnegative_int(self.observed_at_ns, "observed_at_ns")


class ResourceLease:
    def __init__(self, manager: ResourceLeaseManager, token: object, reserved_bytes: int) -> None:
        self._manager = manager
        self._token = token
        self.reserved_bytes = reserved_bytes
        self._released = False

    @property
    def released(self) -> bool:
        return self._released

    def release(self) -> None:
        if not self._released:
            self._manager._release(self._token)
            self._released = True

    def __enter__(self) -> ResourceLease:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()


class ResourceLeaseManager:
    """Atomically leases configured byte budgets against a live resource snapshot."""

    def __init__(
        self,
        policy: ResourcePolicy,
        snapshot_provider: Callable[[], ResourceSnapshot],
        clock: Callable[[], int] = time_ns,
    ) -> None:
        if not isinstance(policy, ResourcePolicy):
            raise ResourceConfigurationError("policy must be a ResourcePolicy")
        if not callable(snapshot_provider) or not callable(clock):
            raise ResourceConfigurationError("snapshot_provider and clock must be callable")
        self._policy = policy
        self._snapshot_provider = snapshot_provider
        self._clock = clock
        self._lock = Lock()
        self._leases: dict[object, int] = {}

    @property
    def reserved_bytes(self) -> int:
        with self._lock:
            return sum(self._leases.values())

    def acquire(self, profile: ResourceProfile) -> ResourceLease:
        if not isinstance(profile, ResourceProfile):
            raise ResourceConfigurationError("profile must be a ResourceProfile")
        if profile.profile_name != self._policy.profile_name:
            raise ResourceAdmissionDenied("Resource profile does not match configured policy")
        if profile.resource_domain != self._policy.resource_domain:
            raise ResourceAdmissionDenied("Resource domain does not match configured policy")

        with self._lock:
            snapshot = self._snapshot_provider()
            self._validate_snapshot(snapshot)
            remaining_bytes = (
                snapshot.available_bytes
                - self._policy.safety_margin_bytes
                - sum(self._leases.values())
            )
            if remaining_bytes < 0 or profile.requested_bytes > remaining_bytes:
                raise ResourceAdmissionDenied("Requested resource budget exceeds admitted capacity")

            token = object()
            self._leases[token] = profile.requested_bytes
            return ResourceLease(self, token, profile.requested_bytes)

    def _validate_snapshot(self, snapshot: ResourceSnapshot) -> None:
        if not isinstance(snapshot, ResourceSnapshot):
            raise ResourceSnapshotError("Snapshot provider returned an invalid resource snapshot")
        if snapshot.resource_domain != self._policy.resource_domain:
            raise ResourceSnapshotError("Snapshot resource domain does not match configured policy")

        now_ns = self._clock()
        try:
            _validate_nonnegative_int(now_ns, "clock value")
        except ResourceConfigurationError as error:
            raise ResourceSnapshotError(str(error)) from error
        age_ns = now_ns - snapshot.observed_at_ns
        if age_ns < 0:
            raise ResourceSnapshotError("Resource snapshot timestamp is in the future")
        if age_ns > self._policy.max_snapshot_age_ns:
            raise ResourceSnapshotError("Resource snapshot is stale")

    def _release(self, token: object) -> None:
        with self._lock:
            self._leases.pop(token, None)