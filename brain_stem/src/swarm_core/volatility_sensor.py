from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from threading import RLock
from time import time_ns
from typing import Callable


class VolatilitySensorError(ValueError):
    pass


class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


@dataclass(frozen=True, slots=True)
class VolatilitySample:
    observed_at_ns: int
    queue_depth: int
    queue_capacity: int
    resource_demand_bytes: int
    resource_capacity_bytes: int
    prompt_urgency: float
    probe_succeeded: bool | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "observed_at_ns",
            "queue_depth",
            "resource_demand_bytes",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise VolatilitySensorError(f"{field_name} must be a non-negative integer")
        for field_name in ("queue_capacity", "resource_capacity_bytes"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise VolatilitySensorError(f"{field_name} must be a positive integer")
        if (
            isinstance(self.prompt_urgency, bool)
            or not isinstance(self.prompt_urgency, (int, float))
            or not math.isfinite(self.prompt_urgency)
            or not 0.0 <= self.prompt_urgency <= 1.0
        ):
            raise VolatilitySensorError("prompt_urgency must be finite and between 0 and 1")
        if self.probe_succeeded is not None and not isinstance(self.probe_succeeded, bool):
            raise VolatilitySensorError("probe_succeeded must be a boolean or None")


@dataclass(frozen=True, slots=True)
class VolatilityPolicy:
    policy_version: str
    maximum_sample_age_ns: int
    queue_trip_ratio: float
    queue_recovery_ratio: float
    resource_trip_ratio: float
    resource_recovery_ratio: float
    cooldown_ns: int
    urgency_gain: float
    maximum_attention_multiplier: float
    high_urgency_threshold: float
    high_urgency_multiplier: float

    def __post_init__(self) -> None:
        if not isinstance(self.policy_version, str) or not self.policy_version.strip():
            raise VolatilitySensorError("policy_version must be a non-empty string")
        for field_name in ("maximum_sample_age_ns", "cooldown_ns"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise VolatilitySensorError(f"{field_name} must be a positive integer")
        for field_name in (
            "queue_trip_ratio",
            "queue_recovery_ratio",
            "resource_trip_ratio",
            "resource_recovery_ratio",
            "high_urgency_threshold",
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise VolatilitySensorError(f"{field_name} must be finite and between 0 and 1")
        if self.queue_recovery_ratio >= self.queue_trip_ratio:
            raise VolatilitySensorError("queue recovery threshold must be below its trip threshold")
        if self.resource_recovery_ratio >= self.resource_trip_ratio:
            raise VolatilitySensorError("resource recovery threshold must be below its trip threshold")
        for field_name in ("urgency_gain", "maximum_attention_multiplier", "high_urgency_multiplier"):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise VolatilitySensorError(f"{field_name} must be finite and non-negative")
        if self.maximum_attention_multiplier < 1.0 or self.high_urgency_multiplier < 1.0:
            raise VolatilitySensorError("attention multipliers must be at least 1")
        if self.high_urgency_multiplier > self.maximum_attention_multiplier:
            raise VolatilitySensorError("high urgency multiplier exceeds configured maximum")


@dataclass(frozen=True, slots=True)
class VolatilityDecision:
    policy_version: str
    circuit_state: CircuitState
    queue_ratio: float
    resource_ratio: float
    prompt_urgency: float
    attention_multiplier: float
    allow_new_work: bool
    allow_probe_work: bool
    reason: str


class VolatilitySensor:
    """Deterministic Block 13 brake; it observes pressure but cannot raise budgets."""

    def __init__(self, policy: VolatilityPolicy, *, clock: Callable[[], int] = time_ns) -> None:
        if not isinstance(policy, VolatilityPolicy):
            raise TypeError("policy must be a VolatilityPolicy")
        if not callable(clock):
            raise ValueError("clock must be callable")
        self._policy = policy
        self._clock = clock
        self._state = CircuitState.CLOSED
        self._opened_at_ns: int | None = None
        self._lock = RLock()

    @property
    def circuit_state(self) -> CircuitState:
        with self._lock:
            return self._state

    def evaluate(self, sample: VolatilitySample) -> VolatilityDecision:
        if not isinstance(sample, VolatilitySample):
            raise TypeError("sample must be a VolatilitySample")
        now_ns = self._clock()
        if isinstance(now_ns, bool) or not isinstance(now_ns, int) or now_ns < 0:
            raise VolatilitySensorError("clock must return a non-negative integer nanosecond timestamp")
        age_ns = now_ns - sample.observed_at_ns
        queue_ratio = sample.queue_depth / sample.queue_capacity
        resource_ratio = sample.resource_demand_bytes / sample.resource_capacity_bytes
        stale = age_ns < 0 or age_ns > self._policy.maximum_sample_age_ns
        trip = queue_ratio >= self._policy.queue_trip_ratio or resource_ratio >= self._policy.resource_trip_ratio
        recovered = (
            queue_ratio <= self._policy.queue_recovery_ratio
            and resource_ratio <= self._policy.resource_recovery_ratio
        )

        with self._lock:
            reason = "within-policy"
            if stale:
                self._open(now_ns)
                reason = "stale-or-future-sample"
            elif self._state is CircuitState.CLOSED and trip:
                self._open(now_ns)
                reason = "pressure-trip"
            elif self._state is CircuitState.OPEN:
                if (
                    self._opened_at_ns is not None
                    and now_ns - self._opened_at_ns >= self._policy.cooldown_ns
                    and recovered
                ):
                    self._state = CircuitState.HALF_OPEN
                    reason = "cooldown-recovered-probe-required"
                else:
                    reason = "circuit-open"
            elif self._state is CircuitState.HALF_OPEN:
                if trip or sample.probe_succeeded is False:
                    self._open(now_ns)
                    reason = "half-open-probe-failed"
                elif recovered and sample.probe_succeeded is True:
                    self._state = CircuitState.CLOSED
                    self._opened_at_ns = None
                    reason = "half-open-probe-succeeded"
                else:
                    reason = "half-open-probe-pending"

            linear_multiplier = 1.0 + self._policy.urgency_gain * sample.prompt_urgency
            attention_multiplier = min(self._policy.maximum_attention_multiplier, linear_multiplier)
            if sample.prompt_urgency >= self._policy.high_urgency_threshold:
                attention_multiplier = max(attention_multiplier, self._policy.high_urgency_multiplier)
            return VolatilityDecision(
                policy_version=self._policy.policy_version,
                circuit_state=self._state,
                queue_ratio=queue_ratio,
                resource_ratio=resource_ratio,
                prompt_urgency=sample.prompt_urgency,
                attention_multiplier=min(attention_multiplier, self._policy.maximum_attention_multiplier),
                allow_new_work=self._state is CircuitState.CLOSED,
                allow_probe_work=self._state is CircuitState.HALF_OPEN,
                reason=reason,
            )

    def _open(self, now_ns: int) -> None:
        self._state = CircuitState.OPEN
        self._opened_at_ns = now_ns