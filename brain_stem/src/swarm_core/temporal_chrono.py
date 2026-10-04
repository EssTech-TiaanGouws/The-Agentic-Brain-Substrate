from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from substrate.contracts import ScopeVector

from .world_state import WorldStateRecord


class TemporalChronoError(ValueError):
    pass


class DriftDirection(str, Enum):
    INCREASING = "INCREASING"
    DECREASING = "DECREASING"
    STABLE = "STABLE"


@dataclass(frozen=True, slots=True)
class TemporalChronoPolicy:
    policy_version: str
    maximum_observations: int
    maximum_age_ns: int
    maximum_gap_ns: int
    stable_rate_threshold_per_second: float
    maximum_forecast_horizon_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.policy_version, str) or not self.policy_version.strip():
            raise TemporalChronoError("policy_version must be a non-empty string")
        for field_name in ("maximum_observations", "maximum_age_ns", "maximum_gap_ns"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise TemporalChronoError(f"{field_name} must be a positive integer")
        for field_name in (
            "stable_rate_threshold_per_second",
            "maximum_forecast_horizon_seconds",
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise TemporalChronoError(f"{field_name} must be finite and non-negative")
        if self.maximum_forecast_horizon_seconds <= 0:
            raise TemporalChronoError("maximum_forecast_horizon_seconds must be positive")


@dataclass(frozen=True, slots=True)
class TemporalObservation:
    assertion_id: str
    occurred_at_ns: int
    value: float


@dataclass(frozen=True, slots=True)
class TemporalForecast:
    horizon_seconds: float
    predicted_value: float


@dataclass(frozen=True, slots=True)
class TemporalDriftReport:
    policy_version: str
    scope: ScopeVector
    fact_key: str
    subject: str
    predicate: str
    direction: DriftDirection
    observations: tuple[TemporalObservation, ...]
    slope_per_second: float
    intercept: float
    r_squared: float
    forecasts: tuple[TemporalForecast, ...]
    provenance_assertion_ids: tuple[str, ...]


class TemporalChronoAnalyzer:
    """Read-only trend analysis; output is advisory and never updates the ledger."""

    def __init__(self, policy: TemporalChronoPolicy) -> None:
        if not isinstance(policy, TemporalChronoPolicy):
            raise TypeError("policy must be a TemporalChronoPolicy")
        self._policy = policy

    def analyze(
        self,
        records: Iterable[WorldStateRecord],
        *,
        scope: ScopeVector,
        fact_key: str,
        subject: str,
        predicate: str,
        now_ns: int,
        forecast_horizons_seconds: tuple[float, ...] = (),
    ) -> TemporalDriftReport:
        if not isinstance(scope, ScopeVector) or not scope.is_complete():
            raise TemporalChronoError("a complete four-field scope is required")
        for field_name, value in (("fact_key", fact_key), ("subject", subject), ("predicate", predicate)):
            if not isinstance(value, str) or not value.strip():
                raise TemporalChronoError(f"{field_name} must be a non-empty string")
        if isinstance(now_ns, bool) or not isinstance(now_ns, int) or now_ns < 0:
            raise TemporalChronoError("now_ns must be a non-negative integer")
        if not isinstance(forecast_horizons_seconds, tuple) or any(
            isinstance(horizon, bool)
            or not isinstance(horizon, (int, float))
            or not math.isfinite(horizon)
            or horizon <= 0
            or horizon > self._policy.maximum_forecast_horizon_seconds
            for horizon in forecast_horizons_seconds
        ):
            raise TemporalChronoError("forecast horizons must be positive and within the configured maximum")

        matching: list[WorldStateRecord] = []
        for record in records:
            if not isinstance(record, WorldStateRecord):
                raise TemporalChronoError("records must contain WorldStateRecord values")
            assertion = record.assertion
            if assertion.scope != scope:
                raise TemporalChronoError("temporal series crosses the trusted scope boundary")
            if assertion.fact_key == fact_key and assertion.subject == subject and assertion.predicate == predicate:
                if record.occurred_at_ns <= now_ns and now_ns - record.occurred_at_ns <= self._policy.maximum_age_ns:
                    matching.append(record)
        matching.sort(key=lambda record: (record.occurred_at_ns, record.sequence, record.assertion.assertion_id))
        matching = matching[-self._policy.maximum_observations :]
        if len(matching) < 2:
            raise TemporalChronoError("at least two recent matching assertions are required")

        observations = tuple(
            TemporalObservation(
                record.assertion.assertion_id,
                record.occurred_at_ns,
                self._numeric_value(record.assertion.value),
            )
            for record in matching
        )
        times_seconds = tuple((item.occurred_at_ns - observations[0].occurred_at_ns) / 1_000_000_000 for item in observations)
        if any(
            current.occurred_at_ns - previous.occurred_at_ns > self._policy.maximum_gap_ns
            for previous, current in zip(observations, observations[1:])
        ):
            raise TemporalChronoError("temporal series contains a gap above the configured maximum")
        if len(set(times_seconds)) != len(times_seconds):
            raise TemporalChronoError("temporal series requires distinct observation timestamps")

        values = tuple(item.value for item in observations)
        mean_time = math.fsum(times_seconds) / len(times_seconds)
        mean_value = math.fsum(values) / len(values)
        denominator = math.fsum((time_value - mean_time) ** 2 for time_value in times_seconds)
        if denominator <= 0:
            raise TemporalChronoError("temporal observations do not span a time interval")
        slope = math.fsum(
            (time_value - mean_time) * (value - mean_value)
            for time_value, value in zip(times_seconds, values)
        ) / denominator
        intercept = mean_value - slope * mean_time
        residual_sum = math.fsum(
            (value - (intercept + slope * time_value)) ** 2
            for time_value, value in zip(times_seconds, values)
        )
        total_sum = math.fsum((value - mean_value) ** 2 for value in values)
        r_squared = 1.0 if total_sum == 0 else max(0.0, min(1.0, 1.0 - residual_sum / total_sum))
        if abs(slope) <= self._policy.stable_rate_threshold_per_second:
            direction = DriftDirection.STABLE
        elif slope > 0:
            direction = DriftDirection.INCREASING
        else:
            direction = DriftDirection.DECREASING
        forecasts = tuple(
            TemporalForecast(
                float(horizon),
                intercept + slope * (times_seconds[-1] + float(horizon)),
            )
            for horizon in forecast_horizons_seconds
        )
        return TemporalDriftReport(
            self._policy.policy_version,
            scope,
            fact_key,
            subject,
            predicate,
            direction,
            observations,
            slope,
            intercept,
            r_squared,
            forecasts,
            tuple(item.assertion_id for item in observations),
        )

    @staticmethod
    def _numeric_value(value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise TemporalChronoError("temporal drift currently accepts finite numeric assertion values only")
        return float(value)