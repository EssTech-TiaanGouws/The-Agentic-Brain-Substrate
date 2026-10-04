from __future__ import annotations

import hashlib
import unittest

from substrate.contracts import ScopeVector
from src.swarm_core.temporal_chrono import (
    DriftDirection,
    TemporalChronoAnalyzer,
    TemporalChronoError,
    TemporalChronoPolicy,
)
from src.swarm_core.world_state import WorldStateAssertion, WorldStateRecord


SCOPE = ScopeVector("tenant-a", "user-a", "project-a", "workspace-a")
OTHER_SCOPE = ScopeVector("tenant-b", "user-a", "project-a", "workspace-a")


def record(
    assertion_id: str,
    timestamp_ns: int,
    value: object,
    *,
    sequence: int,
    scope: ScopeVector = SCOPE,
    fact_key: str = "environment:temperature",
) -> WorldStateRecord:
    assertion = WorldStateAssertion(
        assertion_id=assertion_id,
        fact_key=fact_key,
        scope=scope,
        subject="sensor:room-a",
        predicate="temperature-celsius",
        value=value,
        source_ref=f"sensor:{assertion_id}",
        source_sha256=hashlib.sha256(assertion_id.encode("utf-8")).hexdigest(),
        transaction_id=f"tx:{assertion_id}",
    )
    return WorldStateRecord(
        sequence,
        assertion,
        timestamp_ns,
        "review:accepted",
        None,
        (),
        None,
        hashlib.sha256(f"event:{assertion_id}".encode("utf-8")).hexdigest(),
    )


class TemporalChronoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = TemporalChronoPolicy(
            policy_version="chrono-v1",
            maximum_observations=8,
            maximum_age_ns=10_000_000_000,
            maximum_gap_ns=5_000_000_000,
            stable_rate_threshold_per_second=0.01,
            maximum_forecast_horizon_seconds=10.0,
        )
        self.analyzer = TemporalChronoAnalyzer(self.policy)

    def test_numeric_drift_forecast_and_provenance_are_deterministic(self) -> None:
        observations = (
            record("obs-3", 2_000_000_000, 24, sequence=3),
            record("obs-1", 0, 20, sequence=1),
            record("obs-2", 1_000_000_000, 22, sequence=2),
        )

        report = self.analyzer.analyze(
            observations,
            scope=SCOPE,
            fact_key="environment:temperature",
            subject="sensor:room-a",
            predicate="temperature-celsius",
            now_ns=3_000_000_000,
            forecast_horizons_seconds=(2.0,),
        )

        self.assertEqual(report.direction, DriftDirection.INCREASING)
        self.assertAlmostEqual(report.slope_per_second, 2.0)
        self.assertAlmostEqual(report.r_squared, 1.0)
        self.assertAlmostEqual(report.forecasts[0].predicted_value, 28.0)
        self.assertEqual(report.provenance_assertion_ids, ("obs-1", "obs-2", "obs-3"))

    def test_stable_and_decreasing_series_are_classified(self) -> None:
        stable = self.analyzer.analyze(
            (record("a", 1, 5, sequence=1), record("b", 1_000_000_001, 5, sequence=2)),
            scope=SCOPE,
            fact_key="environment:temperature",
            subject="sensor:room-a",
            predicate="temperature-celsius",
            now_ns=1_000_000_001,
        )
        decreasing = self.analyzer.analyze(
            (record("c", 2_000_000_001, 3, sequence=3), record("d", 1_000_000_001, 5, sequence=2)),
            scope=SCOPE,
            fact_key="environment:temperature",
            subject="sensor:room-a",
            predicate="temperature-celsius",
            now_ns=2_000_000_001,
        )
        self.assertEqual(stable.direction, DriftDirection.STABLE)
        self.assertEqual(decreasing.direction, DriftDirection.DECREASING)

    def test_cross_scope_stale_gapped_and_nonnumeric_series_are_rejected(self) -> None:
        common = {
            "scope": SCOPE,
            "fact_key": "environment:temperature",
            "subject": "sensor:room-a",
            "predicate": "temperature-celsius",
            "now_ns": 10_000_000_000,
        }
        with self.assertRaises(TemporalChronoError):
            self.analyzer.analyze(
                (record("a", 9_000_000_000, 1, sequence=1), record("b", 9_500_000_000, 2, sequence=2, scope=OTHER_SCOPE)),
                **common,
            )
        with self.assertRaises(TemporalChronoError):
            self.analyzer.analyze(
                (record("old-a", 0, 1, sequence=1), record("old-b", 1, 2, sequence=2)),
                **{**common, "now_ns": 10_000_000_001},
            )
        with self.assertRaises(TemporalChronoError):
            self.analyzer.analyze(
                (record("gap-a", 0, 1, sequence=1), record("gap-b", 6_000_000_000, 2, sequence=2)),
                **common,
            )
        with self.assertRaises(TemporalChronoError):
            self.analyzer.analyze(
                (record("text-a", 9_000_000_000, "hot", sequence=1), record("text-b", 9_500_000_000, "warm", sequence=2)),
                **common,
            )

    def test_forecast_horizon_is_bounded_by_policy(self) -> None:
        with self.assertRaises(TemporalChronoError):
            self.analyzer.analyze(
                (record("a", 0, 1, sequence=1), record("b", 1_000_000_000, 2, sequence=2)),
                scope=SCOPE,
                fact_key="environment:temperature",
                subject="sensor:room-a",
                predicate="temperature-celsius",
                now_ns=1_000_000_000,
                forecast_horizons_seconds=(11.0,),
            )


if __name__ == "__main__":
    unittest.main()