from __future__ import annotations

import unittest

from src.swarm_core.volatility_sensor import (
    CircuitState,
    VolatilityPolicy,
    VolatilitySample,
    VolatilitySensor,
    VolatilitySensorError,
)


class VolatilitySensorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now_ns = 100
        self.policy = VolatilityPolicy(
            policy_version="volatility-v1",
            maximum_sample_age_ns=20,
            queue_trip_ratio=0.8,
            queue_recovery_ratio=0.5,
            resource_trip_ratio=0.9,
            resource_recovery_ratio=0.7,
            cooldown_ns=10,
            urgency_gain=2.0,
            maximum_attention_multiplier=3.0,
            high_urgency_threshold=0.8,
            high_urgency_multiplier=2.5,
        )
        self.sensor = VolatilitySensor(self.policy, clock=lambda: self.now_ns)

    def sample(
        self,
        *,
        observed_at_ns: int = 100,
        queue_depth: int = 2,
        resource_demand_bytes: int = 30,
        prompt_urgency: float = 0.2,
        probe_succeeded: bool | None = None,
    ) -> VolatilitySample:
        return VolatilitySample(
            observed_at_ns=observed_at_ns,
            queue_depth=queue_depth,
            queue_capacity=10,
            resource_demand_bytes=resource_demand_bytes,
            resource_capacity_bytes=100,
            prompt_urgency=prompt_urgency,
            probe_succeeded=probe_succeeded,
        )

    def test_high_urgency_scales_attention_within_the_configured_cap(self) -> None:
        decision = self.sensor.evaluate(self.sample(prompt_urgency=0.9))

        self.assertEqual(decision.policy_version, "volatility-v1")
        self.assertEqual(decision.circuit_state, CircuitState.CLOSED)
        self.assertEqual(decision.attention_multiplier, 2.8)
        self.assertTrue(decision.allow_new_work)
        self.assertFalse(decision.allow_probe_work)

    def test_stale_or_future_sample_opens_the_brake(self) -> None:
        stale = self.sensor.evaluate(self.sample(observed_at_ns=70))
        self.assertEqual(stale.circuit_state, CircuitState.OPEN)
        self.assertFalse(stale.allow_new_work)
        self.assertEqual(stale.reason, "stale-or-future-sample")

        future_sensor = VolatilitySensor(self.policy, clock=lambda: 100)
        future = future_sensor.evaluate(self.sample(observed_at_ns=101))
        self.assertEqual(future.circuit_state, CircuitState.OPEN)

    def test_queue_or_resource_pressure_trips_the_circuit(self) -> None:
        queue_trip = self.sensor.evaluate(self.sample(queue_depth=8))
        self.assertEqual(queue_trip.circuit_state, CircuitState.OPEN)
        self.assertEqual(queue_trip.reason, "pressure-trip")

        resource_sensor = VolatilitySensor(self.policy, clock=lambda: self.now_ns)
        resource_trip = resource_sensor.evaluate(self.sample(resource_demand_bytes=90))
        self.assertEqual(resource_trip.circuit_state, CircuitState.OPEN)

    def test_cooldown_requires_safe_sample_and_successful_half_open_probe(self) -> None:
        self.sensor.evaluate(self.sample(queue_depth=9))
        self.now_ns += self.policy.cooldown_ns

        half_open = self.sensor.evaluate(self.sample(queue_depth=2, resource_demand_bytes=40))
        self.assertEqual(half_open.circuit_state, CircuitState.HALF_OPEN)
        self.assertFalse(half_open.allow_new_work)
        self.assertTrue(half_open.allow_probe_work)

        pending = self.sensor.evaluate(self.sample(queue_depth=2, resource_demand_bytes=40))
        self.assertEqual(pending.circuit_state, CircuitState.HALF_OPEN)
        self.assertFalse(pending.allow_new_work)

        recovered = self.sensor.evaluate(
            self.sample(queue_depth=2, resource_demand_bytes=40, probe_succeeded=True)
        )
        self.assertEqual(recovered.circuit_state, CircuitState.CLOSED)
        self.assertTrue(recovered.allow_new_work)
        self.assertFalse(recovered.allow_probe_work)

    def test_failed_probe_reopens_the_circuit(self) -> None:
        self.sensor.evaluate(self.sample(resource_demand_bytes=95))
        self.now_ns += self.policy.cooldown_ns
        self.sensor.evaluate(self.sample(resource_demand_bytes=40))

        failed = self.sensor.evaluate(
            self.sample(resource_demand_bytes=40, probe_succeeded=False)
        )
        self.assertEqual(failed.circuit_state, CircuitState.OPEN)
        self.assertFalse(failed.allow_new_work)

    def test_invalid_policy_hysteresis_and_nonfinite_urgency_are_rejected(self) -> None:
        with self.assertRaises(VolatilitySensorError):
            VolatilityPolicy(
                policy_version="invalid",
                maximum_sample_age_ns=10,
                queue_trip_ratio=0.5,
                queue_recovery_ratio=0.5,
                resource_trip_ratio=0.9,
                resource_recovery_ratio=0.7,
                cooldown_ns=1,
                urgency_gain=1.0,
                maximum_attention_multiplier=2.0,
                high_urgency_threshold=0.8,
                high_urgency_multiplier=1.5,
            )
        with self.assertRaises(VolatilitySensorError):
            self.sample(prompt_urgency=float("nan"))


if __name__ == "__main__":
    unittest.main()