from __future__ import annotations

import math
from dataclasses import dataclass
from threading import RLock
from time import time_ns
from typing import Callable

from substrate.contracts import ScopeVector

from .world_state import WorldStateLedger, WorldStateRecord


class RelationalContextError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RelationalContextPolicy:
    policy_version: str
    half_life_ns: int
    minimum_retention_score: float
    maximum_results: int

    def __post_init__(self) -> None:
        if not isinstance(self.policy_version, str) or not self.policy_version.strip():
            raise RelationalContextError("policy_version must be a non-empty string")
        if isinstance(self.half_life_ns, bool) or not isinstance(self.half_life_ns, int) or self.half_life_ns <= 0:
            raise RelationalContextError("half_life_ns must be a positive integer")
        if (
            isinstance(self.minimum_retention_score, bool)
            or not isinstance(self.minimum_retention_score, (int, float))
            or not math.isfinite(self.minimum_retention_score)
            or not 0.0 <= self.minimum_retention_score <= 1.0
        ):
            raise RelationalContextError("minimum_retention_score must be finite and between 0 and 1")
        if isinstance(self.maximum_results, bool) or not isinstance(self.maximum_results, int) or self.maximum_results <= 0:
            raise RelationalContextError("maximum_results must be a positive integer")


@dataclass(frozen=True, slots=True)
class RetrievedRelation:
    record: WorldStateRecord
    decay_score: float


class RelationalContextIndex:
    """Read-only Block 7 view over accepted, scoped FB-004 relation assertions."""

    def __init__(
        self,
        *,
        world_state_ledger: WorldStateLedger,
        policy: RelationalContextPolicy,
        clock: Callable[[], int] = time_ns,
    ) -> None:
        if not isinstance(world_state_ledger, WorldStateLedger):
            raise ValueError("world_state_ledger must be a WorldStateLedger")
        if not isinstance(policy, RelationalContextPolicy):
            raise ValueError("policy must be a RelationalContextPolicy")
        if not callable(clock):
            raise ValueError("clock must be callable")
        self._ledger = world_state_ledger
        self._policy = policy
        self._clock = clock
        self._lock = RLock()

    def retrieve(
        self,
        scope: ScopeVector,
        *,
        subject: str | None = None,
        predicate: str | None = None,
    ) -> tuple[RetrievedRelation, ...]:
        if not isinstance(scope, ScopeVector) or not scope.is_complete():
            raise RelationalContextError("A complete four-field scope is required")
        for field, value in (("subject", subject), ("predicate", predicate)):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise RelationalContextError(f"{field} must be a non-empty string or None")
        now_ns = self._clock()
        if isinstance(now_ns, bool) or not isinstance(now_ns, int) or now_ns < 0:
            raise RelationalContextError("clock must return a non-negative integer timestamp")

        with self._lock:
            records = self._ledger.query(scope)
            ranked: list[RetrievedRelation] = []
            for record in records:
                assertion = record.assertion
                if not assertion.fact_key.startswith("relation:"):
                    continue
                if subject is not None and assertion.subject != subject:
                    continue
                if predicate is not None and assertion.predicate != predicate:
                    continue
                age_ns = now_ns - record.occurred_at_ns
                if age_ns < 0:
                    raise RelationalContextError("world-state record timestamp is in the future")
                score = math.exp(-math.log(2.0) * age_ns / self._policy.half_life_ns)
                if score >= self._policy.minimum_retention_score:
                    ranked.append(RetrievedRelation(record, score))

            ranked.sort(
                key=lambda item: (
                    -item.decay_score,
                    -item.record.sequence,
                    item.record.assertion.fact_key,
                )
            )
            return tuple(ranked[: self._policy.maximum_results])