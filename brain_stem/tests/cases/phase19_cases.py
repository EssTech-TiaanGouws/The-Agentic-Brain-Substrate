from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from substrate.contracts import ScopeVector
from src.swarm_core.relational_context import (
    RelationalContextError,
    RelationalContextIndex,
    RelationalContextPolicy,
)
from src.swarm_core.world_state import (
    WorldStateAdmissionDecision,
    WorldStateAssertion,
    WorldStateLedger,
)


SCOPE = ScopeVector("tenant-a", "user-a", "project-a", "workspace-a")
OTHER_SCOPE = ScopeVector("tenant-a", "user-b", "project-a", "workspace-a")


class AcceptRelations:
    def review(self, assertion, active_conflicts):
        return WorldStateAdmissionDecision(True, "relation-review:accepted")


class RelationalContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.now_ns = 1_000
        self.ledger = WorldStateLedger(
            Path(self.temporary_directory.name) / "world.sqlite3",
            admission_verifier=AcceptRelations(),
            clock=lambda: self.now_ns,
        )
        self.addCleanup(self.ledger.close)
        self.policy = RelationalContextPolicy(
            policy_version="relation-decay-v1",
            half_life_ns=100,
            minimum_retention_score=0.3,
            maximum_results=10,
        )
        self.index = RelationalContextIndex(
            world_state_ledger=self.ledger,
            policy=self.policy,
            clock=lambda: self.now_ns,
        )

    def append_relation(
        self,
        assertion_id: str,
        *,
        observed_at_ns: int,
        scope: ScopeVector = SCOPE,
        subject: str = "document-a",
        predicate: str = "mentions",
        value: object = "entity-a",
        fact_key: str | None = None,
    ) -> None:
        self.now_ns = observed_at_ns
        self.ledger.append(
            WorldStateAssertion(
                assertion_id=assertion_id,
                fact_key=fact_key or f"relation:{assertion_id}",
                scope=scope,
                subject=subject,
                predicate=predicate,
                value=value,
                source_ref=f"source:{assertion_id}",
                source_sha256=hashlib.sha256(assertion_id.encode("utf-8")).hexdigest(),
                transaction_id=f"tx:{assertion_id}",
            )
        )

    def test_relations_rank_by_exponential_decay_and_drop_stale_entries(self) -> None:
        self.append_relation("relation-old", observed_at_ns=800, value="old")
        self.append_relation("relation-mid", observed_at_ns=900, value="mid")
        self.append_relation("relation-new", observed_at_ns=1_000, value="new")

        results = self.index.retrieve(SCOPE)

        self.assertEqual(
            tuple(item.record.assertion.assertion_id for item in results),
            ("relation-new", "relation-mid"),
        )
        self.assertAlmostEqual(results[0].decay_score, 1.0)
        self.assertAlmostEqual(results[1].decay_score, 0.5)

    def test_lookup_is_scope_subject_predicate_filtered_and_bounded(self) -> None:
        self.append_relation("wanted", observed_at_ns=1_000, subject="document-a")
        self.append_relation("other-subject", observed_at_ns=1_000, subject="document-b")
        self.append_relation("other-predicate", observed_at_ns=1_000, predicate="owned-by")
        self.append_relation("other-scope", observed_at_ns=1_000, scope=OTHER_SCOPE)
        limited = RelationalContextIndex(
            world_state_ledger=self.ledger,
            policy=RelationalContextPolicy(
                policy_version="relation-decay-v1",
                half_life_ns=100,
                minimum_retention_score=0.0,
                maximum_results=1,
            ),
            clock=lambda: self.now_ns,
        )

        results = limited.retrieve(SCOPE, subject="document-a", predicate="mentions")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].record.assertion.assertion_id, "wanted")

    def test_nonrelation_records_are_not_returned(self) -> None:
        self.append_relation("ordinary-fact", observed_at_ns=1_000, fact_key="workspace:status")
        self.assertEqual(self.index.retrieve(SCOPE), ())

    def test_future_ledger_timestamp_is_rejected(self) -> None:
        self.append_relation("future", observed_at_ns=1_100)
        self.now_ns = 1_000

        with self.assertRaises(RelationalContextError):
            self.index.retrieve(SCOPE)

    def test_policy_rejects_unbounded_or_invalid_retention(self) -> None:
        with self.assertRaises(RelationalContextError):
            RelationalContextPolicy(
                policy_version="invalid",
                half_life_ns=0,
                minimum_retention_score=0.3,
                maximum_results=10,
            )


if __name__ == "__main__":
    unittest.main()