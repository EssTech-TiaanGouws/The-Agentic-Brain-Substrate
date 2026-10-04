from __future__ import annotations

import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from substrate.contracts import ScopeVector
from src.swarm_core.model_catalog import ModelRole
from src.swarm_core.model_lifecycle import (
    ArtifactKind,
    Ed25519ArtifactManifestVerifier,
    ModelArtifactManifest,
    ModelArtifactRegistry,
    artifact_manifest_message,
)
from src.swarm_core.provenance_graph import (
    ProvenanceGraphCorruptionError,
    ProvenanceGraphDenied,
    ProvenanceGraphLinker,
    ProvenanceGraphOrphan,
)
from src.swarm_core.world_state import (
    WorldStateAdmissionDecision,
    WorldStateAdmissionDenied,
    WorldStateAssertion,
    WorldStateConflictError,
    WorldStateCorruptionError,
    WorldStateLedger,
)


SCOPE = ScopeVector("tenant-a", "user-a", "project-a", "workspace-a")


class ApprovingVerifier:
    def review(self, assertion, active_conflicts):
        conflict_ids = tuple(sorted(record.assertion.assertion_id for record in active_conflicts))
        return WorldStateAdmissionDecision(
            approved=True,
            verification_reference="block6-block8-review-001",
            superseded_assertion_ids=conflict_ids,
            resolution_reference="consensus-resolution-001" if conflict_ids else None,
        )


class DenyingVerifier:
    def review(self, assertion, active_conflicts):
        return WorldStateAdmissionDecision(False, "review-denied")


class IncompleteConflictVerifier:
    def review(self, assertion, active_conflicts):
        return WorldStateAdmissionDecision(True, "review-approved")


class FailOnceVerifier:
    def __init__(self) -> None:
        self.failed = False

    def review(self, assertion, active_conflicts):
        if not self.failed:
            self.failed = True
            raise RuntimeError("review provider unavailable")
        return ApprovingVerifier().review(assertion, active_conflicts)


class AllowLinkVerifier:
    def verify(self, record, artifact) -> str:
        return "link-review:accepted"


class DenyLinkVerifier:
    def verify(self, record, artifact) -> str:
        raise ProvenanceGraphDenied("link evidence was rejected")


def make_assertion(
    assertion_id: str = "assertion-1",
    *,
    scope: ScopeVector = SCOPE,
    value: object = "verified",
) -> WorldStateAssertion:
    return WorldStateAssertion(
        assertion_id=assertion_id,
        fact_key="workspace:status",
        scope=scope,
        subject="workspace",
        predicate="status",
        value=value,
        source_ref="source:fixture-1",
        source_sha256=hashlib.sha256(b"fixture source").hexdigest(),
        transaction_id=f"tx-{assertion_id}",
    )


def make_artifact_manifest() -> ModelArtifactManifest:
    return ModelArtifactManifest(
        artifact_id="specialist-test",
        version="1.0.0",
        kind=ArtifactKind.FULL_MODEL,
        role=ModelRole.SPECIALIST,
        relative_path="specialist.bin",
        artifact_sha256=hashlib.sha256(b"specialist artifact").hexdigest(),
        artifact_size_bytes=len(b"specialist artifact"),
        capability_ids=("text.summarize",),
        backend_id="backend.test",
        architecture_id="architecture.test",
        config_sha256=hashlib.sha256(b"config").hexdigest(),
        tokenizer_sha256=hashlib.sha256(b"tokenizer").hexdigest(),
        license_reference="license-review:test",
        provenance_reference="provenance:test",
        training_lineage="EXTERNAL_LICENSED:review-test",
        evaluation_reference="evaluation:test",
        approval_reference="approval:test",
        resource_profile_name="specialist-load",
        resource_domain="host-memory",
        reserved_bytes=1024,
    )


class WorldStateLedgerTests(unittest.TestCase):
    def test_append_query_is_scope_filtered_and_survives_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "world.sqlite3"
            with WorldStateLedger(path, admission_verifier=ApprovingVerifier()) as ledger:
                appended = ledger.append(make_assertion(value={"status": "ready", "count": 2}))
                self.assertEqual(appended.sequence, 1)
                self.assertEqual(len(ledger.query(SCOPE)), 1)
                other_scope = ScopeVector("tenant-a", "user-b", "project-a", "workspace-a")
                self.assertEqual(ledger.query(other_scope), ())

            with WorldStateLedger(path, admission_verifier=ApprovingVerifier()) as reopened:
                records = reopened.query(SCOPE, fact_key="workspace:status")

        self.assertEqual(records[0].assertion.value, {"count": 2, "status": "ready"})
        self.assertEqual(records[0].event_hash, appended.event_hash)

    def test_conflicting_fact_requires_explicit_reconciliation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "world.sqlite3"
            with WorldStateLedger(path, admission_verifier=IncompleteConflictVerifier()) as ledger:
                ledger.append(make_assertion("assertion-old", value="old"))
                with self.assertRaises(WorldStateConflictError):
                    ledger.append(make_assertion("assertion-new", value="new"))

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "world.sqlite3"
            with WorldStateLedger(path, admission_verifier=ApprovingVerifier()) as ledger:
                ledger.append(make_assertion("assertion-old", value="old"))
                ledger.append(make_assertion("assertion-new", value="new"))
                active = ledger.query(SCOPE)

        self.assertEqual(tuple(record.assertion.assertion_id for record in active), ("assertion-new",))
        self.assertEqual(active[0].superseded_assertion_ids, ("assertion-old",))
        self.assertEqual(active[0].resolution_reference, "consensus-resolution-001")

    def test_historical_record_lookup_is_scope_filtered_and_includes_superseded_facts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "world.sqlite3"
            with WorldStateLedger(path, admission_verifier=ApprovingVerifier()) as ledger:
                ledger.append(make_assertion("assertion-old", value="old"))
                ledger.append(make_assertion("assertion-new", value="new"))

                old_record = ledger.get_record("assertion-old", SCOPE)
                other_scope = ScopeVector("tenant-a", "user-b", "project-a", "workspace-a")
                hidden_record = ledger.get_record("assertion-old", other_scope)

        self.assertEqual(old_record.assertion.value, "old")
        self.assertEqual(old_record.assertion.assertion_id, "assertion-old")
        self.assertIsNone(hidden_record)


class ProvenanceGraphLinkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        root = Path(self.temporary_directory.name)
        self.ledger_path = root / "world.sqlite3"
        self.ledger = WorldStateLedger(
            self.ledger_path,
            admission_verifier=ApprovingVerifier(),
        )
        self.addCleanup(self.ledger.close)
        self.assertion_record = self.ledger.append(make_assertion())
        self.manifest = make_artifact_manifest()
        self.artifact_root = root / "artifacts"
        self.artifact_root.mkdir()
        (self.artifact_root / self.manifest.relative_path).write_bytes(b"specialist artifact")
        self.private_key = Ed25519PrivateKey.generate()
        public_key = self.private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        self.registry = ModelArtifactRegistry(
            root / "artifact-registry.sqlite3",
            self.artifact_root,
            manifest_verifier=Ed25519ArtifactManifestVerifier(public_key),
        )
        self.registry.register(
            self.manifest,
            self.private_key.sign(artifact_manifest_message(self.manifest)),
        )
        self.addCleanup(self.registry.close)
        self.graph_path = root / "provenance.sqlite3"

    def make_linker(self, verifier=AllowLinkVerifier()):
        return ProvenanceGraphLinker(
            self.graph_path,
            world_state_ledger=self.ledger,
            artifact_registry=self.registry,
            link_verifier=verifier,
        )

    def test_links_admitted_assertion_to_registered_artifact_source_and_transaction(self) -> None:
        with self.make_linker() as linker:
            record = linker.link_assertion(
                self.assertion_record.assertion.assertion_id,
                SCOPE,
                self.manifest.artifact_sha256,
            )
            self.assertEqual(record.assertion_id, self.assertion_record.assertion.assertion_id)
            self.assertEqual(record.artifact_id, self.manifest.artifact_id)
            self.assertEqual(record.ledger_event_hash, self.assertion_record.event_hash)
            self.assertEqual(
                tuple(edge.relation for edge in record.edges),
                ("SUPPORTS", "PRODUCED", "RECORDED", "ADMITTED"),
            )

    def test_link_is_idempotent_and_persists_across_restart(self) -> None:
        with self.make_linker() as linker:
            first = linker.link_assertion(
                self.assertion_record.assertion.assertion_id,
                SCOPE,
                self.manifest.artifact_sha256,
            )
            replay = linker.link_assertion(
                self.assertion_record.assertion.assertion_id,
                SCOPE,
                self.manifest.artifact_sha256,
            )
        self.assertEqual(first, replay)

        with self.make_linker() as reopened:
            records = reopened.query(SCOPE, assertion_id=self.assertion_record.assertion.assertion_id)
        self.assertEqual(records, (first,))

    def test_orphan_and_cross_scope_links_are_rejected(self) -> None:
        with self.make_linker() as linker:
            with self.assertRaises(ProvenanceGraphOrphan):
                linker.link_assertion("missing-assertion", SCOPE, self.manifest.artifact_sha256)
            other_scope = ScopeVector("tenant-a", "user-b", "project-a", "workspace-a")
            with self.assertRaises(ProvenanceGraphOrphan):
                linker.link_assertion(
                    self.assertion_record.assertion.assertion_id,
                    other_scope,
                    self.manifest.artifact_sha256,
                )
            self.assertEqual(linker.query(SCOPE), ())

    def test_link_verifier_denial_persists_nothing(self) -> None:
        with self.make_linker(DenyLinkVerifier()) as linker:
            with self.assertRaises(ProvenanceGraphDenied):
                linker.link_assertion(
                    self.assertion_record.assertion.assertion_id,
                    SCOPE,
                    self.manifest.artifact_sha256,
                )
            self.assertEqual(linker.query(SCOPE), ())

    def test_database_scope_column_tampering_is_detected(self) -> None:
        with self.make_linker() as linker:
            linker.link_assertion(
                self.assertion_record.assertion.assertion_id,
                SCOPE,
                self.manifest.artifact_sha256,
            )
        connection = sqlite3.connect(self.graph_path)
        connection.execute("DROP TRIGGER provenance_events_no_update")
        connection.execute("UPDATE provenance_events SET user_id = 'user-b'")
        connection.commit()
        connection.close()

        with self.assertRaises(ProvenanceGraphCorruptionError):
            self.make_linker()

    def test_link_query_detects_corruption_of_the_referenced_world_state(self) -> None:
        linker = self.make_linker()
        self.addCleanup(linker.close)
        linker.link_assertion(
            self.assertion_record.assertion.assertion_id,
            SCOPE,
            self.manifest.artifact_sha256,
        )
        connection = sqlite3.connect(self.ledger_path)
        connection.execute("DROP TRIGGER world_events_no_update")
        connection.execute("UPDATE world_events SET payload_json = '{}' ")
        connection.commit()
        connection.close()

        with self.assertRaises(ProvenanceGraphCorruptionError):
            linker.query(SCOPE)

    def test_denied_assertion_is_not_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "world.sqlite3"
            with WorldStateLedger(path, admission_verifier=DenyingVerifier()) as ledger:
                with self.assertRaises(WorldStateAdmissionDenied):
                    ledger.append(make_assertion())
                self.assertEqual(ledger.query(SCOPE), ())

    def test_existing_assertion_id_is_idempotent_but_not_reusable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "world.sqlite3"
            with WorldStateLedger(path, admission_verifier=ApprovingVerifier()) as ledger:
                first = ledger.append(make_assertion())
                replay = ledger.append(make_assertion())
                self.assertEqual(first, replay)
                with self.assertRaises(WorldStateConflictError):
                    ledger.append(make_assertion(value="different"))

    def test_database_tampering_fails_integrity_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "world.sqlite3"
            with WorldStateLedger(path, admission_verifier=ApprovingVerifier()) as ledger:
                ledger.append(make_assertion())
            connection = sqlite3.connect(path)
            connection.execute("DROP TRIGGER world_events_no_update")
            connection.execute("UPDATE world_events SET event_hash = ?", ("0" * 64,))
            connection.commit()
            connection.close()

            with self.assertRaises(WorldStateCorruptionError):
                WorldStateLedger(path, admission_verifier=ApprovingVerifier())

    def test_reviewer_exception_rolls_back_before_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "world.sqlite3"
            with WorldStateLedger(path, admission_verifier=FailOnceVerifier()) as ledger:
                with self.assertRaisesRegex(RuntimeError, "review provider unavailable"):
                    ledger.append(make_assertion())
                record = ledger.append(make_assertion())
                self.assertEqual(record.sequence, 1)


if __name__ == "__main__":
    unittest.main()