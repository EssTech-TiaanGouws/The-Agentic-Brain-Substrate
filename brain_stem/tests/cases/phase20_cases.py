from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from substrate.contracts import ScopeVector
from src.swarm_core.durable_ledger import ReceiptTransition
from src.swarm_core.model_catalog import ModelRole
from src.swarm_core.model_lifecycle import (
    ArtifactKind,
    Ed25519ArtifactManifestVerifier,
    ModelArtifactManifest,
    ModelArtifactRegistry,
    artifact_manifest_message,
)
from src.swarm_core.provenance_graph import (
    ProvenanceGraphLinker,
    ProvenanceLinkVerifier,
)
from src.swarm_core.transaction_coordinator import TransactionResult
from src.swarm_core.workspace_projection import (
    CommittedWorkspaceProjector,
    WorkspaceProjectionError,
)
from src.swarm_core.workspace_writer import WorkspaceWriteCommand
from src.swarm_core.lease_manager import ResourceProfile
from src.swarm_core.world_state import (
    WorldStateAdmissionDecision,
    WorldStateLedger,
)


SCOPE = ScopeVector("tenant-a", "user-a", "project-a", "workspace-a")
ARTIFACT_BYTES = b"trained-scratch-core-artifact"


class AcceptAdmission:
    def review(self, assertion, active_conflicts):
        conflict_ids = tuple(sorted(record.assertion.assertion_id for record in active_conflicts))
        return WorldStateAdmissionDecision(
            approved=True,
            verification_reference="world-state-review:accepted",
            superseded_assertion_ids=conflict_ids,
            resolution_reference="workspace-update-resolution" if conflict_ids else None,
        )


class LinkEvidenceVerifier:
    def verify(self, record, artifact) -> str:
        return "link-evidence:verified"


class RejectLinkEvidenceVerifier:
    def verify(self, record, artifact) -> str:
        raise PermissionError("link review is unavailable")


def make_command() -> WorkspaceWriteCommand:
    return WorkspaceWriteCommand(
        transaction_id="tx-projection",
        correlation_id="turn-projection",
        idempotency_key="idem-projection",
        scope=SCOPE,
        relative_path="reports/result.txt",
        format_key="text/plain",
        content=b"verified result",
        expected_current_sha256=None,
        resource_profile=ResourceProfile("workspace-write", "host-memory", 1024),
        proposal_release_id="release-test",
        proposal_artifact_sha256=hashlib.sha256(ARTIFACT_BYTES).hexdigest(),
    )


class CommittedWorkspaceProjectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        root = Path(self.temporary_directory.name)
        self.scope = SCOPE
        self.ledger = WorldStateLedger(
            root / "world.sqlite3",
            admission_verifier=AcceptAdmission(),
        )
        self.addCleanup(self.ledger.close)
        self.artifact_root = root / "artifacts"
        self.artifact_root.mkdir()
        (self.artifact_root / "core.bin").write_bytes(ARTIFACT_BYTES)
        self.private_key = Ed25519PrivateKey.generate()
        public_key = self.private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        self.registry = ModelArtifactRegistry(
            root / "artifacts.sqlite3",
            self.artifact_root,
            manifest_verifier=Ed25519ArtifactManifestVerifier(public_key),
        )
        self.addCleanup(self.registry.close)
        self.manifest = ModelArtifactManifest(
            artifact_id="stacey-core-projection-test",
            version="1.0.0",
            kind=ArtifactKind.FULL_MODEL,
            role=ModelRole.WORLD_MODEL_CORE,
            relative_path="core.bin",
            artifact_sha256=hashlib.sha256(ARTIFACT_BYTES).hexdigest(),
            artifact_size_bytes=len(ARTIFACT_BYTES),
            capability_ids=("core.reason",),
            backend_id="stacey.core.pytorch.v0",
            architecture_id="stacey.byte_transformer_seq2seq.v0",
            config_sha256=hashlib.sha256(b"config").hexdigest(),
            tokenizer_sha256=hashlib.sha256(b"tokenizer").hexdigest(),
            license_reference="stacey-owned",
            provenance_reference="scratch-run:test",
            training_lineage="STACEY_SCRATCH",
            evaluation_reference="evaluation:test",
            approval_reference="approval:test",
            resource_profile_name="core-load",
            resource_domain="host-memory",
            reserved_bytes=1024,
        )
        self.registry.register(
            self.manifest,
            self.private_key.sign(artifact_manifest_message(self.manifest)),
        )
        self.graph_path = root / "provenance.sqlite3"

    def make_graph(self, verifier=None):
        return ProvenanceGraphLinker(
            self.graph_path,
            world_state_ledger=self.ledger,
            artifact_registry=self.registry,
            link_verifier=verifier or LinkEvidenceVerifier(),
        )

    def test_successful_receipt_projects_idempotent_fact_and_artifact_link(self) -> None:
        command = make_command()
        result = TransactionResult("tx-projection", ReceiptTransition.OUTCOME, "sha256:" + hashlib.sha256(command.content).hexdigest())
        with self.make_graph() as graph:
            projector = CommittedWorkspaceProjector(
                world_state_ledger=self.ledger,
                provenance_graph=graph,
            )
            first = projector.project(command, result)
            replay = projector.project(command, result)

        self.assertEqual(first, replay)
        self.assertEqual(first.assertion.assertion.source_sha256, hashlib.sha256(command.content).hexdigest())
        self.assertEqual(first.provenance.artifact_sha256, self.manifest.artifact_sha256)
        self.assertEqual(first.provenance.ledger_event_hash, first.assertion.event_hash)
        self.assertEqual(self.ledger.query(SCOPE), (first.assertion,))

    def test_nonterminal_or_unbound_receipt_is_not_projected(self) -> None:
        command = make_command()
        prepared = TransactionResult("tx-projection", ReceiptTransition.PREPARED, "prepared")
        with self.make_graph() as graph:
            projector = CommittedWorkspaceProjector(
                world_state_ledger=self.ledger,
                provenance_graph=graph,
            )
            with self.assertRaises(WorkspaceProjectionError):
                projector.project(command, prepared)
            unbound = replace_command(command, proposal_artifact_sha256=None, proposal_release_id=None)
            successful = TransactionResult(
                "tx-projection",
                ReceiptTransition.OUTCOME,
                "sha256:" + hashlib.sha256(command.content).hexdigest(),
            )
            with self.assertRaises(WorkspaceProjectionError):
                projector.project(unbound, successful)
        self.assertEqual(self.ledger.query(SCOPE), ())

    def test_graph_review_failure_is_retryable_after_fact_append(self) -> None:
        command = make_command()
        result = TransactionResult("tx-projection", ReceiptTransition.OUTCOME, "sha256:" + hashlib.sha256(command.content).hexdigest())
        with self.make_graph(RejectLinkEvidenceVerifier()) as graph:
            projector = CommittedWorkspaceProjector(
                world_state_ledger=self.ledger,
                provenance_graph=graph,
            )
            with self.assertRaises(WorkspaceProjectionError) as error:
                projector.project(command, result)
            assertion_id = error.exception.assertion_id
            self.assertIsNotNone(assertion_id)
            self.assertEqual(len(self.ledger.query(SCOPE)), 1)

        with self.make_graph() as retry_graph:
            retried = CommittedWorkspaceProjector(
                world_state_ledger=self.ledger,
                provenance_graph=retry_graph,
            ).project(command, result)
        self.assertEqual(retried.assertion.assertion.assertion_id, assertion_id)
        self.assertEqual(retried.assertion.sequence, 1)


def replace_command(command: WorkspaceWriteCommand, **changes) -> WorkspaceWriteCommand:
    from dataclasses import replace

    return replace(command, **changes)


if __name__ == "__main__":
    unittest.main()