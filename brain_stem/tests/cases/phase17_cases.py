from __future__ import annotations

import base64
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from time import time, time_ns

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from substrate.contracts import ScopeVector
from src.swarm_core.collective_scheduler import (
    CapabilityOutput,
    CollectiveExecutionReport,
    StepState,
)
from src.swarm_core.durable_ledger import DurableReceiptLedger, ReceiptTransition
from src.swarm_core.format_adapters import DocumentFormatRegistry
from src.swarm_core.identity import (
    Ed25519ScopeAuthorizationVerifier,
    SignedScopeGrant,
    intent_authorization_digest,
    scope_grant_message,
)
from src.swarm_core.lease_manager import (
    ResourceLeaseManager,
    ResourcePolicy,
    ResourceProfile,
    ResourceSnapshot,
)
from src.swarm_core.transaction_coordinator import TransactionRequest, StagedOperation
from src.swarm_core.workspace_action import (
    ReviewedWorkspaceProposalExecutor,
    WorkspaceActionProposalError,
)
from src.swarm_core.workspace_writer import (
    ConfiguredWorkspaceResolver,
    WorkspaceWriteCommand,
    WorkspaceWriteService,
)


SCOPE = ScopeVector("tenant-a", "user-a", "project-a", "workspace-a")
AUDIENCE = "reviewed-workspace-action-test"
PROPOSAL_ARTIFACT_SHA256 = hashlib.sha256(b"approved-proposal-artifact").hexdigest()


class TextHandler:
    format_key = "text/plain"

    def validate(self, content: bytes) -> None:
        if not content:
            raise ValueError("empty document")

    def prepare(self, content: bytes) -> bytes:
        return content

    def verify(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()


class OrderedVerifier:
    def __init__(self) -> None:
        self.events: list[str] = []

    def verify_adversarial(self, request: TransactionRequest, stage: StagedOperation) -> str:
        self.events.append("block-6")
        return "block-6:accepted"

    def verify_epistemic(self, request: TransactionRequest, stage: StagedOperation) -> str:
        self.events.append("block-8")
        return "block-8:accepted"


def make_report(
    content: bytes,
    *,
    critic_reference: str | None = "critic:accepted",
    auditor_reference: str | None = "auditor:accepted",
) -> CollectiveExecutionReport:
    payload = json.dumps(
        {
            "relative_path": "notes.txt",
            "format_key": "text/plain",
            "content_base64": base64.b64encode(content).decode("ascii"),
            "expected_current_sha256": hashlib.sha256(b"before").hexdigest(),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    output = CapabilityOutput(
        step_id="write-step",
        capability_id="workspace.write.proposal",
        state=StepState.SUCCEEDED,
        payload=payload,
        error_code=None,
        provenance_references=("source:approved-input",),
        critic_reference=critic_reference,
        auditor_reference=auditor_reference,
        artifact_sha256=PROPOSAL_ARTIFACT_SHA256,
    )
    return CollectiveExecutionReport(
        transaction_id="tx-write-proposal",
        correlation_id="turn-write-proposal",
        release_id="release-approved",
        scope=SCOPE,
        clarification_required=False,
        outputs=(output,),
    )


class ReviewedWorkspaceProposalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        root = Path(self.temporary_directory.name)
        self.workspace_root = root / "workspace"
        self.workspace_root.mkdir()
        self.target = self.workspace_root / "notes.txt"
        self.target.write_bytes(b"before")
        self.private_key = Ed25519PrivateKey.generate()
        public_key = self.private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        self.authorization_verifier = Ed25519ScopeAuthorizationVerifier(
            public_key,
            audience=AUDIENCE,
        )
        self.receipt_ledger = DurableReceiptLedger(root / "receipts.jsonl", file_mode=0o600)
        lease_manager = ResourceLeaseManager(
            ResourcePolicy("workspace-write", "host-memory", 0, 1_000_000),
            lambda: ResourceSnapshot("host-memory", 500_000, time_ns()),
        )
        self.stage_verifier = OrderedVerifier()
        self.writer = WorkspaceWriteService(
            workspace_resolver=ConfiguredWorkspaceResolver({SCOPE: self.workspace_root}),
            format_registry=DocumentFormatRegistry((TextHandler(),)),
            authorization_verifier=self.authorization_verifier,
            lease_manager=lease_manager,
            receipt_ledger=self.receipt_ledger,
            stage_verifier=self.stage_verifier,
            max_content_bytes=4096,
        )
        self.executor = ReviewedWorkspaceProposalExecutor(
            writer=self.writer,
            resource_profile=ResourceProfile("workspace-write", "host-memory", 1024),
            allowed_format_keys=("text/plain",),
            maximum_payload_bytes=2048,
            maximum_content_bytes=1024,
        )

    def signed_authorization(self, content: bytes) -> SignedScopeGrant:
        command = WorkspaceWriteCommand(
            transaction_id="tx-write-proposal",
            correlation_id="turn-write-proposal",
            idempotency_key="fixture-idempotency-key",
            scope=SCOPE,
            relative_path="notes.txt",
            format_key="text/plain",
            content=content,
            expected_current_sha256=hashlib.sha256(b"before").hexdigest(),
            resource_profile=ResourceProfile("workspace-write", "host-memory", 1024),
            proposal_release_id="release-approved",
            proposal_artifact_sha256=PROPOSAL_ARTIFACT_SHA256,
        )
        prepared = self.writer.prepare(command)
        intent = self.writer.confirmation_intent(command, prepared)
        now = int(time())
        unsigned = SignedScopeGrant(
            scope=SCOPE,
            action=intent.action,
            transaction_id=intent.transaction_id,
            correlation_id=intent.correlation_id,
            audience=AUDIENCE,
            issued_at=now,
            expires_at=now + 60,
            grant_id="operator-confirmation:1",
            request_sha256=intent_authorization_digest(intent),
            signature=b"",
        )
        return replace(unsigned, signature=self.private_key.sign(scope_grant_message(unsigned)))

    def test_reviewed_proposal_requires_separate_signature_and_commits_with_receipt(self) -> None:
        content = b"approved proposal content"
        result = self.executor.execute(make_report(content), self.signed_authorization(content))

        self.assertTrue(result.succeeded)
        self.assertEqual(self.target.read_bytes(), content)
        self.assertEqual(self.stage_verifier.events, ["block-6", "block-8"])
        transaction = self.receipt_ledger.get_transaction("tx-write-proposal")
        self.assertEqual(
            tuple(event.transition for event in transaction.events),
            (ReceiptTransition.PREPARED, ReceiptTransition.OUTCOME),
        )

    def test_missing_critic_or_auditor_evidence_never_calls_writer(self) -> None:
        class RecordingWriter:
            def __init__(self) -> None:
                self.calls = []

            def execute(self, command, authorization):
                self.calls.append((command, authorization))
                raise AssertionError("writer must not be called")

        for critic, auditor in ((None, "auditor:ok"), ("critic:ok", " ")):
            with self.subTest(critic=critic, auditor=auditor):
                writer = RecordingWriter()
                executor = ReviewedWorkspaceProposalExecutor(
                    writer=writer,
                    resource_profile=ResourceProfile("workspace-write", "host-memory", 1024),
                )
                with self.assertRaises(WorkspaceActionProposalError):
                    executor.execute(
                        make_report(b"proposal", critic_reference=critic, auditor_reference=auditor),
                        self.signed_authorization(b"proposal"),
                    )
                self.assertEqual(writer.calls, [])

    def test_signature_for_different_content_is_rejected_without_mutation(self) -> None:
        authorization = self.signed_authorization(b"approved content")
        report = make_report(b"changed after approval")

        with self.assertRaises(PermissionError):
            self.executor.execute(report, authorization)

        self.assertEqual(self.target.read_bytes(), b"before")
        self.assertIsNone(self.receipt_ledger.get_transaction("tx-write-proposal"))

    def test_duplicate_json_fields_and_path_traversal_are_rejected(self) -> None:
        duplicate_payload = (
            b'{"relative_path":"notes.txt","relative_path":"other.txt",'
            b'"format_key":"text/plain","content_base64":"eA==",'
            b'"expected_current_sha256":null}'
        )
        report = make_report(b"x")
        duplicate_output = replace(report.outputs[0], payload=duplicate_payload)
        with self.assertRaises(WorkspaceActionProposalError):
            self.executor.execute(replace(report, outputs=(duplicate_output,)), self.signed_authorization(b"x"))

        traversal = make_report(b"x")
        payload = json.loads(traversal.outputs[0].payload)
        payload["relative_path"] = "../outside.txt"
        traversal_output = replace(
            traversal.outputs[0],
            payload=json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        )
        with self.assertRaises(WorkspaceActionProposalError):
            self.executor.execute(
                replace(traversal, outputs=(traversal_output,)),
                self.signed_authorization(b"x"),
            )
        self.assertEqual(self.target.read_bytes(), b"before")


if __name__ == "__main__":
    unittest.main()