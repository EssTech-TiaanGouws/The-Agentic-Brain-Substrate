from __future__ import annotations

import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from time import time, time_ns

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from substrate.contracts import ScopeVector
from src.swarm_core.durable_ledger import DurableReceiptLedger, ReceiptTransition
from src.swarm_core.format_adapters import DocumentFormatRegistry
from src.swarm_core.identity import (
    AuthorizationError,
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
from src.swarm_core.transaction_coordinator import (
    TransactionAborted,
    TransactionRequest,
    StagedOperation,
    UnresolvedTransactionError,
)
from src.swarm_core.workspace_writer import (
    ConfiguredWorkspaceResolver,
    WorkspacePathRejected,
    WorkspaceWriteCommand,
    WorkspaceWriteService,
)


SCOPE = ScopeVector("tenant-a", "user-a", "project-a", "workspace-a")
AUDIENCE = "workspace-writer-test"


class FailTerminalLedger(DurableReceiptLedger):
    def __init__(self, path: Path) -> None:
        self.write_count = 0
        super().__init__(path, file_mode=0o600)

    def _persist_line(self, line: bytes) -> None:
        self.write_count += 1
        if self.write_count == 2:
            raise OSError("injected terminal receipt failure")
        super()._persist_line(line)


class TextHandler:
    format_key = "text/plain"

    def validate(self, content: bytes) -> None:
        if not content:
            raise ValueError("empty text is not accepted")

    def prepare(self, content: bytes) -> bytes:
        return content

    def verify(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()


class OrderedVerifier:
    def __init__(self, fail_at: str | None = None) -> None:
        self.fail_at = fail_at
        self.events: list[str] = []

    def verify_adversarial(self, request: TransactionRequest, stage: StagedOperation) -> str:
        self.events.append("block-6")
        if self.fail_at == "block-6":
            raise RuntimeError("Block 6 rejected staged content")
        return "block-6:verified"

    def verify_epistemic(self, request: TransactionRequest, stage: StagedOperation) -> str:
        self.events.append("block-8")
        if self.fail_at == "block-8":
            raise RuntimeError("Block 8 rejected staged content")
        return "block-8:verified"


def make_command(
    content: bytes,
    *,
    transaction_id: str = "write-1",
    relative_path: str = "notes.txt",
    expected_current_sha256: str | None = None,
    scope: ScopeVector = SCOPE,
) -> WorkspaceWriteCommand:
    return WorkspaceWriteCommand(
        transaction_id=transaction_id,
        correlation_id=f"turn-{transaction_id}",
        idempotency_key=f"idem-{transaction_id}",
        scope=scope,
        relative_path=relative_path,
        format_key="text/plain",
        content=content,
        expected_current_sha256=expected_current_sha256,
        resource_profile=ResourceProfile("workspace-write", "host-memory", 1024),
    )


def build_service(
    root: Path,
    receipt_path: Path,
    *,
    fail_at: str | None = None,
    ledger_override: DurableReceiptLedger | None = None,
):
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    authorization_verifier = Ed25519ScopeAuthorizationVerifier(
        public_key,
        audience=AUDIENCE,
    )
    lease_manager = ResourceLeaseManager(
        ResourcePolicy("workspace-write", "host-memory", 0, 5_000_000_000),
        lambda: ResourceSnapshot("host-memory", 1_000_000, time_ns()),
    )
    ledger = ledger_override or DurableReceiptLedger(receipt_path, file_mode=0o600)
    stage_verifier = OrderedVerifier(fail_at)
    service = WorkspaceWriteService(
        workspace_resolver=ConfiguredWorkspaceResolver({SCOPE: root}),
        format_registry=DocumentFormatRegistry((TextHandler(),)),
        authorization_verifier=authorization_verifier,
        lease_manager=lease_manager,
        receipt_ledger=ledger,
        stage_verifier=stage_verifier,
        max_content_bytes=4096,
    )
    return service, ledger, private_key, stage_verifier


def sign_command(service: WorkspaceWriteService, command: WorkspaceWriteCommand, private_key) -> SignedScopeGrant:
    prepared = service.prepare(command)
    intent = service.confirmation_intent(command, prepared)
    now = int(time())
    unsigned = SignedScopeGrant(
        scope=command.scope,
        action=intent.action,
        transaction_id=command.transaction_id,
        correlation_id=command.correlation_id,
        audience=AUDIENCE,
        issued_at=now,
        expires_at=now + 60,
        grant_id=f"grant-{command.transaction_id}",
        request_sha256=intent_authorization_digest(intent),
        signature=b"",
    )
    return replace(unsigned, signature=private_key.sign(scope_grant_message(unsigned)))


class WorkspaceWriteServiceTests(unittest.TestCase):
    def test_signed_write_uses_receipt_order_and_atomic_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "workspace"
            root.mkdir()
            target = root / "notes.txt"
            target.write_bytes(b"before")
            service, ledger, private_key, verifier = build_service(root, Path(temporary_directory) / "receipts.jsonl")
            command = make_command(
                b"after",
                expected_current_sha256=hashlib.sha256(b"before").hexdigest(),
            )

            result = service.execute(command, sign_command(service, command, private_key))

            self.assertTrue(result.succeeded)
            self.assertEqual(target.read_bytes(), b"after")
            self.assertEqual(verifier.events, ["block-6", "block-8"])
            self.assertEqual(
                tuple(event.transition for event in ledger.get_transaction("write-1").events),
                (ReceiptTransition.PREPARED, ReceiptTransition.OUTCOME),
            )

    def test_verifier_rejection_restores_original_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "workspace"
            root.mkdir()
            target = root / "notes.txt"
            target.write_bytes(b"before")
            service, ledger, private_key, _ = build_service(
                root,
                Path(temporary_directory) / "receipts.jsonl",
                fail_at="block-6",
            )
            command = make_command(
                b"after",
                expected_current_sha256=hashlib.sha256(b"before").hexdigest(),
            )

            with self.assertRaises(TransactionAborted):
                service.execute(command, sign_command(service, command, private_key))

            self.assertEqual(target.read_bytes(), b"before")
            self.assertEqual(ledger.get_transaction("write-1").state, ReceiptTransition.COMPENSATED)

    def test_signed_confirmation_cannot_be_reused_for_different_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "workspace"
            root.mkdir()
            service, ledger, private_key, _ = build_service(root, Path(temporary_directory) / "receipts.jsonl")
            original = make_command(b"approved")
            grant = sign_command(service, original, private_key)
            modified = replace(original, content=b"not approved")

            with self.assertRaises(AuthorizationError):
                service.execute(modified, grant)

            self.assertEqual(ledger.events(), ())

    def test_parent_symlink_and_path_traversal_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory)
            root = base / "workspace"
            outside = base / "outside"
            root.mkdir()
            outside.mkdir()
            (outside / "victim.txt").write_bytes(b"safe")
            (root / "linked").symlink_to(outside, target_is_directory=True)
            service, ledger, private_key, _ = build_service(root, base / "receipts.jsonl")
            command = make_command(b"overwrite", relative_path="linked/victim.txt", expected_current_sha256=None)

            with self.assertRaises(TransactionAborted):
                service.execute(command, sign_command(service, command, private_key))
            with self.assertRaises(WorkspacePathRejected):
                make_command(b"bad", relative_path="../victim.txt")

            self.assertEqual((outside / "victim.txt").read_bytes(), b"safe")

    def test_restart_recovery_restores_existing_file_after_receipt_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory)
            root = base / "workspace"
            root.mkdir()
            target = root / "notes.txt"
            target.write_bytes(b"before")
            receipt_path = base / "receipts.jsonl"
            failed_ledger = FailTerminalLedger(receipt_path)
            service, _, private_key, _ = build_service(root, receipt_path, ledger_override=failed_ledger)
            command = make_command(
                b"after",
                expected_current_sha256=hashlib.sha256(b"before").hexdigest(),
            )

            with self.assertRaises(UnresolvedTransactionError):
                service.execute(command, sign_command(service, command, private_key))
            self.assertEqual(target.read_bytes(), b"after")
            service.close()

            recovery_ledger = DurableReceiptLedger(receipt_path, file_mode=0o600)
            recovery_service, _, _, _ = build_service(
                root,
                receipt_path,
                ledger_override=recovery_ledger,
            )
            recovered = recovery_service.reconcile_unresolved()

            self.assertEqual(recovered, (("write-1", ReceiptTransition.COMPENSATED),))
            self.assertEqual(target.read_bytes(), b"before")
            self.assertEqual(recovery_ledger.get_transaction("write-1").state, ReceiptTransition.COMPENSATED)

    def test_restart_recovery_quarantines_ambiguous_new_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory)
            root = base / "workspace"
            root.mkdir()
            target = root / "new.txt"
            receipt_path = base / "receipts.jsonl"
            failed_ledger = FailTerminalLedger(receipt_path)
            service, _, private_key, _ = build_service(root, receipt_path, ledger_override=failed_ledger)
            command = make_command(b"created", relative_path="new.txt")

            with self.assertRaises(UnresolvedTransactionError):
                service.execute(command, sign_command(service, command, private_key))
            service.close()

            recovery_ledger = DurableReceiptLedger(receipt_path, file_mode=0o600)
            recovery_service, _, _, _ = build_service(
                root,
                receipt_path,
                ledger_override=recovery_ledger,
            )
            recovered = recovery_service.reconcile_unresolved()

            self.assertEqual(recovered, (("write-1", ReceiptTransition.INDETERMINATE),))
            self.assertEqual(target.read_bytes(), b"created")
            self.assertEqual(recovery_ledger.get_transaction("write-1").state, ReceiptTransition.INDETERMINATE)