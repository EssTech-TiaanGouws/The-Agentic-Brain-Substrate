from __future__ import annotations

import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from unittest.mock import patch

from src.swarm_core.durable_ledger import (
    DurableReceiptLedger,
    LedgerCorruptionError,
    LedgerPersistenceError,
    ReceiptConflictError,
    ReceiptTransition,
)
from src.swarm_core.lease_manager import (
    ResourceAdmissionDenied,
    ResourceConfigurationError,
    ResourceLeaseManager,
    ResourcePolicy,
    ResourceProfile,
    ResourceSnapshot,
    ResourceSnapshotError,
)
from substrate.contracts import ScopeVector

RECEIPT_FILE_MODE = 0o600


class MutableSnapshotProvider:
    def __init__(self, snapshot: ResourceSnapshot) -> None:
        self.snapshot = snapshot

    def __call__(self) -> ResourceSnapshot:
        return self.snapshot


class FailingWriteLedger(DurableReceiptLedger):
    def __init__(self, storage_path: Path, fail_on_write: int) -> None:
        self._write_count = 0
        self._fail_on_write = fail_on_write
        super().__init__(storage_path, file_mode=RECEIPT_FILE_MODE, clock=lambda: 1)

    def _persist_line(self, line: bytes) -> None:
        self._write_count += 1
        if self._write_count == self._fail_on_write:
            raise OSError("injected persistence failure")
        super()._persist_line(line)


class Phase2ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.scope = ScopeVector(
            tenant_id="tenant-fixture",
            user_id="user-fixture",
            project_id="project-fixture",
            workspace_id="workspace-fixture",
        )

    def make_ledger(self, filename: str = "receipts.jsonl") -> DurableReceiptLedger:
        return DurableReceiptLedger(
            self.root / filename,
            file_mode=RECEIPT_FILE_MODE,
            clock=lambda: 10,
        )

    def append_prepared(
        self,
        ledger: DurableReceiptLedger,
        transaction_id: str = "transaction-fixture",
        idempotency_key: str = "idempotency-fixture",
    ):
        return ledger.append_prepared(
            transaction_id=transaction_id,
            idempotency_key=idempotency_key,
            scope=self.scope,
            target_ref="target-reference",
            prior_state_ref="prior-state-reference",
            delta_ref="delta-reference",
        )

    def make_resource_manager(
        self,
        *,
        available_bytes: int = 1000,
        safety_margin_bytes: int = 100,
        observed_at_ns: int = 50,
        now_ns: int = 50,
        max_snapshot_age_ns: int = 10,
    ) -> ResourceLeaseManager:
        policy = ResourcePolicy(
            profile_name="fixture-profile",
            resource_domain="fixture-memory-domain",
            safety_margin_bytes=safety_margin_bytes,
            max_snapshot_age_ns=max_snapshot_age_ns,
        )
        provider = MutableSnapshotProvider(
            ResourceSnapshot(
                resource_domain="fixture-memory-domain",
                available_bytes=available_bytes,
                observed_at_ns=observed_at_ns,
            )
        )
        return ResourceLeaseManager(policy, provider, clock=lambda: now_ns)

    @staticmethod
    def make_profile(requested_bytes: int) -> ResourceProfile:
        return ResourceProfile(
            profile_name="fixture-profile",
            resource_domain="fixture-memory-domain",
            requested_bytes=requested_bytes,
        )

    def test_resource_admission_allows_exact_policy_boundary(self) -> None:
        manager = self.make_resource_manager()

        with manager.acquire(self.make_profile(900)) as lease:
            self.assertEqual(lease.reserved_bytes, 900)
            self.assertEqual(manager.reserved_bytes, 900)

        self.assertEqual(manager.reserved_bytes, 0)

    def test_resource_admission_denies_over_budget_without_reserving(self) -> None:
        manager = self.make_resource_manager()

        with self.assertRaises(ResourceAdmissionDenied):
            manager.acquire(self.make_profile(901))

        self.assertEqual(manager.reserved_bytes, 0)

    def test_concurrent_resource_leases_cannot_overcommit_capacity(self) -> None:
        manager = self.make_resource_manager()
        barrier = Barrier(2)

        def acquire_and_hold() -> bool:
            lease = None
            admitted = False
            try:
                lease = manager.acquire(self.make_profile(600))
                admitted = True
            except ResourceAdmissionDenied:
                pass
            barrier.wait()
            if lease is not None:
                lease.release()
            return admitted

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _: acquire_and_hold(), range(2)))

        self.assertEqual(sum(results), 1)
        self.assertEqual(manager.reserved_bytes, 0)

    def test_resource_policy_rejects_stale_and_mismatched_snapshots(self) -> None:
        stale_manager = self.make_resource_manager(observed_at_ns=1, now_ns=50)
        with self.assertRaises(ResourceSnapshotError):
            stale_manager.acquire(self.make_profile(1))

        wrong_domain_policy = ResourcePolicy(
            profile_name="fixture-profile",
            resource_domain="different-domain",
            safety_margin_bytes=0,
            max_snapshot_age_ns=10,
        )
        provider = MutableSnapshotProvider(
            ResourceSnapshot("fixture-memory-domain", 1000, 50)
        )
        wrong_domain_manager = ResourceLeaseManager(
            wrong_domain_policy,
            provider,
            clock=lambda: 50,
        )
        wrong_domain_profile = ResourceProfile(
            profile_name="fixture-profile",
            resource_domain="different-domain",
            requested_bytes=1,
        )
        with self.assertRaises(ResourceSnapshotError):
            wrong_domain_manager.acquire(wrong_domain_profile)

        malformed_snapshot_manager = ResourceLeaseManager(
            wrong_domain_policy,
            lambda: object(),
            clock=lambda: 50,
        )
        with self.assertRaises(ResourceSnapshotError):
            malformed_snapshot_manager.acquire(wrong_domain_profile)

    def test_resource_lease_releases_after_exception_and_is_idempotent(self) -> None:
        manager = self.make_resource_manager()
        lease = manager.acquire(self.make_profile(100))

        with self.assertRaisesRegex(RuntimeError, "worker failure"):
            with lease:
                raise RuntimeError("worker failure")

        lease.release()
        self.assertTrue(lease.released)
        self.assertEqual(manager.reserved_bytes, 0)

    def test_resource_models_reject_invalid_or_unbounded_values(self) -> None:
        with self.assertRaises(ResourceConfigurationError):
            ResourceProfile("fixture-profile", "fixture-memory-domain", True)

        with self.assertRaises(ResourceConfigurationError):
            ResourcePolicy("fixture-profile", "fixture-memory-domain", -1, 1)

    def test_receipt_stream_is_sequential_durable_and_recoverable(self) -> None:
        path = self.root / "receipts.jsonl"
        ledger = DurableReceiptLedger(path, file_mode=RECEIPT_FILE_MODE, clock=lambda: 10)
        prepared = self.append_prepared(ledger)
        completed = ledger.append_terminal(
            prepared.transaction_id,
            ReceiptTransition.OUTCOME,
            "verified-target-reference",
        )

        self.assertEqual([event.sequence for event in ledger.events()], [1, 2])
        self.assertEqual(completed.state, ReceiptTransition.OUTCOME)
        self.assertFalse(completed.unresolved)
        self.assertEqual(path.read_bytes().count(b"\n"), 2)

        recovered = DurableReceiptLedger(path, file_mode=RECEIPT_FILE_MODE, clock=lambda: 11)
        self.assertEqual(recovered.get_transaction(prepared.transaction_id), completed)
        self.assertEqual(recovered.unresolved_transactions(), ())

    def test_ledger_requires_a_valid_injected_file_permission_mode(self) -> None:
        with self.assertRaises(ValueError):
            DurableReceiptLedger(self.root / "boolean-mode.jsonl", file_mode=True)

        with self.assertRaises(ValueError):
            DurableReceiptLedger(self.root / "invalid-mode.jsonl", file_mode=0o1000)

    def test_duplicate_prepare_is_idempotent_and_conflicting_reuse_is_rejected(self) -> None:
        ledger = self.make_ledger()
        first = self.append_prepared(ledger)
        duplicate = self.append_prepared(ledger)

        self.assertEqual(duplicate, first)
        self.assertEqual(len(ledger.events()), 1)
        with self.assertRaises(ReceiptConflictError):
            self.append_prepared(
                ledger,
                transaction_id="different-transaction",
                idempotency_key="idempotency-fixture",
            )
        with self.assertRaises(ReceiptConflictError):
            ledger.append_prepared(
                transaction_id="transaction-fixture",
                idempotency_key="idempotency-fixture",
                scope=self.scope,
                target_ref="different-target",
                prior_state_ref="prior-state-reference",
                delta_ref="delta-reference",
            )

    def test_prepared_receipt_persistence_failure_does_not_record_success(self) -> None:
        path = self.root / "prepared-failure.jsonl"
        ledger = FailingWriteLedger(path, fail_on_write=1)

        with self.assertRaises(LedgerPersistenceError):
            self.append_prepared(ledger)

        recovered = DurableReceiptLedger(path, file_mode=RECEIPT_FILE_MODE, clock=lambda: 11)
        self.assertIsNone(recovered.get_transaction("transaction-fixture"))
        self.assertFalse(path.exists())

    def test_terminal_receipt_failure_recovers_unresolved_prepared_event(self) -> None:
        path = self.root / "terminal-failure.jsonl"
        ledger = FailingWriteLedger(path, fail_on_write=2)
        prepared = self.append_prepared(ledger)

        with self.assertRaises(LedgerPersistenceError):
            ledger.append_terminal(
                prepared.transaction_id,
                ReceiptTransition.OUTCOME,
                "verified-target-reference",
            )

        recovered = DurableReceiptLedger(path, file_mode=RECEIPT_FILE_MODE, clock=lambda: 12)
        unresolved = recovered.unresolved_transactions()
        self.assertEqual(len(unresolved), 1)
        self.assertEqual(unresolved[0].state, ReceiptTransition.PREPARED)

    def test_partial_receipt_write_rolls_back_only_the_uncommitted_tail(self) -> None:
        path = self.root / "partial-write.jsonl"
        ledger = self.make_ledger("partial-write.jsonl")
        original_write = os.write
        write_count = 0

        def partial_then_fail(descriptor: int, content: bytes | memoryview) -> int:
            nonlocal write_count
            write_count += 1
            if write_count == 1:
                return original_write(descriptor, content[: max(1, len(content) // 2)])
            raise OSError("injected partial append failure")

        with patch("src.swarm_core.durable_ledger.os.write", side_effect=partial_then_fail):
            with self.assertRaises(LedgerPersistenceError):
                self.append_prepared(ledger)

        self.assertEqual(path.read_bytes(), b"")
        recovered = self.make_ledger("partial-write.jsonl")
        self.assertIsNone(recovered.get_transaction("transaction-fixture"))

    def test_startup_recovery_exposes_prepared_transaction_for_reconciliation(self) -> None:
        path = self.root / "recovery.jsonl"
        original = DurableReceiptLedger(path, file_mode=RECEIPT_FILE_MODE, clock=lambda: 10)
        prepared = self.append_prepared(original)

        recovered = DurableReceiptLedger(path, file_mode=RECEIPT_FILE_MODE, clock=lambda: 20)

        self.assertEqual(recovered.unresolved_transactions(), (prepared,))
        self.assertTrue(recovered.get_transaction(prepared.transaction_id).unresolved)

    def test_concurrent_receipt_appends_keep_a_contiguous_global_sequence(self) -> None:
        ledger = self.make_ledger("concurrent-receipts.jsonl")

        def append(index: int) -> int:
            transaction = self.append_prepared(
                ledger,
                transaction_id=f"transaction-{index}",
                idempotency_key=f"idempotency-{index}",
            )
            return transaction.events[0].sequence

        with ThreadPoolExecutor(max_workers=8) as executor:
            sequences = list(executor.map(append, range(32)))

        self.assertEqual(sorted(sequences), list(range(1, 33)))
        reopened = DurableReceiptLedger(
            self.root / "concurrent-receipts.jsonl",
            file_mode=RECEIPT_FILE_MODE,
            clock=lambda: 11,
        )
        self.assertEqual([event.sequence for event in reopened.events()], list(range(1, 33)))

    def test_multiple_ledger_instances_share_process_lock_and_replay_state(self) -> None:
        path = self.root / "multi-instance.jsonl"
        ledgers = (
            DurableReceiptLedger(path, file_mode=RECEIPT_FILE_MODE, clock=lambda: 10),
            DurableReceiptLedger(path, file_mode=RECEIPT_FILE_MODE, clock=lambda: 11),
        )

        def append(index: int) -> int:
            return self.append_prepared(
                ledgers[index % len(ledgers)],
                transaction_id=f"instance-transaction-{index}",
                idempotency_key=f"instance-idempotency-{index}",
            ).events[0].sequence

        with ThreadPoolExecutor(max_workers=8) as executor:
            sequences = list(executor.map(append, range(24)))

        self.assertEqual(sorted(sequences), list(range(1, 25)))
        self.assertEqual(
            [event.sequence for event in ledgers[0].events()],
            list(range(1, 25)),
        )

    def test_interleaved_transactions_preserve_append_order(self) -> None:
        ledger = self.make_ledger("interleaved-receipts.jsonl")
        first = self.append_prepared(
            ledger,
            transaction_id="transaction-first",
            idempotency_key="idempotency-first",
        )
        self.append_prepared(
            ledger,
            transaction_id="transaction-second",
            idempotency_key="idempotency-second",
        )
        ledger.append_terminal(
            first.transaction_id,
            ReceiptTransition.OUTCOME,
            "verified-first",
        )

        self.assertEqual(
            [(event.sequence, event.transaction_id) for event in ledger.events()],
            [
                (1, "transaction-first"),
                (2, "transaction-second"),
                (3, "transaction-first"),
            ],
        )

    def test_ledger_detects_tampered_record_and_incomplete_tail(self) -> None:
        path = self.root / "corrupt.jsonl"
        ledger = DurableReceiptLedger(path, file_mode=RECEIPT_FILE_MODE, clock=lambda: 10)
        self.append_prepared(ledger)
        path.write_bytes(path.read_bytes().replace(b"target-reference", b"altered-reference"))

        with self.assertRaises(LedgerCorruptionError):
            DurableReceiptLedger(path, file_mode=RECEIPT_FILE_MODE, clock=lambda: 11)

        incomplete_path = self.root / "incomplete.jsonl"
        incomplete_path.write_bytes(b"{\"partial\":true}")
        with self.assertRaises(LedgerCorruptionError):
            DurableReceiptLedger(
                incomplete_path,
                file_mode=RECEIPT_FILE_MODE,
                clock=lambda: 11,
            )


if __name__ == "__main__":
    unittest.main()