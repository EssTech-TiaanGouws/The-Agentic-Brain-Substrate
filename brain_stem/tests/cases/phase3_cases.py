from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

from src.swarm_core.durable_ledger import (
    DurableReceiptLedger,
    LedgerPersistenceError,
    ReceiptTransition,
)
from src.swarm_core.lease_manager import (
    ResourceAdmissionDenied,
    ResourceLeaseManager,
    ResourcePolicy,
    ResourceProfile,
    ResourceSnapshot,
)
from src.swarm_core.transaction_coordinator import (
    CommitOutcome,
    CompensationOutcome,
    IndeterminateTransactionError,
    StagedOperation,
    TransactionAborted,
    TransactionCoordinator,
    TransactionRequest,
    UnresolvedTransactionError,
)
from substrate.contracts import ScopeVector


RECEIPT_FILE_MODE = 0o600


class FailingWriteLedger(DurableReceiptLedger):
    def __init__(self, path: Path, fail_on_write: int) -> None:
        self._write_count = 0
        self._fail_on_write = fail_on_write
        super().__init__(path, file_mode=RECEIPT_FILE_MODE, clock=lambda: 1)

    def _persist_line(self, line: bytes) -> None:
        self._write_count += 1
        if self._write_count == self._fail_on_write:
            raise OSError("injected receipt persistence failure")
        super()._persist_line(line)


class FakeWorker:
    def __init__(self, events: list[str] | None = None, failure: Exception | None = None) -> None:
        self.events = events if events is not None else []
        self.failure = failure
        self.calls = 0

    def stage(self, request: TransactionRequest) -> StagedOperation:
        self.calls += 1
        self.events.append("stage")
        if self.failure is not None:
            raise self.failure
        return StagedOperation(stage_ref=f"stage:{request.transaction_id}")


class BarrierWorker(FakeWorker):
    def __init__(self, event: Event, release: Event) -> None:
        super().__init__()
        self.entered = event
        self.release = release

    def stage(self, request: TransactionRequest) -> StagedOperation:
        self.calls += 1
        self.entered.set()
        if not self.release.wait(timeout=3):
            raise TimeoutError("concurrent worker release timed out")
        return StagedOperation(stage_ref=f"stage:{request.transaction_id}")


class FakeVerifier:
    def __init__(self, events: list[str] | None = None, fail_at: str | None = None) -> None:
        self.events = events if events is not None else []
        self.fail_at = fail_at

    def verify_adversarial(self, request: TransactionRequest, stage: StagedOperation) -> str:
        self.events.append("block-6")
        if self.fail_at == "block-6":
            raise RuntimeError("adversarial stage rejection")
        return "block-6-verification"

    def verify_epistemic(self, request: TransactionRequest, stage: StagedOperation) -> str:
        self.events.append("block-8")
        if self.fail_at == "block-8":
            raise RuntimeError("epistemic stage rejection")
        return "block-8-verification"


class FakeMutationAuthority:
    def __init__(
        self,
        events: list[str] | None = None,
        *,
        commit_failure: Exception | None = None,
        compensation_verified: bool = True,
        compensation_failure: Exception | None = None,
    ) -> None:
        self.events = events if events is not None else []
        self.commit_failure = commit_failure
        self.compensation_verified = compensation_verified
        self.compensation_failure = compensation_failure
        self.authorize_calls = 0
        self.commit_calls = 0
        self.compensation_calls = 0

    def authorize(self, request: TransactionRequest) -> None:
        self.authorize_calls += 1
        self.events.append("authorize")

    def commit(self, request: TransactionRequest, stage: StagedOperation) -> CommitOutcome:
        self.commit_calls += 1
        self.events.append("commit")
        if self.commit_failure is not None:
            raise self.commit_failure
        return CommitOutcome(verification_ref=f"commit:{request.transaction_id}")

    def compensate(
        self,
        request: TransactionRequest,
        stage: StagedOperation | None,
    ) -> CompensationOutcome:
        self.compensation_calls += 1
        self.events.append("compensate")
        if self.compensation_failure is not None:
            raise self.compensation_failure
        return CompensationOutcome(
            verified=self.compensation_verified,
            verification_ref=f"compensation:{request.transaction_id}",
        )


class Phase3CoordinatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.scope = ScopeVector("tenant-test", "user-test", "project-test", "workspace-test")

    def make_ledger(self, name: str = "receipts.jsonl") -> DurableReceiptLedger:
        return DurableReceiptLedger(
            self.root / name,
            file_mode=RECEIPT_FILE_MODE,
            clock=lambda: 20,
        )

    @staticmethod
    def make_request(transaction_id: str = "transaction-test") -> TransactionRequest:
        return TransactionRequest(
            transaction_id=transaction_id,
            idempotency_key=f"idempotency:{transaction_id}",
            scope=ScopeVector("tenant-test", "user-test", "project-test", "workspace-test"),
            target_ref=f"target:{transaction_id}",
            prior_state_ref=f"prior:{transaction_id}",
            delta_ref=f"delta:{transaction_id}",
            resource_profile=ResourceProfile("test-profile", "host-memory-test", 100),
        )

    @staticmethod
    def make_lease_manager(
        available_bytes: int = 1000,
        safety_margin_bytes: int = 0,
    ) -> ResourceLeaseManager:
        policy = ResourcePolicy(
            profile_name="test-profile",
            resource_domain="host-memory-test",
            safety_margin_bytes=safety_margin_bytes,
            max_snapshot_age_ns=20,
        )
        snapshot = ResourceSnapshot("host-memory-test", available_bytes, 100)
        return ResourceLeaseManager(policy, lambda: snapshot, clock=lambda: 100)

    def make_coordinator(
        self,
        *,
        ledger: DurableReceiptLedger | None = None,
        worker: FakeWorker | None = None,
        verifier: FakeVerifier | None = None,
        authority: FakeMutationAuthority | None = None,
        lease_manager: ResourceLeaseManager | None = None,
    ) -> tuple[TransactionCoordinator, DurableReceiptLedger, FakeWorker, FakeVerifier, FakeMutationAuthority, ResourceLeaseManager]:
        selected_ledger = ledger or self.make_ledger()
        selected_worker = worker or FakeWorker()
        selected_verifier = verifier or FakeVerifier()
        selected_authority = authority or FakeMutationAuthority()
        selected_leases = lease_manager or self.make_lease_manager()
        coordinator = TransactionCoordinator(
            lease_manager=selected_leases,
            ledger=selected_ledger,
            worker=selected_worker,
            verifier=selected_verifier,
            mutation_authority=selected_authority,
        )
        return (
            coordinator,
            selected_ledger,
            selected_worker,
            selected_verifier,
            selected_authority,
            selected_leases,
        )

    def test_success_follows_authorize_stage_critic_audit_commit_outcome(self) -> None:
        events: list[str] = []
        coordinator, ledger, _, _, authority, leases = self.make_coordinator(
            ledger=self.make_ledger(),
            worker=FakeWorker(events),
            verifier=FakeVerifier(events),
            authority=FakeMutationAuthority(events),
        )

        result = coordinator.execute(self.make_request())

        self.assertEqual(events, ["authorize", "stage", "block-6", "block-8", "commit"])
        self.assertTrue(result.succeeded)
        self.assertEqual(ledger.get_transaction(result.transaction_id).state, ReceiptTransition.OUTCOME)
        self.assertEqual(authority.compensation_calls, 0)
        self.assertEqual(leases.reserved_bytes, 0)

    def test_prepared_persistence_failure_denies_before_stage_or_mutation(self) -> None:
        ledger = FailingWriteLedger(self.root / "prepared-failure.jsonl", fail_on_write=1)
        coordinator, _, worker, _, authority, leases = self.make_coordinator(ledger=ledger)

        with self.assertRaises(LedgerPersistenceError):
            coordinator.execute(self.make_request())

        self.assertEqual(worker.calls, 0)
        self.assertEqual(authority.commit_calls, 0)
        self.assertEqual(authority.compensation_calls, 0)
        self.assertEqual(leases.reserved_bytes, 0)

    def test_worker_failure_is_compensated_and_never_returns_success(self) -> None:
        coordinator, ledger, _, _, authority, _ = self.make_coordinator(
            worker=FakeWorker(failure=RuntimeError("stage failed"))
        )

        with self.assertRaises(TransactionAborted):
            coordinator.execute(self.make_request())

        transaction = ledger.get_transaction("transaction-test")
        self.assertEqual(transaction.state, ReceiptTransition.COMPENSATED)
        self.assertEqual(authority.compensation_calls, 1)
        self.assertEqual(authority.commit_calls, 0)

    def test_block_6_rejection_prevents_block_8_and_commit_then_compensates(self) -> None:
        events: list[str] = []
        coordinator, ledger, _, _, authority, _ = self.make_coordinator(
            worker=FakeWorker(events),
            verifier=FakeVerifier(events, fail_at="block-6"),
            authority=FakeMutationAuthority(events),
        )

        with self.assertRaises(TransactionAborted):
            coordinator.execute(self.make_request())

        self.assertEqual(events, ["authorize", "stage", "block-6", "compensate"])
        self.assertEqual(ledger.get_transaction("transaction-test").state, ReceiptTransition.COMPENSATED)
        self.assertEqual(authority.commit_calls, 0)

    def test_commit_failure_with_verified_compensation_is_aborted(self) -> None:
        coordinator, ledger, _, _, authority, _ = self.make_coordinator(
            authority=FakeMutationAuthority(commit_failure=OSError("commit failed"))
        )

        with self.assertRaises(TransactionAborted):
            coordinator.execute(self.make_request())

        self.assertEqual(ledger.get_transaction("transaction-test").state, ReceiptTransition.COMPENSATED)
        self.assertEqual(authority.compensation_calls, 1)

    def test_failed_compensation_raises_indeterminate_and_persists_fault_state(self) -> None:
        coordinator, ledger, _, _, authority, _ = self.make_coordinator(
            worker=FakeWorker(failure=RuntimeError("stage failed")),
            authority=FakeMutationAuthority(compensation_verified=False),
        )

        with self.assertRaises(IndeterminateTransactionError) as captured:
            coordinator.execute(self.make_request())

        self.assertTrue(captured.exception.receipt_persisted)
        self.assertEqual(ledger.get_transaction("transaction-test").state, ReceiptTransition.INDETERMINATE)
        self.assertEqual(authority.compensation_calls, 1)

    def test_compensation_exception_is_strictly_indeterminate(self) -> None:
        coordinator, ledger, _, _, _, _ = self.make_coordinator(
            worker=FakeWorker(failure=RuntimeError("stage failed")),
            authority=FakeMutationAuthority(compensation_failure=OSError("rollback failed")),
        )

        with self.assertRaises(IndeterminateTransactionError):
            coordinator.execute(self.make_request())

        self.assertEqual(ledger.get_transaction("transaction-test").state, ReceiptTransition.INDETERMINATE)

    def test_outcome_receipt_failure_is_unresolved_and_not_compensated(self) -> None:
        path = self.root / "outcome-failure.jsonl"
        ledger = FailingWriteLedger(path, fail_on_write=2)
        coordinator, _, worker, _, authority, leases = self.make_coordinator(ledger=ledger)

        with self.assertRaises(UnresolvedTransactionError):
            coordinator.execute(self.make_request())

        self.assertEqual(worker.calls, 1)
        self.assertEqual(authority.commit_calls, 1)
        self.assertEqual(authority.compensation_calls, 0)
        self.assertEqual(leases.reserved_bytes, 0)
        recovered = DurableReceiptLedger(path, file_mode=RECEIPT_FILE_MODE, clock=lambda: 21)
        self.assertEqual(recovered.unresolved_transactions()[0].state, ReceiptTransition.PREPARED)

    def test_completed_duplicate_replays_without_reexecuting_worker(self) -> None:
        coordinator, _, worker, _, authority, _ = self.make_coordinator()
        request = self.make_request()

        first = coordinator.execute(request)
        replay = coordinator.execute(request)

        self.assertTrue(first.succeeded)
        self.assertTrue(replay.succeeded)
        self.assertTrue(replay.replayed)
        self.assertEqual(worker.calls, 1)
        self.assertEqual(authority.commit_calls, 1)

    def test_unresolved_prepared_transaction_blocks_replay(self) -> None:
        ledger = self.make_ledger("unresolved.jsonl")
        request = self.make_request()
        ledger.append_prepared(
            transaction_id=request.transaction_id,
            idempotency_key=request.idempotency_key,
            scope=request.scope,
            target_ref=request.target_ref,
            prior_state_ref=request.prior_state_ref,
            delta_ref=request.delta_ref,
        )
        coordinator, _, worker, _, authority, _ = self.make_coordinator(ledger=ledger)

        with self.assertRaises(UnresolvedTransactionError):
            coordinator.execute(request)

        self.assertEqual(worker.calls, 0)
        self.assertEqual(authority.commit_calls, 0)

    def test_concurrent_distinct_transactions_share_leases_and_commit(self) -> None:
        entered = Event()
        release = Event()
        worker = BarrierWorker(entered, release)
        lease_manager = self.make_lease_manager(available_bytes=200)
        coordinator, ledger, _, _, _, _ = self.make_coordinator(
            ledger=self.make_ledger("cross-session.jsonl"),
            worker=worker,
            lease_manager=lease_manager,
        )

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(coordinator.execute, self.make_request("transaction-a"))
            second = executor.submit(coordinator.execute, self.make_request("transaction-b"))
            self.assertTrue(entered.wait(timeout=3))
            release.set()
            results = (first.result(timeout=3), second.result(timeout=3))

        self.assertTrue(all(result.succeeded for result in results))
        self.assertEqual(worker.calls, 2)
        self.assertEqual(len(ledger.unresolved_transactions()), 0)
        self.assertEqual(lease_manager.reserved_bytes, 0)

    def test_concurrent_duplicate_transaction_has_one_stage_execution(self) -> None:
        entered = Event()
        release = Event()
        worker = BarrierWorker(entered, release)
        coordinator, _, _, _, _, _ = self.make_coordinator(worker=worker)
        request = self.make_request()

        with ThreadPoolExecutor(max_workers=1) as executor:
            first = executor.submit(coordinator.execute, request)
            self.assertTrue(entered.wait(timeout=3))
            with self.assertRaises(UnresolvedTransactionError):
                coordinator.execute(request)
            release.set()
            self.assertTrue(first.result(timeout=3).succeeded)

        self.assertEqual(worker.calls, 1)

    def test_resource_admission_denial_happens_before_prepared(self) -> None:
        coordinator, ledger, worker, _, authority, _ = self.make_coordinator(
            lease_manager=self.make_lease_manager(available_bytes=10)
        )

        with self.assertRaises(ResourceAdmissionDenied):
            coordinator.execute(self.make_request())

        self.assertEqual(ledger.events(), ())
        self.assertEqual(worker.calls, 0)
        self.assertEqual(authority.commit_calls, 0)


if __name__ == "__main__":
    unittest.main()