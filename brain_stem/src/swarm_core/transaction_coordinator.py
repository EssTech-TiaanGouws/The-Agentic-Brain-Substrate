from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from substrate.contracts import ScopeVector

from .durable_ledger import (
    DurableReceiptLedger,
    LedgerError,
    ReceiptConflictError,
    ReceiptTransaction,
    ReceiptTransition,
)
from .lease_manager import ResourceLeaseManager, ResourceProfile


class TransactionCoordinatorError(RuntimeError):
    pass


class TransactionAborted(TransactionCoordinatorError):
    pass


class UnresolvedTransactionError(TransactionCoordinatorError):
    pass


class IndeterminateTransactionError(TransactionCoordinatorError):
    def __init__(self, transaction_id: str, message: str, receipt_persisted: bool) -> None:
        super().__init__(message)
        self.transaction_id = transaction_id
        self.receipt_persisted = receipt_persisted


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class TransactionRequest:
    transaction_id: str
    idempotency_key: str
    scope: ScopeVector
    target_ref: str
    prior_state_ref: str
    delta_ref: str
    resource_profile: ResourceProfile

    def __post_init__(self) -> None:
        for name in (
            "transaction_id",
            "idempotency_key",
            "target_ref",
            "prior_state_ref",
            "delta_ref",
        ):
            _required_text(getattr(self, name), name)
        if not isinstance(self.scope, ScopeVector) or not self.scope.is_complete():
            raise ValueError("Transaction requires a complete four-field scope vector")
        if not isinstance(self.resource_profile, ResourceProfile):
            raise ValueError("Transaction requires an injected resource profile")


@dataclass(frozen=True, slots=True)
class StagedOperation:
    stage_ref: str

    def __post_init__(self) -> None:
        _required_text(self.stage_ref, "stage_ref")


@dataclass(frozen=True, slots=True)
class CommitOutcome:
    verification_ref: str

    def __post_init__(self) -> None:
        _required_text(self.verification_ref, "verification_ref")


@dataclass(frozen=True, slots=True)
class CompensationOutcome:
    verified: bool
    verification_ref: str

    def __post_init__(self) -> None:
        if not isinstance(self.verified, bool):
            raise ValueError("verified must be a boolean")
        _required_text(self.verification_ref, "verification_ref")


@dataclass(frozen=True, slots=True)
class TransactionResult:
    transaction_id: str
    state: ReceiptTransition
    verification_ref: str
    replayed: bool = False

    @property
    def succeeded(self) -> bool:
        return self.state is ReceiptTransition.OUTCOME


class TransactionWorker(Protocol):
    def stage(self, request: TransactionRequest) -> StagedOperation: ...


class OrderedStageVerifier(Protocol):
    def verify_adversarial(
        self,
        request: TransactionRequest,
        stage: StagedOperation,
    ) -> str: ...

    def verify_epistemic(
        self,
        request: TransactionRequest,
        stage: StagedOperation,
    ) -> str: ...


class MutationAuthority(Protocol):
    def authorize(self, request: TransactionRequest) -> None: ...

    def commit(
        self,
        request: TransactionRequest,
        stage: StagedOperation,
    ) -> CommitOutcome: ...

    def compensate(
        self,
        request: TransactionRequest,
        stage: StagedOperation | None,
    ) -> CompensationOutcome: ...


class TransactionCoordinator:
    """Bounded coordinator; storage, resources, workers, and mutations are injected."""

    def __init__(
        self,
        *,
        lease_manager: ResourceLeaseManager,
        ledger: DurableReceiptLedger,
        worker: TransactionWorker,
        verifier: OrderedStageVerifier,
        mutation_authority: MutationAuthority,
    ) -> None:
        self._lease_manager = lease_manager
        self._ledger = ledger
        self._worker = worker
        self._verifier = verifier
        self._mutation_authority = mutation_authority

    def execute(self, request: TransactionRequest) -> TransactionResult:
        if not isinstance(request, TransactionRequest):
            raise TypeError("request must be a TransactionRequest")
        self._mutation_authority.authorize(request)

        with self._lease_manager.acquire(request.resource_profile):
            transaction, created = self._ledger.prepare_once(
                transaction_id=request.transaction_id,
                idempotency_key=request.idempotency_key,
                scope=request.scope,
                target_ref=request.target_ref,
                prior_state_ref=request.prior_state_ref,
                delta_ref=request.delta_ref,
            )
            if not created:
                return self._resolve_existing(transaction)

            stage: StagedOperation | None = None
            try:
                stage = self._worker.stage(request)
                if not isinstance(stage, StagedOperation):
                    raise TransactionCoordinatorError("Worker returned an invalid staged operation")

                adversarial_ref = self._verifier.verify_adversarial(request, stage)
                _required_text(adversarial_ref, "adversarial_verification_ref")
                epistemic_ref = self._verifier.verify_epistemic(request, stage)
                _required_text(epistemic_ref, "epistemic_verification_ref")

                committed = self._mutation_authority.commit(request, stage)
                if not isinstance(committed, CommitOutcome):
                    raise TransactionCoordinatorError("Mutation authority returned an invalid commit outcome")
            except Exception as primary_error:
                self._compensate_or_raise(request, stage, primary_error)

            try:
                terminal = self._ledger.append_terminal(
                    request.transaction_id,
                    ReceiptTransition.OUTCOME,
                    committed.verification_ref,
                )
            except Exception as receipt_error:
                raise UnresolvedTransactionError(
                    "Commit was attempted but its terminal receipt is uncertain; recovery is required"
                ) from receipt_error

            return TransactionResult(
                transaction_id=request.transaction_id,
                state=terminal.state,
                verification_ref=committed.verification_ref,
            )

    @staticmethod
    def _resolve_existing(transaction: ReceiptTransaction) -> TransactionResult:
        latest = transaction.events[-1]
        if latest.transition is ReceiptTransition.OUTCOME:
            return TransactionResult(
                transaction_id=transaction.transaction_id,
                state=latest.transition,
                verification_ref=latest.verification_ref or "",
                replayed=True,
            )
        if latest.transition is ReceiptTransition.INDETERMINATE:
            raise IndeterminateTransactionError(
                transaction.transaction_id,
                "Transaction remains INDETERMINATE and requires reconciliation",
                receipt_persisted=True,
            )
        if latest.transition is ReceiptTransition.COMPENSATED:
            raise TransactionAborted("Transaction was already compensated")
        raise UnresolvedTransactionError(
            "Transaction has an unresolved PREPARED receipt and requires recovery"
        )

    def _compensate_or_raise(
        self,
        request: TransactionRequest,
        stage: StagedOperation | None,
        primary_error: Exception,
    ) -> None:
        try:
            compensation = self._mutation_authority.compensate(request, stage)
            if not isinstance(compensation, CompensationOutcome):
                raise TransactionCoordinatorError("Mutation authority returned an invalid compensation outcome")
        except Exception as compensation_error:
            compensation = CompensationOutcome(
                verified=False,
                verification_ref=f"compensation-error:{type(compensation_error).__name__}",
            )

        if not compensation.verified:
            receipt_persisted = False
            try:
                self._ledger.append_terminal(
                    request.transaction_id,
                    ReceiptTransition.INDETERMINATE,
                    compensation.verification_ref,
                )
                receipt_persisted = True
            except LedgerError:
                pass
            raise IndeterminateTransactionError(
                request.transaction_id,
                "Compensation could not verify restoration; transaction is INDETERMINATE",
                receipt_persisted=receipt_persisted,
            ) from primary_error

        try:
            self._ledger.append_terminal(
                request.transaction_id,
                ReceiptTransition.COMPENSATED,
                compensation.verification_ref,
            )
        except LedgerError as receipt_error:
            raise UnresolvedTransactionError(
                "Compensation verified but its terminal receipt could not be persisted"
            ) from receipt_error
        raise TransactionAborted("Transaction failed and compensation was verified") from primary_error