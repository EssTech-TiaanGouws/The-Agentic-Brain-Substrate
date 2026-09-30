from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import Lock, RLock
from time import time_ns
from typing import Callable
from weakref import WeakValueDictionary

from substrate.contracts import ScopeVector


class LedgerError(RuntimeError):
    pass


class LedgerCorruptionError(LedgerError):
    pass


class LedgerPersistenceError(LedgerError):
    pass


class LedgerUnavailableError(LedgerError):
    pass


class ReceiptConflictError(LedgerError):
    pass


class ReceiptTransition(str, Enum):
    PREPARED = "PREPARED"
    OUTCOME = "OUTCOME"
    COMPENSATED = "COMPENSATED"
    INDETERMINATE = "INDETERMINATE"


_TERMINAL_TRANSITIONS = frozenset(
    {
        ReceiptTransition.OUTCOME,
        ReceiptTransition.COMPENSATED,
        ReceiptTransition.INDETERMINATE,
    }
)
_RECORD_FIELDS = frozenset(
    {
        "sequence",
        "transaction_id",
        "idempotency_key",
        "transition",
        "occurred_at_ns",
        "scope",
        "target_ref",
        "prior_state_ref",
        "delta_ref",
        "verification_ref",
        "previous_hash",
        "event_hash",
    }
)


def _validate_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("Receipt fields must be finite JSON-compatible values") from error


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class ReceiptEvent:
    sequence: int
    transaction_id: str
    idempotency_key: str
    transition: ReceiptTransition
    occurred_at_ns: int
    scope: ScopeVector
    target_ref: str
    prior_state_ref: str
    delta_ref: str
    verification_ref: str | None
    previous_hash: str | None
    event_hash: str


@dataclass(frozen=True, slots=True)
class ReceiptTransaction:
    events: tuple[ReceiptEvent, ...]

    @property
    def transaction_id(self) -> str:
        return self.events[0].transaction_id

    @property
    def idempotency_key(self) -> str:
        return self.events[0].idempotency_key

    @property
    def state(self) -> ReceiptTransition:
        return self.events[-1].transition

    @property
    def unresolved(self) -> bool:
        return self.state is ReceiptTransition.PREPARED


class DurableReceiptLedger:
    """Append-only JSONL receipt journal, serialized within the current process."""

    _path_locks: WeakValueDictionary[str, RLock] = WeakValueDictionary()
    _path_locks_guard = Lock()

    def __init__(
        self,
        storage_path: str | os.PathLike[str],
        *,
        file_mode: int,
        clock: Callable[[], int] = time_ns,
    ) -> None:
        if not callable(clock):
            raise ValueError("clock must be callable")
        if isinstance(file_mode, bool) or not isinstance(file_mode, int):
            raise ValueError("file_mode must be an integer permission mode")
        permission_bits = stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO
        if file_mode < 0 or file_mode & ~permission_bits:
            raise ValueError("file_mode contains unsupported permission bits")
        self._path = Path(storage_path)
        if not str(self._path) or not self._path.name:
            raise ValueError("storage_path must identify a file")
        if self._path.is_symlink():
            raise ValueError("storage_path must not be a symbolic link")
        if not self._path.parent.is_dir():
            raise ValueError("storage_path parent directory must already exist")
        self._path = self._path.resolve()
        self._file_mode = file_mode
        self._clock = clock
        self._lock = self._lock_for_path(self._path)
        self._poisoned = False
        self._transactions: dict[str, ReceiptTransaction] = {}
        self._idempotency_index: dict[str, str] = {}
        self._events: list[ReceiptEvent] = []
        self._last_hash: str | None = None
        self._load_existing()

    def append_prepared(
        self,
        *,
        transaction_id: str,
        idempotency_key: str,
        scope: ScopeVector,
        target_ref: str,
        prior_state_ref: str,
        delta_ref: str,
    ) -> ReceiptTransaction:
        transaction, _ = self.prepare_once(
            transaction_id=transaction_id,
            idempotency_key=idempotency_key,
            scope=scope,
            target_ref=target_ref,
            prior_state_ref=prior_state_ref,
            delta_ref=delta_ref,
        )
        return transaction

    def prepare_once(
        self,
        *,
        transaction_id: str,
        idempotency_key: str,
        scope: ScopeVector,
        target_ref: str,
        prior_state_ref: str,
        delta_ref: str,
    ) -> tuple[ReceiptTransaction, bool]:
        transaction_id = _validate_text(transaction_id, "transaction_id")
        idempotency_key = _validate_text(idempotency_key, "idempotency_key")
        self._validate_prepared_context(scope, target_ref, prior_state_ref, delta_ref)

        with self._lock:
            self._ensure_available()
            self._load_existing()
            existing = self._transactions.get(transaction_id)
            if existing is not None:
                prepared = existing.events[0]
                matches = (
                    prepared.idempotency_key == idempotency_key
                    and prepared.scope == scope
                    and prepared.target_ref == target_ref
                    and prepared.prior_state_ref == prior_state_ref
                    and prepared.delta_ref == delta_ref
                )
                if matches:
                    return existing, False
                raise ReceiptConflictError("Transaction ID was reused with different receipt data")

            prior_transaction = self._idempotency_index.get(idempotency_key)
            if prior_transaction is not None:
                raise ReceiptConflictError("Idempotency key is already bound to another transaction")

            event = self._create_event(
                sequence=self._next_sequence(),
                transaction_id=transaction_id,
                idempotency_key=idempotency_key,
                transition=ReceiptTransition.PREPARED,
                scope=scope,
                target_ref=target_ref,
                prior_state_ref=prior_state_ref,
                delta_ref=delta_ref,
                verification_ref=None,
            )
            self._append_event(event)
            transaction = ReceiptTransaction(events=(event,))
            self._transactions[transaction_id] = transaction
            self._idempotency_index[idempotency_key] = transaction_id
            self._events.append(event)
            self._last_hash = event.event_hash
            return transaction, True

    def append_terminal(
        self,
        transaction_id: str,
        transition: ReceiptTransition,
        verification_ref: str,
    ) -> ReceiptTransaction:
        transaction_id = _validate_text(transaction_id, "transaction_id")
        verification_ref = _validate_text(verification_ref, "verification_ref")
        if not isinstance(transition, ReceiptTransition) or transition not in _TERMINAL_TRANSITIONS:
            raise ReceiptConflictError("Terminal transition must be OUTCOME, COMPENSATED, or INDETERMINATE")

        with self._lock:
            self._ensure_available()
            self._load_existing()
            transaction = self._transactions.get(transaction_id)
            if transaction is None:
                raise ReceiptConflictError("Cannot append a terminal receipt without PREPARED")
            current = transaction.events[-1]
            if current.transition is transition and current.verification_ref == verification_ref:
                return transaction
            if current.transition is not ReceiptTransition.PREPARED:
                raise ReceiptConflictError("Transaction already has a different terminal receipt")

            prepared = transaction.events[0]
            event = self._create_event(
                sequence=self._next_sequence(),
                transaction_id=transaction_id,
                idempotency_key=prepared.idempotency_key,
                transition=transition,
                scope=prepared.scope,
                target_ref=prepared.target_ref,
                prior_state_ref=prepared.prior_state_ref,
                delta_ref=prepared.delta_ref,
                verification_ref=verification_ref,
            )
            self._append_event(event)
            updated = ReceiptTransaction(events=transaction.events + (event,))
            self._transactions[transaction_id] = updated
            self._events.append(event)
            self._last_hash = event.event_hash
            return updated

    def get_transaction(self, transaction_id: str) -> ReceiptTransaction | None:
        with self._lock:
            self._ensure_available()
            self._load_existing()
            return self._transactions.get(transaction_id)

    def unresolved_transactions(self) -> tuple[ReceiptTransaction, ...]:
        with self._lock:
            self._ensure_available()
            self._load_existing()
            return tuple(
                transaction
                for transaction in self._transactions.values()
                if transaction.unresolved
            )

    def events(self) -> tuple[ReceiptEvent, ...]:
        with self._lock:
            self._ensure_available()
            self._load_existing()
            return tuple(self._events)

    @classmethod
    def _lock_for_path(cls, path: Path) -> RLock:
        key = os.fspath(path)
        with cls._path_locks_guard:
            lock = cls._path_locks.get(key)
            if lock is None:
                lock = RLock()
                cls._path_locks[key] = lock
            return lock

    def _validate_prepared_context(
        self,
        scope: ScopeVector,
        target_ref: str,
        prior_state_ref: str,
        delta_ref: str,
    ) -> None:
        if not isinstance(scope, ScopeVector) or not scope.is_complete():
            raise ValueError("A complete four-field scope vector is required")
        _validate_text(target_ref, "target_ref")
        _validate_text(prior_state_ref, "prior_state_ref")
        _validate_text(delta_ref, "delta_ref")

    def _create_event(
        self,
        *,
        sequence: int,
        transaction_id: str,
        idempotency_key: str,
        transition: ReceiptTransition,
        scope: ScopeVector,
        target_ref: str,
        prior_state_ref: str,
        delta_ref: str,
        verification_ref: str | None,
    ) -> ReceiptEvent:
        occurred_at_ns = self._clock()
        if isinstance(occurred_at_ns, bool) or not isinstance(occurred_at_ns, int):
            raise ValueError("clock must return an integer nanosecond timestamp")
        if occurred_at_ns < 0 or occurred_at_ns > sys.maxsize:
            raise ValueError("clock timestamp is outside the supported range")
        body = {
            "sequence": sequence,
            "transaction_id": transaction_id,
            "idempotency_key": idempotency_key,
            "transition": transition.value,
            "occurred_at_ns": occurred_at_ns,
            "scope": {
                "tenant_id": scope.tenant_id,
                "user_id": scope.user_id,
                "project_id": scope.project_id,
                "workspace_id": scope.workspace_id,
            },
            "target_ref": target_ref,
            "prior_state_ref": prior_state_ref,
            "delta_ref": delta_ref,
            "verification_ref": verification_ref,
            "previous_hash": self._last_hash,
        }
        event_hash = hashlib.sha256(_canonical_json(body)).hexdigest()
        return ReceiptEvent(
            sequence=sequence,
            transaction_id=transaction_id,
            idempotency_key=idempotency_key,
            transition=transition,
            occurred_at_ns=occurred_at_ns,
            scope=scope,
            target_ref=target_ref,
            prior_state_ref=prior_state_ref,
            delta_ref=delta_ref,
            verification_ref=verification_ref,
            previous_hash=self._last_hash,
            event_hash=event_hash,
        )

    def _append_event(self, event: ReceiptEvent) -> None:
        line = _canonical_json(self._event_record(event)) + b"\n"
        try:
            self._persist_line(line)
        except OSError as error:
            self._poisoned = True
            raise LedgerPersistenceError("Receipt event could not be durably appended") from error

    def _persist_line(self, line: bytes) -> None:
        path_preexisted = self._path.exists()
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self._path, flags, self._file_mode)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise OSError("Receipt ledger must be a regular file")
            previous_size = os.fstat(descriptor).st_size
            try:
                view = memoryview(line)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("Receipt ledger append made no progress")
                    view = view[written:]
                os.fsync(descriptor)
                if not path_preexisted:
                    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                    directory_descriptor = os.open(self._path.parent, directory_flags)
                    try:
                        os.fsync(directory_descriptor)
                    finally:
                        os.close(directory_descriptor)
            except OSError as append_error:
                try:
                    os.ftruncate(descriptor, previous_size)
                    os.fsync(descriptor)
                except OSError as rollback_error:
                    raise OSError("Receipt append failed and its uncommitted tail could not be removed") from rollback_error
                raise append_error
        finally:
            os.close(descriptor)

    @staticmethod
    def _event_record(event: ReceiptEvent) -> dict[str, object]:
        body: dict[str, object] = {
            "sequence": event.sequence,
            "transaction_id": event.transaction_id,
            "idempotency_key": event.idempotency_key,
            "transition": event.transition.value,
            "occurred_at_ns": event.occurred_at_ns,
            "scope": {
                "tenant_id": event.scope.tenant_id,
                "user_id": event.scope.user_id,
                "project_id": event.scope.project_id,
                "workspace_id": event.scope.workspace_id,
            },
            "target_ref": event.target_ref,
            "prior_state_ref": event.prior_state_ref,
            "delta_ref": event.delta_ref,
            "verification_ref": event.verification_ref,
            "previous_hash": event.previous_hash,
        }
        body["event_hash"] = event.event_hash
        return body

    def _load_existing(self) -> None:
        self._transactions.clear()
        self._idempotency_index.clear()
        self._events.clear()
        self._last_hash = None
        if not self._path.exists():
            return
        if self._path.is_symlink() or not self._path.is_file():
            raise LedgerCorruptionError("Receipt ledger path must be a regular non-symlink file")
        data = self._path.read_bytes()
        if data and not data.endswith(b"\n"):
            raise LedgerCorruptionError("Receipt ledger ends with an incomplete record")

        previous_hash: str | None = None
        for expected_sequence, line in enumerate(data.splitlines(), start=1):
            try:
                record = json.loads(line, object_pairs_hook=_unique_object)
                event = self._decode_event(record, expected_sequence, previous_hash)
                self._index_loaded_event(event)
            except LedgerError:
                raise
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise LedgerCorruptionError(
                    f"Receipt ledger record {expected_sequence} is invalid"
                ) from error
            previous_hash = event.event_hash
        self._last_hash = previous_hash

    def _decode_event(
        self,
        record: object,
        expected_sequence: int,
        expected_previous_hash: str | None,
    ) -> ReceiptEvent:
        if not isinstance(record, dict) or set(record) != _RECORD_FIELDS:
            raise LedgerCorruptionError("Receipt ledger record has an unexpected shape")
        body = dict(record)
        claimed_hash = body.pop("event_hash")
        actual_hash = hashlib.sha256(_canonical_json(body)).hexdigest()
        if claimed_hash != actual_hash:
            raise LedgerCorruptionError("Receipt ledger hash verification failed")
        sequence = record["sequence"]
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence != expected_sequence:
            raise LedgerCorruptionError("Receipt ledger sequence is not contiguous")
        if record["previous_hash"] != expected_previous_hash:
            raise LedgerCorruptionError("Receipt ledger hash chain is broken")

        scope_data = record["scope"]
        if not isinstance(scope_data, dict) or set(scope_data) != {
            "tenant_id",
            "user_id",
            "project_id",
            "workspace_id",
        }:
            raise LedgerCorruptionError("Receipt scope vector is malformed")
        scope = ScopeVector(**scope_data)
        if not scope.is_complete():
            raise LedgerCorruptionError("Receipt scope vector is incomplete")
        occurred_at_ns = record["occurred_at_ns"]
        if (
            isinstance(occurred_at_ns, bool)
            or not isinstance(occurred_at_ns, int)
            or occurred_at_ns < 0
            or occurred_at_ns > sys.maxsize
        ):
            raise LedgerCorruptionError("Receipt timestamp is invalid")
        transition = ReceiptTransition(record["transition"])
        transaction_id = _validate_text(record["transaction_id"], "transaction_id")
        idempotency_key = _validate_text(record["idempotency_key"], "idempotency_key")
        target_ref = _validate_text(record["target_ref"], "target_ref")
        prior_state_ref = _validate_text(record["prior_state_ref"], "prior_state_ref")
        delta_ref = _validate_text(record["delta_ref"], "delta_ref")
        verification_ref = record["verification_ref"]
        if verification_ref is not None:
            verification_ref = _validate_text(verification_ref, "verification_ref")
        if transition is ReceiptTransition.PREPARED and verification_ref is not None:
            raise LedgerCorruptionError("PREPARED receipt cannot contain terminal verification")
        if transition in _TERMINAL_TRANSITIONS and verification_ref is None:
            raise LedgerCorruptionError("Terminal receipt requires a verification reference")

        return ReceiptEvent(
            sequence=expected_sequence,
            transaction_id=transaction_id,
            idempotency_key=idempotency_key,
            transition=transition,
            occurred_at_ns=occurred_at_ns,
            scope=scope,
            target_ref=target_ref,
            prior_state_ref=prior_state_ref,
            delta_ref=delta_ref,
            verification_ref=verification_ref,
            previous_hash=expected_previous_hash,
            event_hash=claimed_hash,
        )

    def _index_loaded_event(self, event: ReceiptEvent) -> None:
        transaction = self._transactions.get(event.transaction_id)
        if event.transition is ReceiptTransition.PREPARED:
            if transaction is not None or event.idempotency_key in self._idempotency_index:
                raise LedgerCorruptionError("Receipt ledger contains duplicate transaction identity")
            self._transactions[event.transaction_id] = ReceiptTransaction(events=(event,))
            self._idempotency_index[event.idempotency_key] = event.transaction_id
            self._events.append(event)
            return

        if transaction is None or transaction.state is not ReceiptTransition.PREPARED:
            raise LedgerCorruptionError("Terminal receipt does not follow PREPARED")
        prepared = transaction.events[0]
        if (
            event.idempotency_key != prepared.idempotency_key
            or event.scope != prepared.scope
            or event.target_ref != prepared.target_ref
            or event.prior_state_ref != prepared.prior_state_ref
            or event.delta_ref != prepared.delta_ref
        ):
            raise LedgerCorruptionError("Terminal receipt context differs from PREPARED")
        self._transactions[event.transaction_id] = ReceiptTransaction(
            events=transaction.events + (event,)
        )
        self._events.append(event)

    def _next_sequence(self) -> int:
        return len(self._events) + 1

    def _ensure_available(self) -> None:
        if self._poisoned:
            raise LedgerUnavailableError("Ledger must be reopened and recovered after a write failure")