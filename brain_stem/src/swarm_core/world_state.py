from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from time import time_ns
from typing import Callable, Protocol

from substrate.contracts import ScopeVector


class WorldStateError(RuntimeError):
    pass


class WorldStateCorruptionError(WorldStateError):
    pass


class WorldStatePersistenceError(WorldStateError):
    pass


class WorldStateConflictError(WorldStateError):
    pass


class WorldStateAdmissionDenied(WorldStateError):
    pass


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("World-state values must be finite JSON-compatible values") from error


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def _scope_payload(scope: ScopeVector) -> dict[str, str]:
    return {
        "tenant_id": scope.tenant_id,
        "user_id": scope.user_id,
        "project_id": scope.project_id,
        "workspace_id": scope.workspace_id,
    }


@dataclass(frozen=True, slots=True)
class WorldStateAssertion:
    assertion_id: str
    fact_key: str
    scope: ScopeVector
    subject: str
    predicate: str
    value: object
    source_ref: str
    source_sha256: str
    transaction_id: str

    def __post_init__(self) -> None:
        for field in (
            "assertion_id",
            "fact_key",
            "subject",
            "predicate",
            "source_ref",
            "transaction_id",
        ):
            _text(getattr(self, field), field)
        if not isinstance(self.scope, ScopeVector) or not self.scope.is_complete():
            raise ValueError("A complete four-field scope is required")
        if (
            not isinstance(self.source_sha256, str)
            or len(self.source_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.source_sha256)
        ):
            raise ValueError("source_sha256 must be a lowercase SHA-256 digest")
        _canonical_json(self.value)


@dataclass(frozen=True, slots=True)
class WorldStateAdmissionDecision:
    approved: bool
    verification_reference: str
    superseded_assertion_ids: tuple[str, ...] = ()
    resolution_reference: str | None = None


class WorldStateAdmissionVerifier(Protocol):
    def review(
        self,
        assertion: WorldStateAssertion,
        active_conflicts: tuple[WorldStateRecord, ...],
    ) -> WorldStateAdmissionDecision: ...


@dataclass(frozen=True, slots=True)
class WorldStateRecord:
    sequence: int
    assertion: WorldStateAssertion
    occurred_at_ns: int
    verification_reference: str
    resolution_reference: str | None
    superseded_assertion_ids: tuple[str, ...]
    previous_hash: str | None
    event_hash: str


class WorldStateLedger:
    """SQLite-backed append-only semantic ledger, separate from action receipts."""

    _PAYLOAD_FIELDS = frozenset(
        {
            "assertion",
            "occurred_at_ns",
            "verification_reference",
            "resolution_reference",
            "superseded_assertion_ids",
        }
    )
    _ASSERTION_FIELDS = frozenset(
        {
            "assertion_id",
            "fact_key",
            "scope",
            "subject",
            "predicate",
            "value",
            "source_ref",
            "source_sha256",
            "transaction_id",
        }
    )

    def __init__(
        self,
        storage_path: str | os.PathLike[str],
        *,
        admission_verifier: WorldStateAdmissionVerifier,
        file_mode: int = 0o600,
        clock: Callable[[], int] = time_ns,
    ) -> None:
        if not callable(getattr(admission_verifier, "review", None)):
            raise ValueError("admission_verifier must implement review()")
        if not callable(clock):
            raise ValueError("clock must be callable")
        if isinstance(file_mode, bool) or not isinstance(file_mode, int) or file_mode < 0 or file_mode & ~0o777:
            raise ValueError("file_mode must contain only permission bits")
        path = Path(storage_path)
        if not path.name or not path.parent.is_dir() or path.is_symlink():
            raise ValueError("storage_path must be a non-symlink file in an existing directory")
        self._path = path.resolve()
        self._clock = clock
        self._admission_verifier = admission_verifier
        self._lock = RLock()
        self._create_file(file_mode)
        os.chmod(self._path, file_mode, follow_symlinks=False)
        try:
            self._connection = sqlite3.connect(
                self._path,
                timeout=30,
                isolation_level=None,
                check_same_thread=False,
            )
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._initialize_schema()
            self.verify_integrity()
        except WorldStateError:
            connection = getattr(self, "_connection", None)
            if connection is not None:
                connection.close()
            raise
        except (OSError, sqlite3.Error, ValueError) as error:
            connection = getattr(self, "_connection", None)
            if connection is not None:
                connection.close()
            raise WorldStateCorruptionError("World-state database could not be opened or verified") from error

    def _create_file(self, file_mode: int) -> None:
        flags = os.O_CREAT | os.O_EXCL | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self._path, flags, file_mode)
        except FileExistsError:
            if self._path.is_symlink() or not self._path.is_file():
                raise ValueError("storage_path must be a regular non-symlink file")
            return
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        directory_descriptor = os.open(self._path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)

    def _initialize_schema(self) -> None:
        version = self._connection.execute("PRAGMA user_version").fetchone()[0]
        if version == 0:
            existing = self._connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
            if existing:
                raise WorldStateCorruptionError("Unversioned database contains unexpected tables")
            self._connection.executescript(
                """
                CREATE TABLE world_events (
                    sequence INTEGER PRIMARY KEY,
                    assertion_id TEXT NOT NULL UNIQUE,
                    fact_key TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    previous_hash TEXT,
                    event_hash TEXT NOT NULL UNIQUE
                );
                CREATE INDEX world_events_scope_fact ON world_events (
                    tenant_id, user_id, project_id, workspace_id, fact_key, sequence
                );
                CREATE TRIGGER world_events_no_update BEFORE UPDATE ON world_events
                    BEGIN SELECT RAISE(ABORT, 'world-state events are append-only'); END;
                CREATE TRIGGER world_events_no_delete BEFORE DELETE ON world_events
                    BEGIN SELECT RAISE(ABORT, 'world-state events are append-only'); END;
                PRAGMA user_version=1;
                """
            )
        elif version != 1:
            raise WorldStateCorruptionError("Unsupported world-state schema version")

    def append(self, assertion: WorldStateAssertion) -> WorldStateRecord:
        if not isinstance(assertion, WorldStateAssertion):
            raise TypeError("assertion must be a WorldStateAssertion")
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                existing = self._connection.execute(
                    "SELECT sequence, payload_json, previous_hash, event_hash FROM world_events WHERE assertion_id = ?",
                    (assertion.assertion_id,),
                ).fetchone()
                if existing is not None:
                    record = self._decode_row(existing)
                    if self._assertion_payload(record.assertion) != self._assertion_payload(assertion):
                        raise WorldStateConflictError("Assertion ID was reused with different data")
                    self._connection.execute("COMMIT")
                    return record

                conflict_rows = self._connection.execute(
                    """SELECT sequence, payload_json, previous_hash, event_hash
                       FROM world_events WHERE tenant_id = ? AND user_id = ?
                       AND project_id = ? AND workspace_id = ? AND fact_key = ?
                       ORDER BY sequence""",
                    (*_scope_payload(assertion.scope).values(), assertion.fact_key),
                ).fetchall()
                history = tuple(self._decode_row(row) for row in conflict_rows)
                superseded = {
                    assertion_id
                    for record in history
                    for assertion_id in record.superseded_assertion_ids
                }
                active = tuple(record for record in history if record.assertion.assertion_id not in superseded)
                assertion_value = _canonical_json(assertion.value)
                conflicts = tuple(
                    record
                    for record in active
                    if _canonical_json(record.assertion.value) != assertion_value
                )
                decision = self._admission_verifier.review(assertion, conflicts)
                if not isinstance(decision, WorldStateAdmissionDecision) or not isinstance(decision.approved, bool):
                    raise WorldStateAdmissionDenied("Admission verifier returned an invalid decision")
                verification_reference = _text(decision.verification_reference, "verification_reference")
                requested_supersessions = tuple(decision.superseded_assertion_ids)
                if len(set(requested_supersessions)) != len(requested_supersessions):
                    raise WorldStateAdmissionDenied("Admission decision repeats a superseded assertion")
                if not decision.approved:
                    raise WorldStateAdmissionDenied("World-state assertion was not approved")
                if conflicts:
                    expected_ids = {record.assertion.assertion_id for record in conflicts}
                    if set(requested_supersessions) != expected_ids:
                        raise WorldStateConflictError("Conflicting assertions require explicit complete reconciliation")
                    resolution_reference = _text(decision.resolution_reference, "resolution_reference")
                else:
                    if requested_supersessions or decision.resolution_reference is not None:
                        raise WorldStateAdmissionDenied("Admission decision references a nonexistent conflict")
                    resolution_reference = None

                occurred_at_ns = self._clock()
                if isinstance(occurred_at_ns, bool) or not isinstance(occurred_at_ns, int):
                    raise ValueError("clock must return an integer nanosecond timestamp")
                if occurred_at_ns < 0 or occurred_at_ns > sys.maxsize:
                    raise ValueError("clock timestamp is outside the supported range")
                payload = {
                    "assertion": self._assertion_payload(assertion),
                    "occurred_at_ns": occurred_at_ns,
                    "verification_reference": verification_reference,
                    "resolution_reference": resolution_reference,
                    "superseded_assertion_ids": sorted(requested_supersessions),
                }
                sequence_row = self._connection.execute(
                    "SELECT sequence, event_hash FROM world_events ORDER BY sequence DESC LIMIT 1"
                ).fetchone()
                sequence = 1 if sequence_row is None else sequence_row[0] + 1
                previous_hash = None if sequence_row is None else sequence_row[1]
                event_hash = self._event_hash(sequence, payload, previous_hash)
                self._connection.execute(
                    """INSERT INTO world_events
                       (sequence, assertion_id, fact_key, tenant_id, user_id, project_id,
                        workspace_id, payload_json, previous_hash, event_hash)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        sequence,
                        assertion.assertion_id,
                        assertion.fact_key,
                        assertion.scope.tenant_id,
                        assertion.scope.user_id,
                        assertion.scope.project_id,
                        assertion.scope.workspace_id,
                        _canonical_json(payload),
                        previous_hash,
                        event_hash,
                    ),
                )
                self._connection.execute("COMMIT")
                return WorldStateRecord(
                    sequence,
                    assertion,
                    occurred_at_ns,
                    verification_reference,
                    resolution_reference,
                    tuple(sorted(requested_supersessions)),
                    previous_hash,
                    event_hash,
                )
            except WorldStateError:
                self._rollback()
                raise
            except (OSError, sqlite3.Error, TypeError, ValueError) as error:
                self._rollback()
                raise WorldStatePersistenceError("World-state assertion could not be durably appended") from error
            except Exception:
                self._rollback()
                raise

    def query(self, scope: ScopeVector, *, fact_key: str | None = None) -> tuple[WorldStateRecord, ...]:
        if not isinstance(scope, ScopeVector) or not scope.is_complete():
            raise ValueError("A complete four-field scope is required")
        if fact_key is not None:
            _text(fact_key, "fact_key")
        with self._lock:
            try:
                sql = """SELECT sequence, payload_json, previous_hash, event_hash
                         FROM world_events WHERE tenant_id = ? AND user_id = ?
                         AND project_id = ? AND workspace_id = ?"""
                parameters: tuple[object, ...] = tuple(_scope_payload(scope).values())
                if fact_key is not None:
                    sql += " AND fact_key = ?"
                    parameters += (fact_key,)
                sql += " ORDER BY sequence"
                history = tuple(self._decode_row(row) for row in self._connection.execute(sql, parameters).fetchall())
                superseded = {
                    assertion_id
                    for record in history
                    for assertion_id in record.superseded_assertion_ids
                }
                return tuple(
                    record
                    for record in history
                    if record.assertion.assertion_id not in superseded
                )
            except (sqlite3.Error, ValueError, TypeError) as error:
                raise WorldStateCorruptionError("World-state query encountered invalid stored data") from error

    def get_record(self, assertion_id: str, scope: ScopeVector) -> WorldStateRecord | None:
        _text(assertion_id, "assertion_id")
        if not isinstance(scope, ScopeVector) or not scope.is_complete():
            raise ValueError("A complete four-field scope is required")
        with self._lock:
            self.verify_integrity()
            row = self._connection.execute(
                """SELECT sequence, payload_json, previous_hash, event_hash
                   FROM world_events WHERE assertion_id = ? AND tenant_id = ? AND user_id = ?
                   AND project_id = ? AND workspace_id = ?""",
                (assertion_id, *_scope_payload(scope).values()),
            ).fetchone()
            return None if row is None else self._decode_row(row)

    def records(self, scope: ScopeVector) -> tuple[WorldStateRecord, ...]:
        if not isinstance(scope, ScopeVector) or not scope.is_complete():
            raise ValueError("A complete four-field scope is required")
        with self._lock:
            self.verify_integrity()
            rows = self._connection.execute(
                """SELECT sequence, payload_json, previous_hash, event_hash
                   FROM world_events WHERE tenant_id = ? AND user_id = ?
                   AND project_id = ? AND workspace_id = ? ORDER BY sequence""",
                tuple(_scope_payload(scope).values()),
            ).fetchall()
            return tuple(self._decode_row(row) for row in rows)

    def verify_integrity(self) -> None:
        with self._lock:
            previous_hash: str | None = None
            known: dict[str, WorldStateRecord] = {}
            rows = self._connection.execute(
                "SELECT sequence, payload_json, previous_hash, event_hash FROM world_events ORDER BY sequence"
            ).fetchall()
            for expected_sequence, row in enumerate(rows, start=1):
                record = self._decode_row(row, expected_sequence, previous_hash)
                for superseded_id in record.superseded_assertion_ids:
                    predecessor = known.get(superseded_id)
                    if (
                        predecessor is None
                        or predecessor.assertion.scope != record.assertion.scope
                        or predecessor.assertion.fact_key != record.assertion.fact_key
                    ):
                        raise WorldStateCorruptionError("World-state reconciliation references an invalid assertion")
                known[record.assertion.assertion_id] = record
                previous_hash = record.event_hash

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> WorldStateLedger:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def _rollback(self) -> None:
        if self._connection.in_transaction:
            self._connection.execute("ROLLBACK")

    @staticmethod
    def _assertion_payload(assertion: WorldStateAssertion) -> dict[str, object]:
        return {
            "assertion_id": assertion.assertion_id,
            "fact_key": assertion.fact_key,
            "scope": _scope_payload(assertion.scope),
            "subject": assertion.subject,
            "predicate": assertion.predicate,
            "value": json.loads(_canonical_json(assertion.value)),
            "source_ref": assertion.source_ref,
            "source_sha256": assertion.source_sha256,
            "transaction_id": assertion.transaction_id,
        }

    @staticmethod
    def _event_hash(sequence: int, payload: dict[str, object], previous_hash: str | None) -> str:
        body = {"sequence": sequence, "payload": payload, "previous_hash": previous_hash}
        return hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()

    def _decode_row(
        self,
        row: tuple[object, ...],
        expected_sequence: int | None = None,
        expected_previous_hash: str | None = None,
    ) -> WorldStateRecord:
        sequence, payload_json, previous_hash, event_hash = row
        if isinstance(sequence, bool) or not isinstance(sequence, int):
            raise WorldStateCorruptionError("World-state sequence is invalid")
        if expected_sequence is not None and sequence != expected_sequence:
            raise WorldStateCorruptionError("World-state sequence is not contiguous")
        if previous_hash != expected_previous_hash and expected_sequence is not None:
            raise WorldStateCorruptionError("World-state hash chain is broken")
        try:
            payload = json.loads(payload_json, object_pairs_hook=_unique_object)
        except (TypeError, json.JSONDecodeError, ValueError) as error:
            raise WorldStateCorruptionError("World-state payload is malformed") from error
        if not isinstance(payload, dict) or set(payload) != self._PAYLOAD_FIELDS:
            raise WorldStateCorruptionError("World-state payload has an unexpected shape")
        if self._event_hash(sequence, payload, previous_hash) != event_hash:
            raise WorldStateCorruptionError("World-state hash verification failed")
        assertion_data = payload["assertion"]
        if not isinstance(assertion_data, dict) or set(assertion_data) != self._ASSERTION_FIELDS:
            raise WorldStateCorruptionError("World-state assertion has an unexpected shape")
        scope_data = assertion_data["scope"]
        if not isinstance(scope_data, dict) or set(scope_data) != {
            "tenant_id", "user_id", "project_id", "workspace_id"
        }:
            raise WorldStateCorruptionError("World-state scope is malformed")
        try:
            assertion = WorldStateAssertion(
                assertion_id=assertion_data["assertion_id"],
                fact_key=assertion_data["fact_key"],
                scope=ScopeVector(**scope_data),
                subject=assertion_data["subject"],
                predicate=assertion_data["predicate"],
                value=assertion_data["value"],
                source_ref=assertion_data["source_ref"],
                source_sha256=assertion_data["source_sha256"],
                transaction_id=assertion_data["transaction_id"],
            )
            occurred_at_ns = payload["occurred_at_ns"]
            if (
                isinstance(occurred_at_ns, bool)
                or not isinstance(occurred_at_ns, int)
                or occurred_at_ns < 0
                or occurred_at_ns > sys.maxsize
            ):
                raise ValueError("invalid timestamp")
            verification_reference = _text(payload["verification_reference"], "verification_reference")
            resolution_reference = payload["resolution_reference"]
            if resolution_reference is not None:
                resolution_reference = _text(resolution_reference, "resolution_reference")
            superseded = payload["superseded_assertion_ids"]
            if (
                not isinstance(superseded, list)
                or any(not isinstance(item, str) or not item.strip() for item in superseded)
                or len(set(superseded)) != len(superseded)
                or superseded != sorted(superseded)
            ):
                raise ValueError("invalid supersession list")
            if bool(superseded) != bool(resolution_reference):
                raise ValueError("resolution reference and supersession list disagree")
        except (KeyError, TypeError, ValueError) as error:
            raise WorldStateCorruptionError("World-state event fields are invalid") from error
        return WorldStateRecord(
            sequence,
            assertion,
            occurred_at_ns,
            verification_reference,
            resolution_reference,
            tuple(superseded),
            previous_hash,
            event_hash,
        )