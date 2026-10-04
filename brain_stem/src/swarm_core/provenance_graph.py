from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from time import time_ns
from typing import Callable, Protocol

from substrate.contracts import ScopeVector

from .model_lifecycle import ModelArtifactManifest, ModelArtifactRegistry
from .world_state import WorldStateLedger, WorldStateRecord


class ProvenanceGraphError(RuntimeError):
    pass


class ProvenanceGraphCorruptionError(ProvenanceGraphError):
    pass


class ProvenanceGraphDenied(ProvenanceGraphError):
    pass


class ProvenanceGraphOrphan(ProvenanceGraphError):
    pass


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
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
        raise ValueError("Provenance graph values must be finite JSON-compatible values") from error


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
class ProvenanceEdge:
    source_node_id: str
    relation: str
    target_node_id: str

    def __post_init__(self) -> None:
        _text(self.source_node_id, "source_node_id")
        _text(self.target_node_id, "target_node_id")
        if self.relation not in {"SUPPORTS", "PRODUCED", "RECORDED", "ADMITTED"}:
            raise ValueError("relation is not in the provenance graph allowlist")


@dataclass(frozen=True, slots=True)
class ProvenanceGraphRecord:
    sequence: int
    link_id: str
    scope: ScopeVector
    assertion_id: str
    artifact_id: str
    artifact_sha256: str
    source_ref: str
    source_sha256: str
    transaction_id: str
    ledger_event_hash: str
    verification_reference: str
    edges: tuple[ProvenanceEdge, ...]
    occurred_at_ns: int
    previous_hash: str | None
    event_hash: str


class ProvenanceLinkVerifier(Protocol):
    def verify(self, record: WorldStateRecord, artifact: ModelArtifactManifest) -> str: ...


class ProvenanceGraphLinker:
    """Persists admitted, scope-bound artifact/source/transaction links for FB-004 facts."""

    _LINK_FIELDS = frozenset(
        {
            "assertion_id",
            "scope",
            "artifact_id",
            "artifact_sha256",
            "source_ref",
            "source_sha256",
            "transaction_id",
            "ledger_event_hash",
            "verification_reference",
            "edges",
        }
    )

    def __init__(
        self,
        storage_path: str | os.PathLike[str],
        *,
        world_state_ledger: WorldStateLedger,
        artifact_registry: object,
        link_verifier: ProvenanceLinkVerifier,
        file_mode: int = 0o600,
        clock: Callable[[], int] = time_ns,
    ) -> None:
        if not isinstance(world_state_ledger, WorldStateLedger):
            raise ValueError("world_state_ledger must be a verified WorldStateLedger")
        if not isinstance(artifact_registry, ModelArtifactRegistry):
            raise ValueError("artifact_registry must be a signature-verifying ModelArtifactRegistry")
        if not callable(getattr(link_verifier, "verify", None)):
            raise ValueError("link_verifier must implement verify()")
        if not callable(clock):
            raise ValueError("clock must be callable")
        if isinstance(file_mode, bool) or not isinstance(file_mode, int) or file_mode < 0 or file_mode & ~0o777:
            raise ValueError("file_mode must contain only permission bits")
        path = Path(storage_path)
        if not path.name or not path.parent.is_dir() or path.is_symlink():
            raise ValueError("storage_path must be a non-symlink file in an existing directory")
        self._path = path.resolve()
        self._clock = clock
        self._world_state_ledger = world_state_ledger
        self._artifact_registry = artifact_registry
        self._link_verifier = link_verifier
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
            self._initialize_schema()
            self.verify_integrity()
        except ProvenanceGraphError:
            connection = getattr(self, "_connection", None)
            if connection is not None:
                connection.close()
            raise
        except (OSError, sqlite3.Error, ValueError) as error:
            connection = getattr(self, "_connection", None)
            if connection is not None:
                connection.close()
            raise ProvenanceGraphCorruptionError("Provenance graph could not be opened or verified") from error

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
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
            if existing:
                raise ProvenanceGraphCorruptionError("Unversioned graph database contains unexpected tables")
            self._connection.executescript(
                """
                CREATE TABLE provenance_events (
                    sequence INTEGER PRIMARY KEY,
                    link_id TEXT NOT NULL UNIQUE,
                    assertion_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    previous_hash TEXT,
                    event_hash TEXT NOT NULL UNIQUE
                );
                CREATE INDEX provenance_scope_assertion ON provenance_events (
                    tenant_id, user_id, project_id, workspace_id, assertion_id, sequence
                );
                CREATE TRIGGER provenance_events_no_update BEFORE UPDATE ON provenance_events
                    BEGIN SELECT RAISE(ABORT, 'provenance events are append-only'); END;
                CREATE TRIGGER provenance_events_no_delete BEFORE DELETE ON provenance_events
                    BEGIN SELECT RAISE(ABORT, 'provenance events are append-only'); END;
                PRAGMA user_version=1;
                """
            )
            version = 1
        if version != 1:
            raise ProvenanceGraphCorruptionError("Unsupported provenance graph schema version")

    def link_assertion(self, assertion_id: str, scope: ScopeVector, artifact_sha256: str) -> ProvenanceGraphRecord:
        _text(assertion_id, "assertion_id")
        _digest(artifact_sha256, "artifact_sha256")
        if not isinstance(scope, ScopeVector) or not scope.is_complete():
            raise ValueError("A complete four-field scope is required")
        ledger_record = self._world_state_ledger.get_record(assertion_id, scope)
        if ledger_record is None:
            raise ProvenanceGraphOrphan("No accepted FB-004 assertion exists in the requested scope")
        artifact = self._artifact_registry.get_by_digest(artifact_sha256)
        if not isinstance(artifact, ModelArtifactManifest) or artifact.artifact_sha256 != artifact_sha256:
            raise ProvenanceGraphDenied("Artifact registry did not resolve the exact signed artifact")
        verification_reference = _text(
            self._link_verifier.verify(ledger_record, artifact),
            "verification_reference",
        )
        assertion = ledger_record.assertion
        assertion_node = f"assertion:{assertion.assertion_id}"
        link_payload = {
            "assertion_id": assertion.assertion_id,
            "scope": _scope_payload(assertion.scope),
            "artifact_id": artifact.artifact_id,
            "artifact_sha256": artifact.artifact_sha256,
            "source_ref": assertion.source_ref,
            "source_sha256": assertion.source_sha256,
            "transaction_id": assertion.transaction_id,
            "ledger_event_hash": ledger_record.event_hash,
            "verification_reference": verification_reference,
            "edges": [
                {
                    "source_node_id": f"source:sha256:{assertion.source_sha256}",
                    "relation": "SUPPORTS",
                    "target_node_id": assertion_node,
                },
                {
                    "source_node_id": f"artifact:sha256:{artifact.artifact_sha256}",
                    "relation": "PRODUCED",
                    "target_node_id": assertion_node,
                },
                {
                    "source_node_id": f"transaction:{assertion.transaction_id}",
                    "relation": "RECORDED",
                    "target_node_id": assertion_node,
                },
                {
                    "source_node_id": f"verification:{verification_reference}",
                    "relation": "ADMITTED",
                    "target_node_id": assertion_node,
                },
            ],
        }
        link_id = hashlib.sha256(_canonical_json(link_payload).encode("utf-8")).hexdigest()
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                existing = self._connection.execute(
                    "SELECT sequence, link_id, tenant_id, user_id, project_id, workspace_id, "
                    "payload_json, previous_hash, event_hash "
                    "FROM provenance_events WHERE link_id = ?",
                    (link_id,),
                ).fetchone()
                if existing is not None:
                    record = self._decode_row(existing)
                    self._connection.execute("COMMIT")
                    return record

                occurred_at_ns = self._clock()
                if isinstance(occurred_at_ns, bool) or not isinstance(occurred_at_ns, int) or occurred_at_ns < 0:
                    raise ValueError("clock must return a non-negative integer nanosecond timestamp")
                last = self._connection.execute(
                    "SELECT sequence, event_hash FROM provenance_events ORDER BY sequence DESC LIMIT 1"
                ).fetchone()
                sequence = 1 if last is None else last[0] + 1
                previous_hash = None if last is None else last[1]
                payload = {"link": link_payload, "occurred_at_ns": occurred_at_ns}
                payload_json = _canonical_json(payload)
                event_hash = self._event_hash(sequence, payload, previous_hash)
                self._connection.execute(
                    """INSERT INTO provenance_events
                       (sequence, link_id, assertion_id, tenant_id, user_id, project_id,
                        workspace_id, payload_json, previous_hash, event_hash)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        sequence,
                        link_id,
                        assertion.assertion_id,
                        assertion.scope.tenant_id,
                        assertion.scope.user_id,
                        assertion.scope.project_id,
                        assertion.scope.workspace_id,
                        payload_json,
                        previous_hash,
                        event_hash,
                    ),
                )
                self._connection.execute("COMMIT")
                return self._decode_row(
                    (
                        sequence,
                        link_id,
                        assertion.scope.tenant_id,
                        assertion.scope.user_id,
                        assertion.scope.project_id,
                        assertion.scope.workspace_id,
                        payload_json,
                        previous_hash,
                        event_hash,
                    )
                )
            except ProvenanceGraphError:
                self._rollback()
                raise
            except (OSError, sqlite3.Error, TypeError, ValueError) as error:
                self._rollback()
                raise ProvenanceGraphError("Provenance graph link could not be durably appended") from error
            except Exception:
                self._rollback()
                raise

    def query(
        self,
        scope: ScopeVector,
        *,
        assertion_id: str | None = None,
    ) -> tuple[ProvenanceGraphRecord, ...]:
        if not isinstance(scope, ScopeVector) or not scope.is_complete():
            raise ValueError("A complete four-field scope is required")
        if assertion_id is not None:
            _text(assertion_id, "assertion_id")
        with self._lock:
            self.verify_integrity()
            sql = """SELECT sequence, link_id, tenant_id, user_id, project_id, workspace_id,
                     payload_json, previous_hash, event_hash
                     FROM provenance_events WHERE tenant_id = ? AND user_id = ?
                     AND project_id = ? AND workspace_id = ?"""
            parameters: tuple[object, ...] = tuple(_scope_payload(scope).values())
            if assertion_id is not None:
                sql += " AND assertion_id = ?"
                parameters += (assertion_id,)
            sql += " ORDER BY sequence"
            rows = self._connection.execute(sql, parameters).fetchall()
            graph_records = tuple(self._decode_row(row) for row in rows)
            try:
                ledger_records = {
                    record.assertion.assertion_id: record
                    for record in self._world_state_ledger.records(scope)
                }
            except Exception as error:
                raise ProvenanceGraphCorruptionError("Linked FB-004 ledger failed integrity verification") from error
            for graph_record in graph_records:
                ledger_record = ledger_records.get(graph_record.assertion_id)
                if (
                    ledger_record is None
                    or ledger_record.event_hash != graph_record.ledger_event_hash
                    or ledger_record.assertion.source_ref != graph_record.source_ref
                    or ledger_record.assertion.source_sha256 != graph_record.source_sha256
                    or ledger_record.assertion.transaction_id != graph_record.transaction_id
                ):
                    raise ProvenanceGraphCorruptionError("Provenance link no longer matches its FB-004 record")
            return graph_records

    def verify_integrity(self) -> None:
        with self._lock:
            previous_hash: str | None = None
            rows = self._connection.execute(
                "SELECT sequence, link_id, tenant_id, user_id, project_id, workspace_id, "
                "payload_json, previous_hash, event_hash "
                "FROM provenance_events ORDER BY sequence"
            ).fetchall()
            for expected_sequence, row in enumerate(rows, start=1):
                record = self._decode_row(row, expected_sequence, previous_hash)
                previous_hash = record.event_hash

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> ProvenanceGraphLinker:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def _rollback(self) -> None:
        if self._connection.in_transaction:
            self._connection.execute("ROLLBACK")

    @staticmethod
    def _event_hash(sequence: int, payload: dict[str, object], previous_hash: str | None) -> str:
        body = {"sequence": sequence, "payload": payload, "previous_hash": previous_hash}
        return hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()

    def _decode_row(
        self,
        row: tuple[object, ...],
        expected_sequence: int | None = None,
        expected_previous_hash: str | None = None,
    ) -> ProvenanceGraphRecord:
        sequence, link_id, tenant_id, user_id, project_id, workspace_id, payload_json, previous_hash, event_hash = row
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= 0:
            raise ProvenanceGraphCorruptionError("Provenance sequence is invalid")
        if expected_sequence is not None and sequence != expected_sequence:
            raise ProvenanceGraphCorruptionError("Provenance sequence is not contiguous")
        if expected_sequence is not None and previous_hash != expected_previous_hash:
            raise ProvenanceGraphCorruptionError("Provenance hash chain is broken")
        try:
            payload = json.loads(payload_json, object_pairs_hook=_unique_object)
        except (TypeError, json.JSONDecodeError, ValueError) as error:
            raise ProvenanceGraphCorruptionError("Provenance payload is malformed") from error
        if not isinstance(payload, dict) or set(payload) != {"link", "occurred_at_ns"}:
            raise ProvenanceGraphCorruptionError("Provenance event has an unexpected shape")
        if self._event_hash(sequence, payload, previous_hash) != event_hash:
            raise ProvenanceGraphCorruptionError("Provenance event hash verification failed")
        link = payload["link"]
        if not isinstance(link, dict) or set(link) != self._LINK_FIELDS:
            raise ProvenanceGraphCorruptionError("Provenance link has an unexpected shape")
        if hashlib.sha256(_canonical_json(link).encode("utf-8")).hexdigest() != link_id:
            raise ProvenanceGraphCorruptionError("Provenance link ID does not match its payload")
        try:
            scope_data = link["scope"]
            if not isinstance(scope_data, dict) or set(scope_data) != {
                "tenant_id", "user_id", "project_id", "workspace_id"
            }:
                raise ValueError("invalid scope")
            scope = ScopeVector(**scope_data)
            if not scope.is_complete():
                raise ValueError("incomplete scope")
            artifact_sha256 = _digest(link["artifact_sha256"], "artifact_sha256")
            source_sha256 = _digest(link["source_sha256"], "source_sha256")
            ledger_event_hash = _digest(link["ledger_event_hash"], "ledger_event_hash")
            edges_data = link["edges"]
            if not isinstance(edges_data, list):
                raise ValueError("edges must be an array")
            edges = tuple(ProvenanceEdge(**edge) for edge in edges_data)
            occurred_at_ns = payload["occurred_at_ns"]
            if isinstance(occurred_at_ns, bool) or not isinstance(occurred_at_ns, int) or occurred_at_ns < 0:
                raise ValueError("invalid timestamp")
            assertion_id = _text(link["assertion_id"], "assertion_id")
            artifact_id = _text(link["artifact_id"], "artifact_id")
            source_ref = _text(link["source_ref"], "source_ref")
            transaction_id = _text(link["transaction_id"], "transaction_id")
            verification_reference = _text(link["verification_reference"], "verification_reference")
            assertion_node = f"assertion:{assertion_id}"
            expected_edges = (
                ProvenanceEdge(f"source:sha256:{source_sha256}", "SUPPORTS", assertion_node),
                ProvenanceEdge(f"artifact:sha256:{artifact_sha256}", "PRODUCED", assertion_node),
                ProvenanceEdge(f"transaction:{transaction_id}", "RECORDED", assertion_node),
                ProvenanceEdge(f"verification:{verification_reference}", "ADMITTED", assertion_node),
            )
            if edges != expected_edges:
                raise ValueError("graph edges do not match their bound provenance fields")
        except (KeyError, TypeError, ValueError) as error:
            raise ProvenanceGraphCorruptionError("Provenance link fields are invalid") from error
        if self._scope_columns(scope) != (tenant_id, user_id, project_id, workspace_id):
            raise ProvenanceGraphCorruptionError("Provenance row scope or previous hash is malformed")
        if previous_hash is not None:
            try:
                _digest(previous_hash, "previous_hash")
                _digest(event_hash, "event_hash")
            except ValueError as error:
                raise ProvenanceGraphCorruptionError("Provenance hash fields are malformed") from error
        return ProvenanceGraphRecord(
            sequence,
            link_id,
            scope,
            assertion_id,
            artifact_id,
            artifact_sha256,
            source_ref,
            source_sha256,
            transaction_id,
            ledger_event_hash,
            verification_reference,
            edges,
            occurred_at_ns,
            previous_hash,
            event_hash,
        )

    @staticmethod
    def _scope_columns(scope: ScopeVector) -> tuple[str, str, str, str]:
        return (scope.tenant_id, scope.user_id, scope.project_id, scope.workspace_id)
