from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from time import time_ns
from typing import Callable, Mapping, Protocol

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


class GovernanceApprovalError(PermissionError):
    pass


class GovernanceApprovalReplayError(GovernanceApprovalError):
    pass


@dataclass(frozen=True, slots=True)
class SignedGovernanceApproval:
    approval_id: str
    reviewer_id: str
    reviewer_role: str
    action: str
    subject_sha256: str
    audience: str
    issued_at: int
    expires_at: int
    signature: bytes


def governance_approval_message(approval: SignedGovernanceApproval) -> bytes:
    if not isinstance(approval, SignedGovernanceApproval):
        raise TypeError("approval must be a SignedGovernanceApproval")
    payload = {
        "action": approval.action,
        "approval_id": approval.approval_id,
        "audience": approval.audience,
        "expires_at": approval.expires_at,
        "issued_at": approval.issued_at,
        "reviewer_id": approval.reviewer_id,
        "reviewer_role": approval.reviewer_role,
        "subject_sha256": approval.subject_sha256,
        "version": 1,
    }
    return b"STACEY-GOVERNANCE-APPROVAL-v1\0" + json.dumps(
        payload,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


class GovernanceApprovalVerifier(Protocol):
    def verify(
        self,
        approval: SignedGovernanceApproval,
        *,
        action: str,
        subject_sha256: str,
    ) -> str: ...


class Ed25519GovernanceApprovalVerifier:
    def __init__(
        self,
        reviewer_public_keys: Mapping[tuple[str, str], bytes],
        *,
        allowed_roles_by_action: Mapping[str, tuple[str, ...]],
        audience: str,
        clock: Callable[[], float] = time.time,
        max_lifetime_seconds: int = 3600,
    ) -> None:
        if not isinstance(reviewer_public_keys, Mapping) or not reviewer_public_keys:
            raise ValueError("reviewer_public_keys must be a non-empty mapping")
        if not isinstance(allowed_roles_by_action, Mapping) or not allowed_roles_by_action:
            raise ValueError("allowed_roles_by_action must be a non-empty mapping")
        if not isinstance(audience, str) or not audience.strip():
            raise ValueError("audience must be a non-empty string")
        if not callable(clock):
            raise ValueError("clock must be callable")
        if isinstance(max_lifetime_seconds, bool) or not isinstance(max_lifetime_seconds, int):
            raise ValueError("max_lifetime_seconds must be an integer")
        if max_lifetime_seconds <= 0:
            raise ValueError("max_lifetime_seconds must be positive")
        parsed_keys: dict[tuple[str, str], Ed25519PublicKey] = {}
        for identity, encoded_key in reviewer_public_keys.items():
            if (
                not isinstance(identity, tuple)
                or len(identity) != 2
                or any(not isinstance(value, str) or not value.strip() for value in identity)
            ):
                raise ValueError("reviewer key identities must be (reviewer_id, role) tuples")
            try:
                parsed_keys[identity] = Ed25519PublicKey.from_public_bytes(encoded_key)
            except (TypeError, ValueError) as error:
                raise ValueError("reviewer public keys must contain 32 raw Ed25519 bytes") from error
        checked_roles: dict[str, frozenset[str]] = {}
        for action, roles in allowed_roles_by_action.items():
            if not isinstance(action, str) or not action.strip():
                raise ValueError("approval action names must be non-empty strings")
            if not isinstance(roles, tuple) or not roles or any(
                not isinstance(role, str) or not role.strip() for role in roles
            ):
                raise ValueError("each action must have a non-empty tuple of allowed reviewer roles")
            if len(set(roles)) != len(roles):
                raise ValueError("allowed reviewer roles must not contain duplicates")
            checked_roles[action] = frozenset(roles)
        self._public_keys = parsed_keys
        self._allowed_roles = checked_roles
        self._audience = audience
        self._clock = clock
        self._max_lifetime_seconds = max_lifetime_seconds

    def verify(
        self,
        approval: SignedGovernanceApproval,
        *,
        action: str,
        subject_sha256: str,
    ) -> str:
        if not isinstance(approval, SignedGovernanceApproval):
            raise GovernanceApprovalError("A signed governance approval is required")
        for field_name in ("approval_id", "reviewer_id", "reviewer_role", "action", "audience"):
            value = getattr(approval, field_name)
            if not isinstance(value, str) or not value.strip():
                raise GovernanceApprovalError(f"Approval {field_name} is invalid")
        if (
            not isinstance(subject_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", subject_sha256) is None
            or not isinstance(approval.subject_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", approval.subject_sha256) is None
        ):
            raise GovernanceApprovalError("Approval subject must be a lowercase SHA-256 digest")
        if approval.action != action or approval.subject_sha256 != subject_sha256:
            raise GovernanceApprovalError("Approval does not bind to the requested action and exact subject digest")
        if approval.audience != self._audience:
            raise GovernanceApprovalError("Approval audience does not match this deployment")
        if approval.reviewer_role not in self._allowed_roles.get(action, frozenset()):
            raise GovernanceApprovalError("Reviewer role is not authorized for this approval action")
        if (
            isinstance(approval.issued_at, bool)
            or not isinstance(approval.issued_at, int)
            or isinstance(approval.expires_at, bool)
            or not isinstance(approval.expires_at, int)
        ):
            raise GovernanceApprovalError("Approval timestamps are invalid")
        now = self._clock()
        if (
            approval.expires_at <= approval.issued_at
            or approval.expires_at - approval.issued_at > self._max_lifetime_seconds
            or approval.issued_at > now
            or approval.expires_at <= now
        ):
            raise GovernanceApprovalError("Approval is outside its valid time window")
        public_key = self._public_keys.get((approval.reviewer_id, approval.reviewer_role))
        if public_key is None:
            raise GovernanceApprovalError("Reviewer identity is not trusted for this role")
        if not isinstance(approval.signature, bytes) or len(approval.signature) != 64:
            raise GovernanceApprovalError("Approval signature is malformed")
        try:
            public_key.verify(approval.signature, governance_approval_message(approval))
        except InvalidSignature as error:
            raise GovernanceApprovalError("Approval signature is invalid") from error
        return approval.approval_id


class GovernanceApprovalJournal:
    """Private append-only journal that prevents a signed training approval from replaying."""

    def __init__(
        self,
        database_path: str | os.PathLike[str],
        *,
        verifier: GovernanceApprovalVerifier,
        file_mode: int = 0o600,
        clock: Callable[[], int] = time_ns,
    ) -> None:
        path = Path(database_path)
        if path.is_symlink() or not path.parent.is_dir():
            raise ValueError("database_path must be a non-symlink file in an existing directory")
        if not callable(getattr(verifier, "verify", None)) or not callable(clock):
            raise ValueError("verifier and clock must be callable")
        if isinstance(file_mode, bool) or not isinstance(file_mode, int) or file_mode < 0 or file_mode & ~0o777:
            raise ValueError("file_mode must contain only permission bits")
        self._verifier = verifier
        self._clock = clock
        self._lock = RLock()
        self._closed = False
        flags = os.O_CREAT | os.O_EXCL | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags, file_mode)
        except FileExistsError:
            if path.is_symlink() or not path.is_file():
                raise ValueError("database_path must identify a regular non-symlink file")
        else:
            os.fsync(descriptor)
            os.close(descriptor)
        os.chmod(path, file_mode, follow_symlinks=False)
        self._connection = sqlite3.connect(path.resolve(), timeout=30, isolation_level=None, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        version = self._connection.execute("PRAGMA user_version").fetchone()[0]
        if version == 0:
            existing = self._connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
            if existing:
                raise GovernanceApprovalError("Unversioned approval journal contains unexpected tables")
            self._connection.executescript(
                """
                CREATE TABLE consumed_approvals (
                    approval_id TEXT PRIMARY KEY,
                    action TEXT NOT NULL,
                    subject_sha256 TEXT NOT NULL,
                    reviewer_id TEXT NOT NULL,
                    reviewer_role TEXT NOT NULL,
                    consumed_at_ns INTEGER NOT NULL,
                    signature_sha256 TEXT NOT NULL
                );
                CREATE TRIGGER consumed_approvals_no_update BEFORE UPDATE ON consumed_approvals
                    BEGIN SELECT RAISE(ABORT, 'consumed approvals are immutable'); END;
                CREATE TRIGGER consumed_approvals_no_delete BEFORE DELETE ON consumed_approvals
                    BEGIN SELECT RAISE(ABORT, 'consumed approvals are immutable'); END;
                PRAGMA user_version=1;
                """
            )
        elif version != 1:
            raise GovernanceApprovalError("Unsupported approval journal schema version")

    def consume(
        self,
        approval: SignedGovernanceApproval,
        *,
        action: str,
        subject_sha256: str,
    ) -> str:
        approval_id = self._verifier.verify(
            approval,
            action=action,
            subject_sha256=subject_sha256,
        )
        occurred_at_ns = self._clock()
        if isinstance(occurred_at_ns, bool) or not isinstance(occurred_at_ns, int):
            raise GovernanceApprovalError("approval journal clock must return an integer timestamp")
        if occurred_at_ns < 0 or occurred_at_ns > sys.maxsize:
            raise GovernanceApprovalError("approval journal timestamp is outside the supported range")
        signature_sha256 = hashlib.sha256(approval.signature).hexdigest()
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                self._connection.execute(
                    """INSERT INTO consumed_approvals
                       (approval_id, action, subject_sha256, reviewer_id, reviewer_role,
                        consumed_at_ns, signature_sha256)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        approval_id,
                        action,
                        subject_sha256,
                        approval.reviewer_id,
                        approval.reviewer_role,
                        occurred_at_ns,
                        signature_sha256,
                    ),
                )
                self._connection.execute("COMMIT")
            except sqlite3.IntegrityError as error:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise GovernanceApprovalReplayError("Signed approval has already been consumed") from error
            except Exception:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
        return approval_id

    def confirm_consumed(
        self,
        approval: SignedGovernanceApproval,
        *,
        action: str,
        subject_sha256: str,
    ) -> str:
        """Confirm exact prior consumption for checkpoint resume without permitting a new run."""
        approval_id = self._verifier.verify(
            approval,
            action=action,
            subject_sha256=subject_sha256,
        )
        signature_sha256 = hashlib.sha256(approval.signature).hexdigest()
        with self._lock:
            row = self._connection.execute(
                """SELECT action, subject_sha256, reviewer_id, reviewer_role, signature_sha256
                   FROM consumed_approvals WHERE approval_id = ?""",
                (approval_id,),
            ).fetchone()
        expected = (
            action,
            subject_sha256,
            approval.reviewer_id,
            approval.reviewer_role,
            signature_sha256,
        )
        if row != expected:
            raise GovernanceApprovalError("resume approval was not previously consumed for this exact run")
        return approval_id

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    def __enter__(self) -> GovernanceApprovalJournal:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def sha256_bytes(content: bytes) -> str:
    if not isinstance(content, bytes):
        raise TypeError("content must be bytes")
    return hashlib.sha256(content).hexdigest()