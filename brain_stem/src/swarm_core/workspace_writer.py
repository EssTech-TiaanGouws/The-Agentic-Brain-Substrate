from __future__ import annotations

import errno
import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from threading import RLock
from typing import Mapping, Protocol

try:
    import fcntl
except ImportError:
    fcntl = None

from substrate.contracts import Intent, ScopeVector

from .durable_ledger import DurableReceiptLedger, ReceiptTransition, ReceiptTransaction
from .format_adapters import DocumentFormatRegistry, DocumentPayload, PreparedDocument
from .identity import ScopeAuthorizationVerifier, SignedScopeGrant, intent_authorization_digest
from .lease_manager import ResourceLeaseManager, ResourceProfile
from .transaction_coordinator import (
    CommitOutcome,
    CompensationOutcome,
    IndeterminateTransactionError,
    MutationAuthority,
    OrderedStageVerifier,
    StagedOperation,
    TransactionAborted,
    TransactionCoordinator,
    TransactionRequest,
    TransactionResult,
    TransactionWorker,
    UnresolvedTransactionError,
)


class WorkspaceWriteError(RuntimeError):
    pass


class WorkspacePathRejected(WorkspaceWriteError):
    pass


class WorkspaceUnavailableError(WorkspaceWriteError):
    pass


class WorkspaceCleanupError(WorkspaceWriteError):
    pass


class WorkspaceRootResolver(Protocol):
    def resolve(self, scope: ScopeVector) -> Path: ...


class ConfiguredWorkspaceResolver:
    """Maps trusted four-field scopes to operator-configured local directories."""

    def __init__(self, roots: Mapping[ScopeVector, str | os.PathLike[str]]) -> None:
        if not isinstance(roots, Mapping) or not roots:
            raise ValueError("At least one configured workspace root is required")
        resolved: dict[ScopeVector, Path] = {}
        for scope, configured_path in roots.items():
            if not isinstance(scope, ScopeVector) or not scope.is_complete():
                raise ValueError("Workspace roots must be keyed by complete trusted scopes")
            path = Path(configured_path)
            if path.is_symlink() or not path.is_dir():
                raise ValueError("Workspace root must be an existing non-symlink directory")
            resolved[scope] = path.resolve(strict=True)
        self._roots = resolved

    def resolve(self, scope: ScopeVector) -> Path:
        try:
            return self._roots[scope]
        except KeyError as error:
            raise WorkspaceUnavailableError("No workspace is configured for this trusted scope") from error


def normalize_workspace_relative_path(value: str) -> tuple[str, ...]:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise WorkspacePathRejected("Workspace path must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    parts = path.parts
    if path.is_absolute() or not parts or any(part in {"", ".", ".."} for part in parts):
        raise WorkspacePathRejected("Workspace path must not be absolute or traverse parent directories")
    if any(part.startswith(".stacey-") for part in parts):
        raise WorkspacePathRejected("Workspace path uses a reserved transaction filename")
    if path.as_posix() != value:
        raise WorkspacePathRejected("Workspace path must use its normalized representation")
    return parts


@dataclass(frozen=True, slots=True)
class WorkspaceWriteCommand:
    transaction_id: str
    correlation_id: str
    idempotency_key: str
    scope: ScopeVector
    relative_path: str
    format_key: str
    content: bytes
    expected_current_sha256: str | None
    resource_profile: ResourceProfile
    proposal_release_id: str | None = None
    proposal_artifact_sha256: str | None = None

    def __post_init__(self) -> None:
        for field in ("transaction_id", "correlation_id", "idempotency_key", "format_key"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a non-empty string")
        normalize_workspace_relative_path(self.relative_path)
        if not isinstance(self.scope, ScopeVector) or not self.scope.is_complete():
            raise ValueError("A complete four-field scope is required")
        if not isinstance(self.content, bytes):
            raise ValueError("content must be bytes")
        if self.expected_current_sha256 is not None and (
            not isinstance(self.expected_current_sha256, str)
            or len(self.expected_current_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.expected_current_sha256)
        ):
            raise ValueError("expected_current_sha256 must be a lowercase SHA-256 digest or None")
        if (self.proposal_release_id is None) != (self.proposal_artifact_sha256 is None):
            raise ValueError("proposal release and artifact digest must be supplied together")
        if self.proposal_release_id is not None:
            if not isinstance(self.proposal_release_id, str) or not self.proposal_release_id.strip():
                raise ValueError("proposal_release_id must be a non-empty string")
            if (
                not isinstance(self.proposal_artifact_sha256, str)
                or len(self.proposal_artifact_sha256) != 64
                or any(character not in "0123456789abcdef" for character in self.proposal_artifact_sha256)
            ):
                raise ValueError("proposal_artifact_sha256 must be a lowercase SHA-256 digest")
        if not isinstance(self.resource_profile, ResourceProfile):
            raise ValueError("resource_profile must be a ResourceProfile")


@dataclass(slots=True)
class _Operation:
    command: WorkspaceWriteCommand
    prepared: PreparedDocument
    intent: Intent
    request: TransactionRequest
    root: Path
    authorization: SignedScopeGrant


@dataclass(slots=True)
class _StagedFile:
    parent_fd: int
    lock_fd: int
    target_name: str
    stage_name: str
    backup_name: str
    previous_digest: str | None
    new_digest: str
    committed: bool = False
    compensated: bool = False


class WorkspaceWriteService(TransactionWorker, MutationAuthority):
    def __init__(
        self,
        *,
        workspace_resolver: WorkspaceRootResolver,
        format_registry: DocumentFormatRegistry,
        authorization_verifier: ScopeAuthorizationVerifier,
        lease_manager: ResourceLeaseManager,
        receipt_ledger: DurableReceiptLedger,
        stage_verifier: OrderedStageVerifier,
        max_content_bytes: int,
    ) -> None:
        if isinstance(max_content_bytes, bool) or not isinstance(max_content_bytes, int) or max_content_bytes <= 0:
            raise ValueError("max_content_bytes must be a positive integer")
        if fcntl is None:
            raise WorkspaceUnavailableError("This workspace writer requires a POSIX file-lock backend")
        self._workspace_resolver = workspace_resolver
        self._format_registry = format_registry
        self._authorization_verifier = authorization_verifier
        self._receipt_ledger = receipt_ledger
        self._max_content_bytes = max_content_bytes
        self._operations: dict[str, _Operation] = {}
        self._staged: dict[str, _StagedFile] = {}
        self._lock = RLock()
        self._coordinator = TransactionCoordinator(
            lease_manager=lease_manager,
            ledger=receipt_ledger,
            worker=self,
            verifier=stage_verifier,
            mutation_authority=self,
        )

    def prepare(self, command: WorkspaceWriteCommand) -> PreparedDocument:
        if not isinstance(command, WorkspaceWriteCommand):
            raise TypeError("command must be a WorkspaceWriteCommand")
        if len(command.content) > self._max_content_bytes:
            raise WorkspaceWriteError("Workspace document exceeds the configured byte limit")
        return self._format_registry.prepare(DocumentPayload(command.format_key, command.content))

    @staticmethod
    def confirmation_intent(
        command: WorkspaceWriteCommand,
        prepared: PreparedDocument,
    ) -> Intent:
        if not isinstance(command, WorkspaceWriteCommand) or not isinstance(prepared, PreparedDocument):
            raise TypeError("A write command and prepared document are required")
        if prepared.format_key != command.format_key:
            raise ValueError("Prepared document format does not match the command")
        goal = json.dumps(
            {
                "content_sha256": prepared.digest,
                "expected_current_sha256": command.expected_current_sha256,
                "format_key": prepared.format_key,
                "proposal_artifact_sha256": command.proposal_artifact_sha256,
                "proposal_release_id": command.proposal_release_id,
                "relative_path": command.relative_path,
            },
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return Intent(
            transaction_id=command.transaction_id,
            correlation_id=command.correlation_id,
            action="WRITE_WORKSPACE_FILE",
            goal=goal,
            scope=command.scope,
        )

    def execute(
        self,
        command: WorkspaceWriteCommand,
        authorization: SignedScopeGrant,
    ) -> TransactionResult:
        prepared = self.prepare(command)
        intent = self.confirmation_intent(command, prepared)
        self._authorization_verifier.authorize(intent, authorization)
        root = self._workspace_resolver.resolve(command.scope)
        prior_state_ref = (
            "ABSENT"
            if command.expected_current_sha256 is None
            else f"sha256:{command.expected_current_sha256}"
        )
        request = TransactionRequest(
            transaction_id=command.transaction_id,
            idempotency_key=command.idempotency_key,
            scope=command.scope,
            target_ref=command.relative_path,
            prior_state_ref=prior_state_ref,
            delta_ref=f"sha256:{prepared.digest}",
            resource_profile=command.resource_profile,
        )
        operation = _Operation(command, prepared, intent, request, root, authorization)
        with self._lock:
            if command.transaction_id in self._operations:
                raise WorkspaceWriteError("Transaction is already active in this workspace writer")
            self._operations[command.transaction_id] = operation
        try:
            result = self._coordinator.execute(request)
        except (UnresolvedTransactionError, IndeterminateTransactionError):
            raise
        except Exception:
            self._finalize(command.transaction_id)
            raise
        self._finalize(command.transaction_id)
        return result

    def authorize(self, request: TransactionRequest) -> None:
        operation = self._operation_for(request)
        self._authorization_verifier.authorize(operation.intent, operation.authorization)

    def stage(self, request: TransactionRequest) -> StagedOperation:
        operation = self._operation_for(request)
        parts = normalize_workspace_relative_path(operation.command.relative_path)
        parent_fd = self._open_parent(operation.root, parts[:-1])
        target_name = parts[-1]
        lock_name, stage_name, backup_name = self._transaction_names(
            request.transaction_id,
            operation.command.relative_path,
        )
        lock_fd = -1
        try:
            lock_fd = self._acquire_lock(parent_fd, lock_name)
            previous_digest = self._digest_at(parent_fd, target_name)
            if previous_digest != operation.command.expected_current_sha256:
                raise WorkspaceWriteError("Target changed since the signed write confirmation")
            stage_fd = os.open(
                stage_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=parent_fd,
            )
            try:
                self._write_all(stage_fd, operation.prepared.content)
                os.fsync(stage_fd)
            finally:
                os.close(stage_fd)
            self._staged[request.transaction_id] = _StagedFile(
                parent_fd,
                lock_fd,
                target_name,
                stage_name,
                backup_name,
                previous_digest,
                operation.prepared.digest,
            )
            return StagedOperation(stage_ref=request.transaction_id)
        except Exception:
            if lock_fd >= 0:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    os.close(lock_fd)
                except OSError:
                    pass
            if stage_name:
                try:
                    os.unlink(stage_name, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass
            os.close(parent_fd)
            raise

    def commit(self, request: TransactionRequest, stage: StagedOperation) -> CommitOutcome:
        staged = self._staged_for(request, stage)
        if self._digest_at(staged.parent_fd, staged.target_name) != staged.previous_digest:
            raise WorkspaceWriteError("Target changed after staging; refusing to overwrite it")
        try:
            if staged.previous_digest is not None:
                os.link(
                    staged.target_name,
                    staged.backup_name,
                    src_dir_fd=staged.parent_fd,
                    dst_dir_fd=staged.parent_fd,
                    follow_symlinks=False,
                )
                os.fsync(staged.parent_fd)
            os.replace(
                staged.stage_name,
                staged.target_name,
                src_dir_fd=staged.parent_fd,
                dst_dir_fd=staged.parent_fd,
            )
            staged.committed = True
            os.fsync(staged.parent_fd)
        except OSError as error:
            raise WorkspaceWriteError("Atomic workspace commit failed") from error
        if self._digest_at(staged.parent_fd, staged.target_name) != staged.new_digest:
            raise WorkspaceWriteError("Committed workspace content failed digest verification")
        return CommitOutcome(verification_ref=f"sha256:{staged.new_digest}")

    def compensate(
        self,
        request: TransactionRequest,
        stage: StagedOperation | None,
    ) -> CompensationOutcome:
        staged = self._staged.get(request.transaction_id)
        if staged is None:
            return CompensationOutcome(True, "no-workspace-stage")
        try:
            if staged.committed:
                if self._digest_at(staged.parent_fd, staged.target_name) != staged.new_digest:
                    return CompensationOutcome(False, "target-changed-after-commit")
                if staged.previous_digest is None:
                    os.unlink(staged.target_name, dir_fd=staged.parent_fd)
                else:
                    if self._digest_at(staged.parent_fd, staged.backup_name) != staged.previous_digest:
                        return CompensationOutcome(False, "verified-backup-unavailable")
                    os.replace(
                        staged.backup_name,
                        staged.target_name,
                        src_dir_fd=staged.parent_fd,
                        dst_dir_fd=staged.parent_fd,
                    )
                os.fsync(staged.parent_fd)
                if self._digest_at(staged.parent_fd, staged.target_name) != staged.previous_digest:
                    return CompensationOutcome(False, "restored-state-digest-mismatch")
            else:
                self._unlink_if_present(staged.parent_fd, staged.stage_name)
                self._unlink_if_present(staged.parent_fd, staged.backup_name)
                os.fsync(staged.parent_fd)
            staged.compensated = True
            return CompensationOutcome(True, f"workspace-restored:{request.transaction_id}")
        except OSError as error:
            return CompensationOutcome(False, f"workspace-compensation-error:{type(error).__name__}")

    def _operation_for(self, request: TransactionRequest) -> _Operation:
        with self._lock:
            operation = self._operations.get(request.transaction_id)
        if operation is None or operation.request != request:
            raise WorkspaceWriteError("Transaction has no matching authorized workspace operation")
        return operation

    def _staged_for(self, request: TransactionRequest, stage: StagedOperation) -> _StagedFile:
        if not isinstance(stage, StagedOperation) or stage.stage_ref != request.transaction_id:
            raise WorkspaceWriteError("Staged operation does not match the transaction")
        try:
            return self._staged[request.transaction_id]
        except KeyError as error:
            raise WorkspaceWriteError("Staged workspace data is unavailable") from error

    def _finalize(self, transaction_id: str) -> None:
        staged = self._staged.get(transaction_id)
        if staged is not None:
            try:
                self._unlink_if_present(staged.parent_fd, staged.stage_name)
                self._unlink_if_present(staged.parent_fd, staged.backup_name)
                os.fsync(staged.parent_fd)
            except OSError as error:
                raise WorkspaceCleanupError("Workspace transaction files could not be confirmed released") from error
            finally:
                fcntl.flock(staged.lock_fd, fcntl.LOCK_UN)
                os.close(staged.lock_fd)
                os.close(staged.parent_fd)
            self._staged.pop(transaction_id, None)
        with self._lock:
            self._operations.pop(transaction_id, None)

    def close(self) -> None:
        """Release process locks while preserving unresolved stage/backup files."""
        for staged in tuple(self._staged.values()):
            fcntl.flock(staged.lock_fd, fcntl.LOCK_UN)
            os.close(staged.lock_fd)
            os.close(staged.parent_fd)
        self._staged.clear()
        with self._lock:
            self._operations.clear()

    def reconcile_unresolved(self) -> tuple[tuple[str, ReceiptTransition], ...]:
        results: list[tuple[str, ReceiptTransition]] = []
        for transaction in self._receipt_ledger.unresolved_transactions():
            results.append((transaction.transaction_id, self._reconcile_transaction(transaction)))
        return tuple(results)

    def _reconcile_transaction(self, transaction: ReceiptTransaction) -> ReceiptTransition:
        prepared = transaction.events[0]
        prior_digest = self._digest_reference(prepared.prior_state_ref, allow_absent=True)
        new_digest = self._digest_reference(prepared.delta_ref, allow_absent=False)
        if prior_digest is _INVALID_DIGEST or new_digest is _INVALID_DIGEST:
            return self._append_recovery_terminal(
                transaction.transaction_id,
                ReceiptTransition.INDETERMINATE,
                "recovery-invalid-receipt-reference",
            )
        try:
            parts = normalize_workspace_relative_path(prepared.target_ref)
            root = self._workspace_resolver.resolve(prepared.scope)
            parent_fd = self._open_parent(root, parts[:-1])
        except (OSError, WorkspaceWriteError, ValueError):
            return self._append_recovery_terminal(
                transaction.transaction_id,
                ReceiptTransition.INDETERMINATE,
                "recovery-workspace-unavailable",
            )
        target_name = parts[-1]
        lock_name, stage_name, backup_name = self._transaction_names(
            transaction.transaction_id,
            prepared.target_ref,
        )
        lock_fd = -1
        terminal_written = False
        try:
            lock_fd = self._acquire_lock(parent_fd, lock_name)
            current_digest = self._digest_at(parent_fd, target_name)
            stage_digest = self._digest_at(parent_fd, stage_name)
            backup_digest = self._digest_at(parent_fd, backup_name)
            if current_digest == prior_digest:
                safe_stage = stage_digest in (None, new_digest)
                safe_backup = backup_digest is None or backup_digest == prior_digest
                if safe_stage and safe_backup:
                    transition = self._append_recovery_terminal(
                        transaction.transaction_id,
                        ReceiptTransition.COMPENSATED,
                        "recovery-target-unchanged",
                    )
                    terminal_written = True
                    self._unlink_if_present(parent_fd, stage_name)
                    self._unlink_if_present(parent_fd, backup_name)
                    os.fsync(parent_fd)
                    return transition
            if current_digest == new_digest and prior_digest is not None and backup_digest == prior_digest:
                os.replace(
                    backup_name,
                    target_name,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                )
                os.fsync(parent_fd)
                if self._digest_at(parent_fd, target_name) == prior_digest:
                    transition = self._append_recovery_terminal(
                        transaction.transaction_id,
                        ReceiptTransition.COMPENSATED,
                        "recovery-restored-verified-backup",
                    )
                    terminal_written = True
                    self._unlink_if_present(parent_fd, stage_name)
                    os.fsync(parent_fd)
                    return transition
            return self._append_recovery_terminal(
                transaction.transaction_id,
                ReceiptTransition.INDETERMINATE,
                "recovery-target-state-uncertain",
            )
        except (OSError, WorkspaceWriteError):
            if terminal_written:
                raise WorkspaceCleanupError("Recovered transaction is terminal but cleanup remains uncertain")
            return self._append_recovery_terminal(
                transaction.transaction_id,
                ReceiptTransition.INDETERMINATE,
                "recovery-filesystem-error",
            )
        finally:
            if lock_fd >= 0:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)
            os.close(parent_fd)

    def _append_recovery_terminal(
        self,
        transaction_id: str,
        transition: ReceiptTransition,
        verification_ref: str,
    ) -> ReceiptTransition:
        return self._receipt_ledger.append_terminal(
            transaction_id,
            transition,
            verification_ref,
        ).state

    @staticmethod
    def _open_parent(root: Path, parent_parts: tuple[str, ...]) -> int:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        current_fd = os.open(root, flags | getattr(os, "O_CLOEXEC", 0))
        try:
            if not stat.S_ISDIR(os.fstat(current_fd).st_mode):
                raise WorkspacePathRejected("Configured workspace root is not a directory")
            for part in parent_parts:
                next_fd = os.open(part, flags | getattr(os, "O_CLOEXEC", 0), dir_fd=current_fd)
                os.close(current_fd)
                current_fd = next_fd
            return current_fd
        except Exception:
            os.close(current_fd)
            raise

    @staticmethod
    def _digest_at(directory_fd: int, name: str) -> str | None:
        try:
            descriptor = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
                dir_fd=directory_fd,
            )
        except FileNotFoundError:
            return None
        except OSError as error:
            if error.errno == errno.ELOOP:
                raise WorkspacePathRejected("Workspace target must not be a symbolic link") from error
            raise
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise WorkspacePathRejected("Workspace target must be a regular file")
            digest = hashlib.sha256()
            while chunk := os.read(descriptor, 1024 * 1024):
                digest.update(chunk)
            return digest.hexdigest()
        finally:
            os.close(descriptor)

    @staticmethod
    def _write_all(descriptor: int, content: bytes) -> None:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("Workspace stage write made no progress")
            view = view[written:]

    @staticmethod
    def _unlink_if_present(directory_fd: int, name: str) -> None:
        try:
            os.unlink(name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass

    @staticmethod
    def _transaction_names(transaction_id: str, relative_path: str) -> tuple[str, str, str]:
        transaction_digest = hashlib.sha256(transaction_id.encode("utf-8")).hexdigest()
        path_digest = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()
        return (
            f".stacey-lock-{path_digest}",
            f".stacey-stage-{transaction_digest}",
            f".stacey-backup-{transaction_digest}",
        )

    @staticmethod
    def _acquire_lock(directory_fd: int, lock_name: str) -> int:
        descriptor = os.open(
            lock_name,
            os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=directory_fd,
        )
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise WorkspacePathRejected("Workspace transaction lock is not a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        return descriptor

    @staticmethod
    def _digest_reference(value: str, *, allow_absent: bool) -> str | None | object:
        if allow_absent and value == "ABSENT":
            return None
        if not value.startswith("sha256:"):
            return _INVALID_DIGEST
        digest = value.removeprefix("sha256:")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            return _INVALID_DIGEST
        return digest


_INVALID_DIGEST = object()