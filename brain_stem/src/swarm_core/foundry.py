from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import BoundedSemaphore, RLock
from time import time_ns
from typing import Callable, Protocol

from models.stacey.core.config import StaceyCoreConfig
from src.swarm_core.core_evaluation import EvaluationEvidence
from src.swarm_core.hardware_profile import HardwareProfile
from src.swarm_core.model_catalog import (
    CandidateStatus,
    ModelCandidate,
    ModelCandidateCatalog,
    ModelRole,
)
from src.swarm_core.training_governance import (
    GovernanceApprovalJournal,
    GovernanceApprovalVerifier,
    SignedGovernanceApproval,
)
from src.swarm_core.training_readiness import CoreTrainingReadinessChecker, CoreTrainingRunPlan, TrainingMethod
from training.stacey.corpus import ApprovedStaceyDataset
from training.stacey.trainer import (
    StaceyTrainerConfig,
    StaceyTrainingRunReport,
    train_stacey_from_scratch,
    training_config_sha256,
)


class FoundryError(RuntimeError):
    pass


class FoundryJobConflict(FoundryError):
    pass


class FoundryCapacityDenied(FoundryError):
    pass


class FoundryJobState(str, Enum):
    SUBMITTED = "SUBMITTED"
    RUNNING = "RUNNING"
    CANDIDATE_PROPOSED = "CANDIDATE_PROPOSED"
    EVALUATED = "EVALUATED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


_TERMINAL_STATES = frozenset(
    {FoundryJobState.EVALUATED, FoundryJobState.REJECTED, FoundryJobState.FAILED}
)
_ALLOWED_TRANSITIONS = {
    FoundryJobState.SUBMITTED: frozenset({FoundryJobState.RUNNING, FoundryJobState.FAILED}),
    FoundryJobState.RUNNING: frozenset(
        {FoundryJobState.CANDIDATE_PROPOSED, FoundryJobState.FAILED}
    ),
    FoundryJobState.CANDIDATE_PROPOSED: frozenset(
        {FoundryJobState.EVALUATED, FoundryJobState.REJECTED, FoundryJobState.FAILED}
    ),
}


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _digest(value: object, field: str, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
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
        raise ValueError("Foundry event must contain finite JSON-compatible values") from error


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class FoundryJobEvent:
    sequence: int
    job_id: str
    job_sha256: str
    state: FoundryJobState
    candidate_id: str
    plan_sha256: str
    dataset_sha256: str
    artifact_sha256: str | None
    evidence_reference: str | None
    error_type: str | None
    occurred_at_ns: int
    previous_hash: str | None
    event_hash: str


class FoundryJobJournal:
    """Append-only, hash-chained job state journal; it contains no approval operation."""

    _PAYLOAD_FIELDS = frozenset(
        {
            "job_id",
            "job_sha256",
            "state",
            "candidate_id",
            "plan_sha256",
            "dataset_sha256",
            "artifact_sha256",
            "evidence_reference",
            "error_type",
            "occurred_at_ns",
        }
    )

    _PAYLOAD_FIELDS = frozenset(
        {
            "job_id",
            "job_sha256",
            "state",
            "candidate_id",
            "plan_sha256",
            "dataset_sha256",
            "artifact_sha256",
            "evidence_reference",
            "error_type",
            "occurred_at_ns",
        }
    )

    def __init__(
        self,
        database_path: str | os.PathLike[str],
        *,
        file_mode: int = 0o600,
        clock: Callable[[], int] = time_ns,
    ) -> None:
        path = Path(database_path)
        if path.is_symlink() or not path.name or not path.parent.is_dir():
            raise ValueError("database_path must be a non-symlink file in an existing directory")
        if not callable(clock):
            raise ValueError("clock must be callable")
        if isinstance(file_mode, bool) or not isinstance(file_mode, int) or file_mode < 0 or file_mode & ~0o777:
            raise ValueError("file_mode must contain only permission bits")
        self._path = path.resolve()
        self._clock = clock
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
        except FoundryError:
            connection = getattr(self, "_connection", None)
            if connection is not None:
                connection.close()
            raise
        except (OSError, sqlite3.Error, ValueError) as error:
            connection = getattr(self, "_connection", None)
            if connection is not None:
                connection.close()
            raise FoundryError("Foundry job journal could not be opened or verified") from error

    def _create_file(self, file_mode: int) -> None:
        flags = os.O_CREAT | os.O_EXCL | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self._path, flags, file_mode)
        except FileExistsError:
            if self._path.is_symlink() or not self._path.is_file():
                raise ValueError("database_path must identify a regular non-symlink file")
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
                raise FoundryError("Unversioned Foundry journal contains unexpected tables")
            self._connection.executescript(
                """
                CREATE TABLE foundry_events (
                    sequence INTEGER PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    previous_hash TEXT,
                    event_hash TEXT NOT NULL UNIQUE
                );
                CREATE INDEX foundry_events_job ON foundry_events (job_id, sequence);
                CREATE TRIGGER foundry_events_no_update BEFORE UPDATE ON foundry_events
                    BEGIN SELECT RAISE(ABORT, 'Foundry events are append-only'); END;
                CREATE TRIGGER foundry_events_no_delete BEFORE DELETE ON foundry_events
                    BEGIN SELECT RAISE(ABORT, 'Foundry events are append-only'); END;
                PRAGMA user_version=1;
                """
            )
        elif version != 1:
            raise FoundryError("Unsupported Foundry journal schema version")

    def append(
        self,
        *,
        job_id: str,
        job_sha256: str,
        state: FoundryJobState,
        candidate_id: str,
        plan_sha256: str,
        dataset_sha256: str,
        artifact_sha256: str | None = None,
        evidence_reference: str | None = None,
        error_type: str | None = None,
    ) -> FoundryJobEvent:
        _text(job_id, "job_id")
        _digest(job_sha256, "job_sha256")
        _text(candidate_id, "candidate_id")
        _digest(plan_sha256, "plan_sha256")
        _digest(dataset_sha256, "dataset_sha256")
        _digest(artifact_sha256, "artifact_sha256", optional=True)
        if not isinstance(state, FoundryJobState):
            raise ValueError("state must be a FoundryJobState")
        if evidence_reference is not None:
            _text(evidence_reference, "evidence_reference")
        if error_type is not None:
            _text(error_type, "error_type")
        if state in {FoundryJobState.EVALUATED, FoundryJobState.REJECTED}:
            if artifact_sha256 is None or evidence_reference is None or error_type is not None:
                raise ValueError("evaluation terminal states require artifact and evidence, not an error")
        elif state is FoundryJobState.FAILED:
            if error_type is None:
                raise ValueError("FAILED requires an error type")
        elif error_type is not None:
            raise ValueError("non-failed states cannot contain an error type")
        if state in {FoundryJobState.CANDIDATE_PROPOSED} and artifact_sha256 is None:
            raise ValueError("candidate proposal state requires an artifact digest")

        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                self._verify_integrity_locked()
                prior_row = self._connection.execute(
                    "SELECT sequence, job_id, payload_json, previous_hash, event_hash "
                    "FROM foundry_events WHERE job_id = ? ORDER BY sequence DESC LIMIT 1",
                    (job_id,),
                ).fetchone()
                prior = None if prior_row is None else self._decode_row(prior_row)
                self._validate_transition(
                    prior,
                    state,
                    job_sha256,
                    candidate_id,
                    plan_sha256,
                    dataset_sha256,
                    artifact_sha256,
                )
                occurred_at_ns = self._clock()
                if (
                    isinstance(occurred_at_ns, bool)
                    or not isinstance(occurred_at_ns, int)
                    or not 0 <= occurred_at_ns <= sys.maxsize
                ):
                    raise ValueError("clock must return a supported non-negative nanosecond timestamp")
                last = self._connection.execute(
                    "SELECT sequence, event_hash FROM foundry_events ORDER BY sequence DESC LIMIT 1"
                ).fetchone()
                sequence = 1 if last is None else last[0] + 1
                previous_hash = None if last is None else last[1]
                payload = {
                    "job_id": job_id,
                    "job_sha256": job_sha256,
                    "state": state.value,
                    "candidate_id": candidate_id,
                    "plan_sha256": plan_sha256,
                    "dataset_sha256": dataset_sha256,
                    "artifact_sha256": artifact_sha256,
                    "evidence_reference": evidence_reference,
                    "error_type": error_type,
                    "occurred_at_ns": occurred_at_ns,
                }
                payload_json = _canonical_json(payload)
                event_hash = self._event_hash(sequence, payload, previous_hash)
                self._connection.execute(
                    "INSERT INTO foundry_events (sequence, job_id, payload_json, previous_hash, event_hash) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (sequence, job_id, payload_json, previous_hash, event_hash),
                )
                self._connection.execute("COMMIT")
                return self._decode_row((sequence, job_id, payload_json, previous_hash, event_hash))
            except FoundryError:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
            except (OSError, sqlite3.Error, TypeError, ValueError) as error:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise FoundryError("Foundry job event could not be durably appended") from error
            except Exception:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise

    def events(self, job_id: str) -> tuple[FoundryJobEvent, ...]:
        _text(job_id, "job_id")
        with self._lock:
            self._verify_integrity_locked()
            rows = self._connection.execute(
                "SELECT sequence, job_id, payload_json, previous_hash, event_hash "
                "FROM foundry_events WHERE job_id = ? ORDER BY sequence",
                (job_id,),
            ).fetchall()
            return tuple(self._decode_row(row) for row in rows)

    def verify_integrity(self) -> None:
        with self._lock:
            self._verify_integrity_locked()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> FoundryJobJournal:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def _verify_integrity_locked(self) -> None:
        previous_hash: str | None = None
        latest_by_job: dict[str, FoundryJobEvent] = {}
        rows = self._connection.execute(
            "SELECT sequence, job_id, payload_json, previous_hash, event_hash "
            "FROM foundry_events ORDER BY sequence"
        ).fetchall()
        for expected_sequence, row in enumerate(rows, start=1):
            event = self._decode_row(row, expected_sequence, previous_hash)
            previous = latest_by_job.get(event.job_id)
            self._validate_transition(
                previous,
                event.state,
                event.job_sha256,
                event.candidate_id,
                event.plan_sha256,
                event.dataset_sha256,
                event.artifact_sha256,
                verifying=True,
            )
            latest_by_job[event.job_id] = event
            previous_hash = event.event_hash

    @staticmethod
    def _validate_transition(
        prior: FoundryJobEvent | None,
        state: FoundryJobState,
        job_sha256: str,
        candidate_id: str,
        plan_sha256: str,
        dataset_sha256: str,
        artifact_sha256: str | None,
        *,
        verifying: bool = False,
    ) -> None:
        error_type = FoundryError if verifying else FoundryJobConflict
        if prior is None:
            if state is not FoundryJobState.SUBMITTED:
                raise error_type("A Foundry job must begin in SUBMITTED state")
            return
        if prior.state in _TERMINAL_STATES or state not in _ALLOWED_TRANSITIONS.get(prior.state, frozenset()):
            raise error_type("Foundry job state transition is invalid")
        if (
            prior.job_sha256 != job_sha256
            or prior.candidate_id != candidate_id
            or prior.plan_sha256 != plan_sha256
            or prior.dataset_sha256 != dataset_sha256
        ):
            raise error_type("Foundry job identity or input digest changed between events")
        if prior.artifact_sha256 is not None and artifact_sha256 != prior.artifact_sha256:
            raise error_type("Foundry job artifact digest changed between events")

    @staticmethod
    def _event_hash(sequence: int, payload: dict[str, object], previous_hash: str | None) -> str:
        body = {"sequence": sequence, "payload": payload, "previous_hash": previous_hash}
        return hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()

    def _decode_row(
        self,
        row: tuple[object, ...],
        expected_sequence: int | None = None,
        expected_previous_hash: str | None = None,
    ) -> FoundryJobEvent:
        sequence, job_id_column, payload_json, previous_hash, event_hash = row
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= 0:
            raise FoundryError("Foundry sequence is invalid")
        if expected_sequence is not None and sequence != expected_sequence:
            raise FoundryError("Foundry sequence is not contiguous")
        if expected_sequence is not None and previous_hash != expected_previous_hash:
            raise FoundryError("Foundry hash chain is broken")
        try:
            payload = json.loads(payload_json, object_pairs_hook=_unique_object)
        except (TypeError, json.JSONDecodeError, ValueError) as error:
            raise FoundryError("Foundry event payload is malformed") from error
        if not isinstance(payload, dict) or set(payload) != self._PAYLOAD_FIELDS:
            raise FoundryError("Foundry event has an unexpected shape")
        if payload["job_id"] != job_id_column:
            raise FoundryError("Foundry indexed job ID does not match its payload")
        if self._event_hash(sequence, payload, previous_hash) != event_hash:
            raise FoundryError("Foundry event hash verification failed")
        try:
            event = FoundryJobEvent(
                sequence=sequence,
                job_id=_text(payload["job_id"], "job_id"),
                job_sha256=_digest(payload["job_sha256"], "job_sha256"),
                state=FoundryJobState(payload["state"]),
                candidate_id=_text(payload["candidate_id"], "candidate_id"),
                plan_sha256=_digest(payload["plan_sha256"], "plan_sha256"),
                dataset_sha256=_digest(payload["dataset_sha256"], "dataset_sha256"),
                artifact_sha256=_digest(payload["artifact_sha256"], "artifact_sha256", optional=True),
                evidence_reference=(
                    None
                    if payload["evidence_reference"] is None
                    else _text(payload["evidence_reference"], "evidence_reference")
                ),
                error_type=(None if payload["error_type"] is None else _text(payload["error_type"], "error_type")),
                occurred_at_ns=payload["occurred_at_ns"],
                previous_hash=previous_hash,
                event_hash=event_hash,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise FoundryError("Foundry event fields are invalid") from error
        if (
            isinstance(event.occurred_at_ns, bool)
            or not isinstance(event.occurred_at_ns, int)
            or not 0 <= event.occurred_at_ns <= sys.maxsize
        ):
            raise FoundryError("Foundry timestamp is invalid")
        return event


@dataclass(frozen=True, slots=True)
class FoundryPolicy:
    maximum_steps: int
    maximum_wall_time_seconds: int
    maximum_parameter_count: int
    maximum_training_examples: int
    maximum_validation_examples: int
    maximum_artifact_size_bytes: int
    maximum_concurrent_jobs: int
    protected_evaluation_suite_id: str

    def __post_init__(self) -> None:
        for field in (
            "maximum_steps",
            "maximum_wall_time_seconds",
            "maximum_parameter_count",
            "maximum_training_examples",
            "maximum_validation_examples",
            "maximum_artifact_size_bytes",
            "maximum_concurrent_jobs",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        _text(self.protected_evaluation_suite_id, "protected_evaluation_suite_id")


@dataclass(frozen=True, slots=True)
class FoundryCandidateJob:
    job_id: str
    plan: CoreTrainingRunPlan
    dataset: ApprovedStaceyDataset
    hardware_profile: HardwareProfile
    model_config: StaceyCoreConfig
    trainer_config: StaceyTrainerConfig
    run_approval: SignedGovernanceApproval

    def __post_init__(self) -> None:
        _text(self.job_id, "job_id")
        if not isinstance(self.plan, CoreTrainingRunPlan):
            raise ValueError("plan must be a CoreTrainingRunPlan")
        if not isinstance(self.dataset, ApprovedStaceyDataset):
            raise ValueError("dataset must be an ApprovedStaceyDataset")
        if self.plan.dataset != self.dataset.manifest:
            raise ValueError("training plan does not bind the supplied approved dataset")
        if not isinstance(self.hardware_profile, HardwareProfile):
            raise ValueError("hardware_profile must be a measured HardwareProfile")
        if not isinstance(self.model_config, StaceyCoreConfig):
            raise ValueError("model_config must be a StaceyCoreConfig")
        if not isinstance(self.trainer_config, StaceyTrainerConfig):
            raise ValueError("trainer_config must be a StaceyTrainerConfig")
        if not isinstance(self.run_approval, SignedGovernanceApproval):
            raise ValueError("run_approval must be a SignedGovernanceApproval")


class FoundryEvaluator(Protocol):
    def evaluate(
        self,
        *,
        candidate: ModelCandidate,
        checkpoint_path: Path,
        training_report: StaceyTrainingRunReport,
    ) -> EvaluationEvidence: ...


@dataclass(frozen=True, slots=True)
class FoundryExecutionResult:
    training_report: StaceyTrainingRunReport
    candidate: ModelCandidate
    final_event: FoundryJobEvent


class StaceyFoundry:
    """Bounded scratch-Core builder; it can propose/evaluate but never approve or activate."""

    def __init__(
        self,
        *,
        artifact_root: str | os.PathLike[str],
        journal: FoundryJobJournal,
        candidate_catalog: ModelCandidateCatalog,
        approval_verifier: GovernanceApprovalVerifier,
        approval_journal: GovernanceApprovalJournal,
        evaluator: FoundryEvaluator,
        policy: FoundryPolicy,
        checkpoint_file_mode: int = 0o600,
        training_runner: Callable[..., StaceyTrainingRunReport] = train_stacey_from_scratch,
    ) -> None:
        root = Path(artifact_root)
        if root.is_symlink() or not root.is_dir():
            raise ValueError("artifact_root must be an existing non-symlink directory")
        if not isinstance(journal, FoundryJobJournal):
            raise ValueError("journal must be a FoundryJobJournal")
        if not isinstance(candidate_catalog, ModelCandidateCatalog):
            raise ValueError("candidate_catalog must be a ModelCandidateCatalog")
        if not callable(getattr(approval_verifier, "verify", None)):
            raise ValueError("approval_verifier must implement verify()")
        if not callable(getattr(approval_journal, "consume", None)):
            raise ValueError("approval_journal must implement consume()")
        if not callable(getattr(evaluator, "evaluate", None)) or not callable(training_runner):
            raise ValueError("evaluator and training_runner must be callable services")
        if not isinstance(policy, FoundryPolicy):
            raise ValueError("policy must be a FoundryPolicy")
        if isinstance(checkpoint_file_mode, bool) or not isinstance(checkpoint_file_mode, int):
            raise ValueError("checkpoint_file_mode must be an integer permission mode")
        allowed_file_mode = stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO
        if checkpoint_file_mode < 0 or checkpoint_file_mode & ~allowed_file_mode:
            raise ValueError("checkpoint_file_mode contains unsupported permission bits")
        self._artifact_root = root.resolve(strict=True)
        self._journal = journal
        self._catalog = candidate_catalog
        self._approval_verifier = approval_verifier
        self._approval_journal = approval_journal
        self._evaluator = evaluator
        self._policy = policy
        self._checkpoint_file_mode = checkpoint_file_mode
        self._training_runner = training_runner
        self._capacity = BoundedSemaphore(policy.maximum_concurrent_jobs)

    def execute(self, job: FoundryCandidateJob) -> FoundryExecutionResult:
        if not isinstance(job, FoundryCandidateJob):
            raise TypeError("job must be a FoundryCandidateJob")
        self._validate_budget(job)
        job_digest = self._job_digest(job)
        plan_digest = CoreTrainingReadinessChecker.plan_digest(job.plan)
        dataset_digest = job.dataset.manifest.content_sha256
        destination = self._artifact_path(job.job_id, job.plan.candidate_id)
        if not self._capacity.acquire(blocking=False):
            raise FoundryCapacityDenied("Foundry concurrent-job capacity is exhausted")
        running_recorded = False
        artifact_sha256: str | None = None
        try:
            if destination.exists() or destination.is_symlink():
                raise FoundryJobConflict("Foundry artifact path already exists")
            self._journal.append(
                job_id=job.job_id,
                job_sha256=job_digest,
                state=FoundryJobState.SUBMITTED,
                candidate_id=job.plan.candidate_id,
                plan_sha256=plan_digest,
                dataset_sha256=dataset_digest,
            )
            self._journal.append(
                job_id=job.job_id,
                job_sha256=job_digest,
                state=FoundryJobState.RUNNING,
                candidate_id=job.plan.candidate_id,
                plan_sha256=plan_digest,
                dataset_sha256=dataset_digest,
            )
            running_recorded = True
            training_report = self._training_runner(
                plan=job.plan,
                dataset=job.dataset,
                hardware_profile=job.hardware_profile,
                model_config=job.model_config,
                trainer_config=job.trainer_config,
                approval_verifier=self._approval_verifier,
                approval_journal=self._approval_journal,
                run_approval=job.run_approval,
                checkpoint_path=destination,
                checkpoint_file_mode=self._checkpoint_file_mode,
            )
            artifact_sha256 = self._verify_training_output(job, training_report, destination, plan_digest)
            candidate = ModelCandidate(
                candidate_id=job.plan.candidate_id,
                version=job.plan.candidate_version,
                role=ModelRole.WORLD_MODEL_CORE,
                capability_ids=job.plan.capability_ids,
                artifact_sha256=artifact_sha256,
                artifact_size_bytes=destination.stat().st_size,
                parameter_count=job.plan.parameter_count,
                training_sources=job.dataset.manifest.sources,
            )
            self._catalog.propose(candidate)
            self._journal.append(
                job_id=job.job_id,
                job_sha256=job_digest,
                state=FoundryJobState.CANDIDATE_PROPOSED,
                candidate_id=job.plan.candidate_id,
                plan_sha256=plan_digest,
                dataset_sha256=dataset_digest,
                artifact_sha256=artifact_sha256,
                evidence_reference=f"candidate-proposal:{job.plan.candidate_id}",
            )
            evidence = self._evaluator.evaluate(
                candidate=candidate,
                checkpoint_path=destination,
                training_report=training_report,
            )
            if not isinstance(evidence, EvaluationEvidence):
                raise FoundryError("evaluator returned untyped evidence")
            if evidence.artifact_sha256 != artifact_sha256:
                raise FoundryError("evaluation evidence does not bind the proposed artifact")
            if evidence.suite_id != self._policy.protected_evaluation_suite_id:
                raise FoundryError("evaluation evidence is not from the configured protected suite")
            evaluated = self._catalog.record_evaluation(job.plan.candidate_id, evidence)
            terminal_state = (
                FoundryJobState.EVALUATED
                if evaluated.status is CandidateStatus.EVALUATED
                else FoundryJobState.REJECTED
            )
            final_event = self._journal.append(
                job_id=job.job_id,
                job_sha256=job_digest,
                state=terminal_state,
                candidate_id=job.plan.candidate_id,
                plan_sha256=plan_digest,
                dataset_sha256=dataset_digest,
                artifact_sha256=artifact_sha256,
                evidence_reference=f"{evidence.suite_id}:{evidence.run_reference}",
            )
            return FoundryExecutionResult(training_report, evaluated, final_event)
        except Exception as error:
            if running_recorded:
                try:
                    self._journal.append(
                        job_id=job.job_id,
                        job_sha256=job_digest,
                        state=FoundryJobState.FAILED,
                        candidate_id=job.plan.candidate_id,
                        plan_sha256=plan_digest,
                        dataset_sha256=dataset_digest,
                        artifact_sha256=artifact_sha256,
                        error_type=type(error).__name__,
                    )
                except Exception as journal_error:
                    raise FoundryError("Foundry failed and could not persist its failure event") from journal_error
            raise
        finally:
            self._capacity.release()

    def _validate_budget(self, job: FoundryCandidateJob) -> None:
        profile = job.plan.execution_profile
        if job.plan.method is not TrainingMethod.FROM_SCRATCH:
            raise FoundryError("Stacey Foundry accepts only from-scratch Core jobs")
        if job.plan.architecture_reference != job.model_config.architecture_id:
            raise FoundryError("candidate architecture differs from the approved run plan")
        if job.plan.tokenizer_reference != job.model_config.tokenizer_id:
            raise FoundryError("candidate tokenizer differs from the approved run plan")
        if job.plan.training_config_sha256 != training_config_sha256(job.model_config, job.trainer_config):
            raise FoundryError("model/trainer config differs from the approved run digest")
        if job.plan.parameter_count > self._policy.maximum_parameter_count:
            raise FoundryCapacityDenied("candidate parameter count exceeds Foundry policy")
        if profile.maximum_steps > self._policy.maximum_steps:
            raise FoundryCapacityDenied("training step budget exceeds Foundry policy")
        if profile.maximum_wall_time_seconds > self._policy.maximum_wall_time_seconds:
            raise FoundryCapacityDenied("training wall-time budget exceeds Foundry policy")
        if len(job.dataset.training_examples) > self._policy.maximum_training_examples:
            raise FoundryCapacityDenied("training split exceeds Foundry policy")
        if len(job.dataset.validation_examples) > self._policy.maximum_validation_examples:
            raise FoundryCapacityDenied("validation split exceeds Foundry policy")

    def _job_digest(self, job: FoundryCandidateJob) -> str:
        payload = {
            "job_id": job.job_id,
            "plan_sha256": CoreTrainingReadinessChecker.plan_digest(job.plan),
            "dataset_manifest_sha256": job.dataset.manifest_sha256,
            "dataset_sha256": job.dataset.manifest.content_sha256,
            "hardware_profile": job.hardware_profile.to_payload(),
            "run_approval_id": job.run_approval.approval_id,
            "run_approval_signature_sha256": hashlib.sha256(job.run_approval.signature).hexdigest(),
            "artifact_store": str(self._artifact_root),
        }
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    def _artifact_path(self, job_id: str, candidate_id: str) -> Path:
        filename_digest = hashlib.sha256(f"{job_id}\0{candidate_id}".encode("utf-8")).hexdigest()
        return self._artifact_root / f"{filename_digest}.stacey.pt"

    def _verify_training_output(
        self,
        job: FoundryCandidateJob,
        report: StaceyTrainingRunReport,
        destination: Path,
        plan_digest: str,
    ) -> str:
        if not isinstance(report, StaceyTrainingRunReport):
            raise FoundryError("training runner returned an untyped report")
        if (
            report.candidate_id != job.plan.candidate_id
            or report.readiness_plan_sha256 != plan_digest
            or report.dataset_sha256 != job.dataset.manifest.content_sha256
        ):
            raise FoundryError("training report changed the candidate, plan, or dataset identity")
        if destination.is_symlink() or not destination.is_file():
            raise FoundryError("training runner did not produce a regular candidate artifact")
        if Path(report.checkpoint_path).resolve(strict=True) != destination.resolve(strict=True):
            raise FoundryError("training report path does not match the Foundry-owned artifact path")
        digest = hashlib.sha256()
        size = 0
        with destination.open("rb") as artifact:
            while chunk := artifact.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        if size <= 0 or size > self._policy.maximum_artifact_size_bytes:
            raise FoundryCapacityDenied("candidate artifact size is outside Foundry policy")
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != report.checkpoint_sha256:
            raise FoundryError("training report digest does not match the produced artifact")
        return actual_sha256