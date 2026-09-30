from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from enum import Enum
from threading import RLock


class ModelCatalogError(ValueError):
    pass


class CandidateTransitionError(ModelCatalogError):
    pass


class ModelRole(str, Enum):
    WORLD_MODEL_CORE = "WORLD_MODEL_CORE"
    SPECIALIST = "SPECIALIST"


class CandidateStatus(str, Enum):
    PROPOSED = "PROPOSED"
    EVALUATED = "EVALUATED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class TrainingDataAuthorization(str, Enum):
    APPROVED_DATASET = "APPROVED_DATASET"
    EXPLICIT_OPT_IN = "EXPLICIT_OPT_IN"


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelCatalogError(f"{field_name} must be a non-empty string")
    return value


def _validate_text_tuple(values: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(values, tuple) or not values:
        raise ModelCatalogError(f"{field_name} must be a non-empty tuple")
    validated = tuple(_required_text(value, field_name) for value in values)
    if len(set(validated)) != len(validated):
        raise ModelCatalogError(f"{field_name} must not contain duplicates")
    return validated


@dataclass(frozen=True, slots=True)
class TrainingSource:
    source_id: str
    authorization: TrainingDataAuthorization
    authorization_reference: str

    def __post_init__(self) -> None:
        _required_text(self.source_id, "source_id")
        _required_text(self.authorization_reference, "authorization_reference")
        if not isinstance(self.authorization, TrainingDataAuthorization):
            raise ModelCatalogError("Training source requires an approved-data authorization type")


@dataclass(frozen=True, slots=True)
class EvaluationEvidence:
    suite_id: str
    run_reference: str
    artifact_sha256: str
    baseline_reference: str
    passed: bool
    metrics: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        for field_name in ("suite_id", "run_reference", "baseline_reference"):
            _required_text(getattr(self, field_name), field_name)
        if not isinstance(self.artifact_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.artifact_sha256
        ):
            raise ModelCatalogError("Evaluation evidence must bind to a lowercase SHA-256 artifact digest")
        if not isinstance(self.passed, bool):
            raise ModelCatalogError("passed must be a boolean")
        if not isinstance(self.metrics, tuple):
            raise ModelCatalogError("metrics must be a tuple of named numeric measurements")
        metric_names: set[str] = set()
        for metric in self.metrics:
            if not isinstance(metric, tuple) or len(metric) != 2:
                raise ModelCatalogError("each metric must be a (name, value) tuple")
            name, value = metric
            _required_text(name, "metric name")
            if name in metric_names:
                raise ModelCatalogError("metric names must be unique")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ModelCatalogError("metric values must be finite numbers")
            metric_names.add(name)


@dataclass(frozen=True, slots=True)
class ModelCandidate:
    candidate_id: str
    version: str
    role: ModelRole
    capability_ids: tuple[str, ...]
    artifact_sha256: str
    artifact_size_bytes: int
    parameter_count: int
    training_sources: tuple[TrainingSource, ...]
    teacher_candidate_ids: tuple[str, ...] = ()
    status: CandidateStatus = CandidateStatus.PROPOSED
    evaluation: EvaluationEvidence | None = None
    approval_reference: str | None = None

    def __post_init__(self) -> None:
        _required_text(self.candidate_id, "candidate_id")
        _required_text(self.version, "version")
        if not isinstance(self.role, ModelRole):
            raise ModelCatalogError("role must be a ModelRole")
        _validate_text_tuple(self.capability_ids, "capability_ids")
        if not isinstance(self.artifact_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.artifact_sha256
        ):
            raise ModelCatalogError("artifact_sha256 must be a lowercase SHA-256 digest")
        for field_name in ("artifact_size_bytes", "parameter_count"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ModelCatalogError(f"{field_name} must be a positive integer")
        if not isinstance(self.training_sources, tuple) or not self.training_sources:
            raise ModelCatalogError("candidate requires approved or explicitly opted-in training sources")
        if any(not isinstance(source, TrainingSource) for source in self.training_sources):
            raise ModelCatalogError("training_sources must contain TrainingSource records")
        if self.teacher_candidate_ids:
            _validate_text_tuple(self.teacher_candidate_ids, "teacher_candidate_ids")
        if not isinstance(self.status, CandidateStatus):
            raise ModelCatalogError("status must be a CandidateStatus")

        if self.status is CandidateStatus.PROPOSED:
            if self.evaluation is not None or self.approval_reference is not None:
                raise ModelCatalogError("proposed candidates cannot contain evaluation or approval")
        elif self.status is CandidateStatus.EVALUATED:
            if self.evaluation is None or not self.evaluation.passed or self.approval_reference is not None:
                raise ModelCatalogError("evaluated candidates require a passing evaluation and no approval")
        elif self.status is CandidateStatus.REJECTED:
            if self.evaluation is None or self.evaluation.passed or self.approval_reference is not None:
                raise ModelCatalogError("rejected candidates require a failing evaluation and no approval")
        elif self.status is CandidateStatus.APPROVED:
            if self.evaluation is None or not self.evaluation.passed:
                raise ModelCatalogError("approved candidates require a passing evaluation")
            _required_text(self.approval_reference, "approval_reference")


class ModelCandidateCatalog:
    """In-memory governance registry; it never discovers or loads local model files."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._candidates: dict[str, ModelCandidate] = {}

    def propose(self, candidate: ModelCandidate) -> ModelCandidate:
        if not isinstance(candidate, ModelCandidate) or candidate.status is not CandidateStatus.PROPOSED:
            raise ModelCatalogError("Only a proposed ModelCandidate may be registered")
        with self._lock:
            if candidate.candidate_id in self._candidates:
                raise ModelCatalogError("candidate_id is already registered")
            self._candidates[candidate.candidate_id] = candidate
            return candidate

    def record_evaluation(
        self,
        candidate_id: str,
        evidence: EvaluationEvidence,
    ) -> ModelCandidate:
        _required_text(candidate_id, "candidate_id")
        if not isinstance(evidence, EvaluationEvidence):
            raise ModelCatalogError("evidence must be EvaluationEvidence")
        with self._lock:
            candidate = self._require_candidate(candidate_id)
            if candidate.status is not CandidateStatus.PROPOSED:
                raise CandidateTransitionError("Only proposed candidates can receive their first evaluation")
            if evidence.artifact_sha256 != candidate.artifact_sha256:
                raise ModelCatalogError("Evaluation evidence does not match the candidate artifact")
            status = CandidateStatus.EVALUATED if evidence.passed else CandidateStatus.REJECTED
            updated = replace(candidate, status=status, evaluation=evidence)
            self._candidates[candidate_id] = updated
            return updated

    def approve(self, candidate_id: str, approval_reference: str) -> ModelCandidate:
        _required_text(candidate_id, "candidate_id")
        _required_text(approval_reference, "approval_reference")
        with self._lock:
            candidate = self._require_candidate(candidate_id)
            if candidate.status is not CandidateStatus.EVALUATED:
                raise CandidateTransitionError("Only a passing evaluated candidate can be approved")
            updated = replace(
                candidate,
                status=CandidateStatus.APPROVED,
                approval_reference=approval_reference,
            )
            self._candidates[candidate_id] = updated
            return updated

    def get(self, candidate_id: str) -> ModelCandidate | None:
        with self._lock:
            return self._candidates.get(candidate_id)

    def approved_for(self, capability_id: str) -> tuple[ModelCandidate, ...]:
        _required_text(capability_id, "capability_id")
        with self._lock:
            return tuple(
                candidate
                for candidate in self._candidates.values()
                if candidate.status is CandidateStatus.APPROVED
                and capability_id in candidate.capability_ids
            )

    def _require_candidate(self, candidate_id: str) -> ModelCandidate:
        try:
            return self._candidates[candidate_id]
        except KeyError as error:
            raise ModelCatalogError(f"Unknown candidate_id: {candidate_id}") from error