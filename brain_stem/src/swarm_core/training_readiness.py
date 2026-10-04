from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from .model_catalog import TrainingSource


class TrainingPlanError(ValueError):
    pass


class TrainingMethod(str, Enum):
    FROM_SCRATCH = "FROM_SCRATCH"
    DISTILLATION = "DISTILLATION"
    FINE_TUNING = "FINE_TUNING"


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrainingPlanError(f"{field_name} must be a non-empty string")
    return value


def _positive_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= sys.maxsize:
        raise TrainingPlanError(f"{field_name} must be a positive integer")
    return value


def _nonnegative_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= sys.maxsize:
        raise TrainingPlanError(f"{field_name} must be a non-negative integer")
    return value


def _string_tuple(values: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(values, tuple) or not values:
        raise TrainingPlanError(f"{field_name} must be a non-empty tuple")
    checked = tuple(_required_text(value, field_name) for value in values)
    if len(set(checked)) != len(checked):
        raise TrainingPlanError(f"{field_name} must not contain duplicates")
    return checked


def _sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise TrainingPlanError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class TrainingDatasetManifest:
    dataset_id: str
    version: str
    content_sha256: str
    provenance_reference: str
    license_review_reference: str
    training_record_ids: tuple[str, ...]
    validation_record_ids: tuple[str, ...]
    test_record_ids: tuple[str, ...]
    sources: tuple[TrainingSource, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "dataset_id",
            "version",
            "provenance_reference",
            "license_review_reference",
        ):
            _required_text(getattr(self, field_name), field_name)
        _sha256(self.content_sha256, "content_sha256")
        training = _string_tuple(self.training_record_ids, "training_record_ids")
        validation = _string_tuple(self.validation_record_ids, "validation_record_ids")
        test = _string_tuple(self.test_record_ids, "test_record_ids")
        partitions = (set(training), set(validation), set(test))
        if any(partitions[left] & partitions[right] for left in range(3) for right in range(left + 1, 3)):
            raise TrainingPlanError("training, validation, and test record IDs must be disjoint")
        if not isinstance(self.sources, tuple) or not self.sources:
            raise TrainingPlanError("dataset manifest requires authorized provenance sources")
        if any(not isinstance(source, TrainingSource) for source in self.sources):
            raise TrainingPlanError("dataset sources must be TrainingSource values")


@dataclass(frozen=True, slots=True)
class TrainingExecutionProfile:
    profile_id: str
    accelerator_profile_id: str
    required_device_memory_bytes: int
    maximum_steps: int
    maximum_wall_time_seconds: int
    checkpoint_interval_steps: int
    precision_profile_id: str

    def __post_init__(self) -> None:
        for field_name in ("profile_id", "accelerator_profile_id", "precision_profile_id"):
            _required_text(getattr(self, field_name), field_name)
        _nonnegative_integer(self.required_device_memory_bytes, "required_device_memory_bytes")
        for field_name in (
            "maximum_steps",
            "maximum_wall_time_seconds",
            "checkpoint_interval_steps",
        ):
            _positive_integer(getattr(self, field_name), field_name)


@dataclass(frozen=True, slots=True)
class CoreTrainingRunPlan:
    candidate_id: str
    candidate_version: str
    capability_ids: tuple[str, ...]
    method: TrainingMethod
    architecture_reference: str
    tokenizer_reference: str
    parameter_count: int
    context_length_tokens: int
    training_config_sha256: str
    random_seed: int
    dataset: TrainingDatasetManifest
    execution_profile: TrainingExecutionProfile
    run_approval_reference: str
    teacher_candidate_id: str | None = None
    starting_candidate_id: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "candidate_id",
            "candidate_version",
            "architecture_reference",
            "tokenizer_reference",
            "run_approval_reference",
        ):
            _required_text(getattr(self, field_name), field_name)
        _string_tuple(self.capability_ids, "capability_ids")
        if not isinstance(self.method, TrainingMethod):
            raise TrainingPlanError("method must be a TrainingMethod")
        _positive_integer(self.parameter_count, "parameter_count")
        _positive_integer(self.context_length_tokens, "context_length_tokens")
        _sha256(self.training_config_sha256, "training_config_sha256")
        _nonnegative_integer(self.random_seed, "random_seed")
        if not isinstance(self.dataset, TrainingDatasetManifest):
            raise TrainingPlanError("dataset must be a TrainingDatasetManifest")
        if not isinstance(self.execution_profile, TrainingExecutionProfile):
            raise TrainingPlanError("execution_profile must be a TrainingExecutionProfile")
        if self.method is TrainingMethod.FROM_SCRATCH:
            if self.teacher_candidate_id is not None or self.starting_candidate_id is not None:
                raise TrainingPlanError("from-scratch training cannot reference a teacher or starting candidate")
        elif self.method is TrainingMethod.DISTILLATION:
            _required_text(self.teacher_candidate_id, "teacher_candidate_id")
            if self.starting_candidate_id is not None:
                raise TrainingPlanError("distillation uses a teacher, not a fine-tuning starting candidate")
        elif self.method is TrainingMethod.FINE_TUNING:
            _required_text(self.starting_candidate_id, "starting_candidate_id")


@dataclass(frozen=True, slots=True)
class TrainingEnvironmentEvidence:
    backend_id: str
    backend_version: str
    execution_profile_id: str
    available_device_memory_bytes: int
    supported_methods: tuple[TrainingMethod, ...]
    accessible_dataset_ids: tuple[str, ...]
    available_teacher_candidate_ids: tuple[str, ...]
    artifact_store_reference: str
    approved_run_references: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "backend_id",
            "backend_version",
            "execution_profile_id",
            "artifact_store_reference",
        ):
            _required_text(getattr(self, field_name), field_name)
        _nonnegative_integer(self.available_device_memory_bytes, "available_device_memory_bytes")
        for field_name in (
            "supported_methods",
            "accessible_dataset_ids",
            "available_teacher_candidate_ids",
            "approved_run_references",
        ):
            values = getattr(self, field_name)
            if not isinstance(values, tuple):
                raise TrainingPlanError(f"{field_name} must be a tuple")
            if field_name == "supported_methods":
                if any(not isinstance(value, TrainingMethod) for value in values):
                    raise TrainingPlanError("supported_methods must contain TrainingMethod values")
            else:
                for value in values:
                    _required_text(value, field_name)
                if len(set(values)) != len(values):
                    raise TrainingPlanError(f"{field_name} must not contain duplicates")


@dataclass(frozen=True, slots=True)
class TrainingReadinessReport:
    candidate_id: str
    plan_sha256: str
    ready_to_start: bool
    blockers: tuple[str, ...]
    backend_id: str
    artifact_store_reference: str


class CoreTrainingReadinessChecker:
    """Checks declared training prerequisites; it does not execute a training job."""

    def assess(
        self,
        plan: CoreTrainingRunPlan,
        environment: TrainingEnvironmentEvidence,
    ) -> TrainingReadinessReport:
        if not isinstance(plan, CoreTrainingRunPlan):
            raise TrainingPlanError("plan must be a CoreTrainingRunPlan")
        if not isinstance(environment, TrainingEnvironmentEvidence):
            raise TrainingPlanError("environment must be TrainingEnvironmentEvidence")

        blockers: list[str] = []
        profile = plan.execution_profile
        if profile.profile_id != environment.execution_profile_id:
            blockers.append("execution profile does not match measured training environment")
        if profile.required_device_memory_bytes > environment.available_device_memory_bytes:
            blockers.append("measured accelerator memory is below the declared run requirement")
        if plan.method not in environment.supported_methods:
            blockers.append("training backend does not declare support for this method")
        if plan.dataset.dataset_id not in environment.accessible_dataset_ids:
            blockers.append("declared dataset is not accessible to the training environment")
        if plan.run_approval_reference not in environment.approved_run_references:
            blockers.append("training run lacks an explicit approval reference")
        if plan.method is TrainingMethod.DISTILLATION:
            if plan.teacher_candidate_id not in environment.available_teacher_candidate_ids:
                blockers.append("declared teacher candidate is unavailable to the training environment")

        plan_digest = self.plan_digest(plan)
        return TrainingReadinessReport(
            candidate_id=plan.candidate_id,
            plan_sha256=plan_digest,
            ready_to_start=not blockers,
            blockers=tuple(blockers),
            backend_id=environment.backend_id,
            artifact_store_reference=environment.artifact_store_reference,
        )

    @staticmethod
    def plan_digest(plan: CoreTrainingRunPlan) -> str:
        if not isinstance(plan, CoreTrainingRunPlan):
            raise TrainingPlanError("plan must be a CoreTrainingRunPlan")
        import hashlib
        import json

        payload = {
            "candidate_id": plan.candidate_id,
            "candidate_version": plan.candidate_version,
            "capability_ids": plan.capability_ids,
            "method": plan.method.value,
            "architecture_reference": plan.architecture_reference,
            "tokenizer_reference": plan.tokenizer_reference,
            "parameter_count": plan.parameter_count,
            "context_length_tokens": plan.context_length_tokens,
            "training_config_sha256": plan.training_config_sha256,
            "random_seed": plan.random_seed,
            "dataset_id": plan.dataset.dataset_id,
            "dataset_version": plan.dataset.version,
            "dataset_sha256": plan.dataset.content_sha256,
            "dataset_provenance_reference": plan.dataset.provenance_reference,
            "dataset_license_review_reference": plan.dataset.license_review_reference,
            "dataset_sources": [
                {
                    "source_id": source.source_id,
                    "authorization": source.authorization.value,
                    "authorization_reference": source.authorization_reference,
                }
                for source in plan.dataset.sources
            ],
            "training_record_ids": plan.dataset.training_record_ids,
            "validation_record_ids": plan.dataset.validation_record_ids,
            "test_record_ids": plan.dataset.test_record_ids,
            "execution_profile_id": plan.execution_profile.profile_id,
            "accelerator_profile_id": plan.execution_profile.accelerator_profile_id,
            "required_device_memory_bytes": plan.execution_profile.required_device_memory_bytes,
            "maximum_steps": plan.execution_profile.maximum_steps,
            "maximum_wall_time_seconds": plan.execution_profile.maximum_wall_time_seconds,
            "checkpoint_interval_steps": plan.execution_profile.checkpoint_interval_steps,
            "precision_profile_id": plan.execution_profile.precision_profile_id,
            "run_approval_reference": plan.run_approval_reference,
            "teacher_candidate_id": plan.teacher_candidate_id,
            "starting_candidate_id": plan.starting_candidate_id,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class CoreTrainingBackend(Protocol):
    """Interface for a future external or local training backend adapter."""

    backend_id: str

    def train(self, plan: CoreTrainingRunPlan) -> object: ...