from __future__ import annotations

import unittest
from dataclasses import replace

from src.swarm_core.model_catalog import TrainingDataAuthorization, TrainingSource
from src.swarm_core.training_readiness import (
    CoreTrainingReadinessChecker,
    CoreTrainingRunPlan,
    TrainingDatasetManifest,
    TrainingEnvironmentEvidence,
    TrainingExecutionProfile,
    TrainingMethod,
    TrainingPlanError,
)


DATASET_SHA256 = "d" * 64
TRAINING_CONFIG_SHA256 = "e" * 64


def make_dataset() -> TrainingDatasetManifest:
    return TrainingDatasetManifest(
        dataset_id="core-reasoning-dataset",
        version="v1",
        content_sha256=DATASET_SHA256,
        provenance_reference="dataset-provenance-record",
        license_review_reference="license-review-record",
        training_record_ids=("train-a", "train-b"),
        validation_record_ids=("validation-a",),
        test_record_ids=("test-a",),
        sources=(
            TrainingSource(
                source_id="approved-curated-source",
                authorization=TrainingDataAuthorization.APPROVED_DATASET,
                authorization_reference="source-approval-record",
            ),
        ),
    )


def make_profile() -> TrainingExecutionProfile:
    return TrainingExecutionProfile(
        profile_id="training-profile-24gb",
        accelerator_profile_id="accelerator-profile-a",
        required_device_memory_bytes=20_000_000_000,
        maximum_steps=10_000,
        maximum_wall_time_seconds=86_400,
        checkpoint_interval_steps=500,
        precision_profile_id="precision-profile-bf16",
    )


def make_plan(
    *,
    method: TrainingMethod = TrainingMethod.DISTILLATION,
    dataset: TrainingDatasetManifest | None = None,
    execution_profile: TrainingExecutionProfile | None = None,
) -> CoreTrainingRunPlan:
    return CoreTrainingRunPlan(
        candidate_id="world-core-candidate-001",
        candidate_version="0.1.0",
        capability_ids=("core.intent_understanding", "core.reasoning_planning"),
        method=method,
        architecture_reference="architecture-spec-ref",
        tokenizer_reference="tokenizer-spec-ref",
        parameter_count=700_000_000,
        context_length_tokens=4096,
        training_config_sha256=TRAINING_CONFIG_SHA256,
        random_seed=17,
        dataset=dataset or make_dataset(),
        execution_profile=execution_profile or make_profile(),
        run_approval_reference="training-run-approval-ref",
        teacher_candidate_id="teacher-candidate-001" if method is TrainingMethod.DISTILLATION else None,
        starting_candidate_id="starting-candidate-001" if method is TrainingMethod.FINE_TUNING else None,
    )


def make_environment(**changes: object) -> TrainingEnvironmentEvidence:
    values: dict[str, object] = {
        "backend_id": "external-training-backend",
        "backend_version": "backend-version-ref",
        "execution_profile_id": "training-profile-24gb",
        "available_device_memory_bytes": 24_000_000_000,
        "supported_methods": (
            TrainingMethod.FROM_SCRATCH,
            TrainingMethod.DISTILLATION,
            TrainingMethod.FINE_TUNING,
        ),
        "accessible_dataset_ids": ("core-reasoning-dataset",),
        "available_teacher_candidate_ids": ("teacher-candidate-001",),
        "artifact_store_reference": "artifact-store-ref",
        "approved_run_references": ("training-run-approval-ref",),
    }
    values.update(changes)
    return TrainingEnvironmentEvidence(**values)  # type: ignore[arg-type]


class CoreTrainingReadinessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.checker = CoreTrainingReadinessChecker()

    def test_complete_distillation_plan_is_ready_only_against_matching_environment(self) -> None:
        report = self.checker.assess(make_plan(), make_environment())

        self.assertTrue(report.ready_to_start)
        self.assertEqual(report.blockers, ())
        self.assertEqual(report.backend_id, "external-training-backend")
        self.assertEqual(len(report.plan_sha256), 64)

    def test_dataset_splits_must_be_disjoint(self) -> None:
        with self.assertRaises(TrainingPlanError):
            replace(make_dataset(), validation_record_ids=("train-a",))

    def test_inaccessible_dataset_blocks_run(self) -> None:
        environment = make_environment(accessible_dataset_ids=())

        report = self.checker.assess(make_plan(), environment)

        self.assertFalse(report.ready_to_start)
        self.assertIn("declared dataset is not accessible to the training environment", report.blockers)

    def test_unapproved_run_blocks_training(self) -> None:
        environment = make_environment(approved_run_references=())

        report = self.checker.assess(make_plan(), environment)

        self.assertFalse(report.ready_to_start)
        self.assertIn("training run lacks an explicit approval reference", report.blockers)

    def test_insufficient_measured_accelerator_memory_blocks_training(self) -> None:
        environment = make_environment(available_device_memory_bytes=1)

        report = self.checker.assess(make_plan(), environment)

        self.assertFalse(report.ready_to_start)
        self.assertIn("measured accelerator memory is below the declared run requirement", report.blockers)

    def test_cpu_only_execution_profile_allows_zero_device_memory_requirement(self) -> None:
        profile = TrainingExecutionProfile(
            profile_id="cpu-scratch-pilot",
            accelerator_profile_id="cpu",
            required_device_memory_bytes=0,
            maximum_steps=1,
            maximum_wall_time_seconds=60,
            checkpoint_interval_steps=1,
            precision_profile_id="fp32",
        )

        self.assertEqual(profile.required_device_memory_bytes, 0)

    def test_backend_must_support_the_selected_training_method(self) -> None:
        environment = make_environment(supported_methods=(TrainingMethod.FROM_SCRATCH,))

        report = self.checker.assess(make_plan(), environment)

        self.assertFalse(report.ready_to_start)
        self.assertIn("training backend does not declare support for this method", report.blockers)

    def test_distillation_teacher_must_be_available(self) -> None:
        environment = make_environment(available_teacher_candidate_ids=())

        report = self.checker.assess(make_plan(), environment)

        self.assertFalse(report.ready_to_start)
        self.assertIn("declared teacher candidate is unavailable to the training environment", report.blockers)

    def test_plan_digest_binds_data_provenance_and_split_membership(self) -> None:
        plan = make_plan()
        changed_provenance = replace(
            plan,
            dataset=replace(plan.dataset, provenance_reference="different-provenance"),
        )
        changed_splits = replace(
            plan,
            dataset=replace(plan.dataset, training_record_ids=("train-a", "train-c")),
        )

        base_digest = self.checker.assess(plan, make_environment()).plan_sha256
        self.assertNotEqual(base_digest, self.checker.assess(changed_provenance, make_environment()).plan_sha256)
        self.assertNotEqual(base_digest, self.checker.assess(changed_splits, make_environment()).plan_sha256)

    def test_training_methods_require_their_own_lineage(self) -> None:
        with self.assertRaises(TrainingPlanError):
            replace(make_plan(), teacher_candidate_id=None)
        with self.assertRaises(TrainingPlanError):
            replace(
                make_plan(),
                method=TrainingMethod.FINE_TUNING,
                teacher_candidate_id=None,
                starting_candidate_id=None,
            )


if __name__ == "__main__":
    unittest.main()