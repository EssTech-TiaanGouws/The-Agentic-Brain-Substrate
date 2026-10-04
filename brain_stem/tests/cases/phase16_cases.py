from __future__ import annotations

import hashlib
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from time import time_ns

from models.stacey.core.config import (
    STACEY_CORE_ARCHITECTURE_ID,
    STACEY_CORE_TOKENIZER_ID,
    StaceyCoreConfig,
)
from models.stacey.core.inputs import HardwareCapabilityMatrix, IntentKind, IntentVector, UnifiedContextIngress
from models.stacey.core.outputs import (
    CalibrationMetrics,
    ClarificationDirective,
    CoreDecisionEnvelope,
    TaskDependencyGraph,
)
from substrate.contracts import ScopeVector
from src.swarm_core.core_evaluation import EvaluationEvidence
from src.swarm_core.foundry import (
    FoundryCandidateJob,
    FoundryCapacityDenied,
    FoundryError,
    FoundryJobConflict,
    FoundryJobJournal,
    FoundryJobState,
    FoundryPolicy,
    StaceyFoundry,
)
from src.swarm_core.hardware_profile import HardwareProfile
from src.swarm_core.model_catalog import (
    CandidateStatus,
    ModelCandidateCatalog,
    TrainingDataAuthorization,
    TrainingSource,
)
from src.swarm_core.training_governance import SignedGovernanceApproval
from src.swarm_core.training_readiness import (
    CoreTrainingReadinessChecker,
    CoreTrainingRunPlan,
    TrainingDatasetManifest,
    TrainingExecutionProfile,
    TrainingMethod,
)
from training.stacey.corpus import ApprovedStaceyDataset, StaceyTrainingExample
from training.stacey.trainer import (
    StaceyTrainerConfig,
    StaceyTrainingRunReport,
    training_config_sha256,
)


SCOPE = ScopeVector("tenant-a", "user-a", "project-a", "workspace-a")
PROTECTED_SUITE_ID = "suite.protected.core"


class FakeApprovalVerifier:
    def verify(self, approval, *, action, subject_sha256):
        return approval.approval_id


class FakeApprovalJournal:
    def consume(self, approval, *, action, subject_sha256):
        return approval.approval_id


class FakeTrainer:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("training failed")
        checkpoint_path = Path(kwargs["checkpoint_path"])
        checkpoint_path.write_bytes(b"trained-core-candidate")
        plan = kwargs["plan"]
        dataset = kwargs["dataset"]
        return StaceyTrainingRunReport(
            candidate_id=plan.candidate_id,
            readiness_plan_sha256=CoreTrainingReadinessChecker.plan_digest(plan),
            dataset_sha256=dataset.manifest.content_sha256,
            optimizer_steps=1,
            best_validation_loss=0.5,
            final_training_loss=0.7,
            checkpoint_path=str(checkpoint_path),
            checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            stop_reason="maximum_steps",
        )


class FixedEvaluator:
    def __init__(self, *, passed: bool = True, suite_id: str = PROTECTED_SUITE_ID) -> None:
        self.passed = passed
        self.suite_id = suite_id
        self.calls = 0

    def evaluate(self, *, candidate, checkpoint_path, training_report):
        self.calls += 1
        return EvaluationEvidence(
            suite_id=self.suite_id,
            run_reference="protected-run:1",
            artifact_sha256=candidate.artifact_sha256,
            baseline_reference="baseline:core-v0",
            passed=self.passed,
            metrics=(("case_pass_rate", 1.0 if self.passed else 0.0),),
        )


def make_job(*, candidate_id: str = "candidate-core-test", job_id: str = "job-core-test"):
    source = TrainingSource(
        "source:reviewed-synthetic",
        TrainingDataAuthorization.APPROVED_DATASET,
        "rights-review:fixture",
    )
    dataset_manifest = TrainingDatasetManifest(
        dataset_id="dataset:fixture",
        version="1.0",
        content_sha256=hashlib.sha256(b"dataset-bytes").hexdigest(),
        provenance_reference="provenance:fixture",
        license_review_reference="license-review:fixture",
        training_record_ids=("train-1",),
        validation_record_ids=("validation-1",),
        test_record_ids=("holdout-1",),
        sources=(source,),
    )
    dataset_approval = SignedGovernanceApproval(
        approval_id="dataset-approval",
        reviewer_id="reviewer-test",
        reviewer_role="DATA_STEWARD",
        action="DATASET_APPROVE",
        subject_sha256=hashlib.sha256(b"dataset-manifest").hexdigest(),
        audience="foundry-test",
        issued_at=1,
        expires_at=2,
        signature=b"test-signature",
    )
    ingress = UnifiedContextIngress(
        protocol_version="1.0",
        transaction_id="tx-training",
        correlation_id="turn-training",
        ingress_timestamp_ns=1,
        intent_vector=IntentVector(IntentKind.USER_LANGUAGE, "clarify an ambiguous request"),
        scope_vector=SCOPE,
        canonical_state_assertions=(),
        hardware_capability_matrix=HardwareCapabilityMatrix((), (), ()),
    )
    decision = CoreDecisionEnvelope(
        protocol_version="1.0",
        transaction_id=ingress.transaction_id,
        correlation_id=ingress.correlation_id,
        assigned_block_id="BLOCK_0_CORE",
        predicted_consequences_summary="Ask the user to clarify.",
        calibration_metrics=CalibrationMetrics(0.9),
        declarative_intent_action="CLARIFY",
        scope_vector=SCOPE,
        task_dependency_graph=TaskDependencyGraph((), (), ()),
        clarification=ClarificationDirective(True, ("AMBIGUOUS_INTENT",)),
    )

    def example(example_id: str, source_group_id: str) -> StaceyTrainingExample:
        return StaceyTrainingExample(
            example_id=example_id,
            source_group_id=source_group_id,
            source_id=source.source_id,
            authorization_reference=source.authorization_reference,
            reviewer_reference="reviewer:fixture",
            ingress=ingress,
            target_decision_jsonl="{}\n",
            decision=decision,
        )

    dataset = ApprovedStaceyDataset(
        manifest=dataset_manifest,
        manifest_sha256=hashlib.sha256(b"dataset-manifest").hexdigest(),
        approval=dataset_approval,
        training_examples=(example("train-1", "group-train"),),
        validation_examples=(example("validation-1", "group-validation"),),
        holdout_record_ids=("holdout-1",),
        holdout_source_group_ids=("group-holdout",),
        holdout_sha256=hashlib.sha256(b"holdout").hexdigest(),
        holdout_evaluation_reference="evaluation:protected-fixture",
    )
    model_config = StaceyCoreConfig(
        architecture_id=STACEY_CORE_ARCHITECTURE_ID,
        tokenizer_id=STACEY_CORE_TOKENIZER_ID,
        vocabulary_size=259,
        model_dimension=8,
        attention_heads=2,
        encoder_layers=1,
        decoder_layers=1,
        feedforward_dimension=16,
        maximum_input_tokens=1024,
        maximum_output_tokens=64,
        dropout_probability=0.0,
    )
    trainer_config = StaceyTrainerConfig(
        device_id="cpu",
        learning_rate=0.001,
        weight_decay=0.0,
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        validation_interval_steps=1,
        early_stopping_patience=1,
        maximum_hardware_profile_age_seconds=3600,
    )
    plan = CoreTrainingRunPlan(
        candidate_id=candidate_id,
        candidate_version="0.1.0",
        capability_ids=("core.reason",),
        method=TrainingMethod.FROM_SCRATCH,
        architecture_reference=model_config.architecture_id,
        tokenizer_reference=model_config.tokenizer_id,
        parameter_count=100,
        context_length_tokens=1024,
        training_config_sha256=training_config_sha256(model_config, trainer_config),
        random_seed=1,
        dataset=dataset_manifest,
        execution_profile=TrainingExecutionProfile(
            profile_id="cpu-test",
            accelerator_profile_id="cpu",
            required_device_memory_bytes=0,
            maximum_steps=1,
            maximum_wall_time_seconds=10,
            checkpoint_interval_steps=1,
            precision_profile_id="fp32",
        ),
        run_approval_reference="run-approval",
    )
    run_approval = SignedGovernanceApproval(
        approval_id="run-approval",
        reviewer_id="reviewer-test",
        reviewer_role="MODEL_TRAINING",
        action="TRAINING_RUN_APPROVE",
        subject_sha256=hashlib.sha256(b"training-plan").hexdigest(),
        audience="foundry-test",
        issued_at=1,
        expires_at=2,
        signature=b"test-signature",
    )
    hardware_profile = HardwareProfile(
        observed_at_ns=time_ns(),
        operating_system="test-os",
        machine_architecture="test-arch",
        logical_cpu_count=2,
        host_memory_total_bytes=1_000_000,
        host_memory_available_bytes=900_000,
        tensor_runtime="torch-test",
        cuda_runtime=None,
        accelerators=(),
    )
    return FoundryCandidateJob(
        job_id=job_id,
        plan=plan,
        dataset=dataset,
        hardware_profile=hardware_profile,
        model_config=model_config,
        trainer_config=trainer_config,
        run_approval=run_approval,
    )


class StaceyFoundryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        root = Path(self.temporary_directory.name)
        self.artifact_root = root / "artifacts"
        self.artifact_root.mkdir()
        self.journal_path = root / "foundry.sqlite3"
        self.journal = FoundryJobJournal(self.journal_path)
        self.addCleanup(self.journal.close)
        self.catalog = ModelCandidateCatalog()
        self.trainer = FakeTrainer()
        self.evaluator = FixedEvaluator()
        self.foundry = self.make_foundry()

    def make_foundry(
        self,
        *,
        trainer: FakeTrainer | None = None,
        evaluator: FixedEvaluator | None = None,
        policy: FoundryPolicy | None = None,
    ) -> StaceyFoundry:
        return StaceyFoundry(
            artifact_root=self.artifact_root,
            journal=self.journal,
            candidate_catalog=self.catalog,
            approval_verifier=FakeApprovalVerifier(),
            approval_journal=FakeApprovalJournal(),
            evaluator=evaluator or self.evaluator,
            policy=policy
            or FoundryPolicy(
                maximum_steps=10,
                maximum_wall_time_seconds=30,
                maximum_parameter_count=1_000_000,
                maximum_training_examples=4,
                maximum_validation_examples=4,
                maximum_artifact_size_bytes=1024,
                maximum_concurrent_jobs=1,
                protected_evaluation_suite_id=PROTECTED_SUITE_ID,
            ),
            training_runner=trainer or self.trainer,
        )

    def test_foundry_proposes_and_evaluates_without_approving_or_activating(self) -> None:
        result = self.foundry.execute(make_job())

        self.assertEqual(result.candidate.status, CandidateStatus.EVALUATED)
        self.assertEqual(self.catalog.approved_for("core.reason"), ())
        self.assertFalse(hasattr(self.foundry, "approve"))
        self.assertFalse(hasattr(self.foundry, "activate_release"))
        self.assertEqual(
            tuple(event.state for event in self.journal.events("job-core-test")),
            (
                FoundryJobState.SUBMITTED,
                FoundryJobState.RUNNING,
                FoundryJobState.CANDIDATE_PROPOSED,
                FoundryJobState.EVALUATED,
            ),
        )

    def test_failed_protected_evaluation_rejects_candidate(self) -> None:
        self.evaluator = FixedEvaluator(passed=False)
        self.foundry = self.make_foundry(evaluator=self.evaluator)

        result = self.foundry.execute(make_job())

        self.assertEqual(result.candidate.status, CandidateStatus.REJECTED)
        self.assertEqual(self.catalog.approved_for("core.reason"), ())
        self.assertEqual(result.final_event.state, FoundryJobState.REJECTED)

    def test_policy_denies_excess_steps_before_training_or_journaling(self) -> None:
        job = make_job()
        oversized_profile = replace(job.plan.execution_profile, maximum_steps=11)
        oversized_job = replace(job, plan=replace(job.plan, execution_profile=oversized_profile))

        with self.assertRaises(FoundryCapacityDenied):
            self.foundry.execute(oversized_job)

        self.assertEqual(self.trainer.calls, 0)
        self.assertEqual(self.journal.events(oversized_job.job_id), ())

    def test_trainer_failure_is_durably_recorded(self) -> None:
        failing_trainer = FakeTrainer(fail=True)
        self.foundry = self.make_foundry(trainer=failing_trainer)

        with self.assertRaisesRegex(RuntimeError, "training failed"):
            self.foundry.execute(make_job())

        events = self.journal.events("job-core-test")
        self.assertEqual(events[-1].state, FoundryJobState.FAILED)
        self.assertEqual(events[-1].error_type, "RuntimeError")

    def test_existing_candidate_job_cannot_be_replayed(self) -> None:
        job = make_job()
        self.foundry.execute(job)

        with self.assertRaises(FoundryJobConflict):
            self.foundry.execute(job)

        self.assertEqual(self.trainer.calls, 1)

    def test_unexpected_evaluation_suite_is_failed_without_candidate_approval(self) -> None:
        self.evaluator = FixedEvaluator(suite_id="visible-fixture-only")
        self.foundry = self.make_foundry(evaluator=self.evaluator)

        with self.assertRaises(FoundryError):
            self.foundry.execute(make_job())

        self.assertEqual(self.catalog.approved_for("core.reason"), ())
        self.assertEqual(self.journal.events("job-core-test")[-1].state, FoundryJobState.FAILED)

    def test_journal_tampering_is_rejected_after_restart(self) -> None:
        self.foundry.execute(make_job())
        self.journal.close()
        connection = sqlite3.connect(self.journal_path)
        connection.execute("DROP TRIGGER foundry_events_no_update")
        connection.execute("UPDATE foundry_events SET event_hash = ? WHERE sequence = 1", ("0" * 64,))
        connection.commit()
        connection.close()

        with self.assertRaises(FoundryError):
            FoundryJobJournal(self.journal_path)


if __name__ == "__main__":
    unittest.main()