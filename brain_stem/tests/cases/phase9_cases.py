from __future__ import annotations

import json
import hashlib
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from time import time
from unittest.mock import patch

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from models.stacey.core.inputs import (
    HardwareCapabilityMatrix,
    IntentKind,
    IntentVector,
    ResourceMeasurement,
    SpecialistAvailability,
    SpecialistSlot,
    UnifiedContextIngress,
)
from models.stacey.core.checkpoint import load_model_checkpoint
from models.stacey.core.config import (
    STACEY_CORE_ARCHITECTURE_ID,
    STACEY_CORE_TOKENIZER_ID,
    StaceyCoreConfig,
)
from models.stacey.core.initialize import initialize_stacey_core
from models.stacey.core.outputs import (
    CalibrationMetrics,
    ClarificationDirective,
    CoreDecisionEnvelope,
    ResourceEstimate,
    TaskDependencyGraph,
    TaskGraphNode,
)
from substrate.contracts import ScopeVector
from src.swarm_core.hardware_profile import (
    AcceleratorSnapshot,
    HardwareProbeError,
    HardwareProfile,
    probe_hardware_profile,
)
from src.swarm_core.model_catalog import TrainingDataAuthorization
from src.swarm_core.training_readiness import (
    CoreTrainingRunPlan,
    CoreTrainingReadinessChecker,
    TrainingExecutionProfile,
    TrainingMethod,
)
from src.swarm_core.training_governance import (
    Ed25519GovernanceApprovalVerifier,
    GovernanceApprovalJournal,
    SignedGovernanceApproval,
    governance_approval_message,
)
from training.stacey.corpus import (
    StaceyCorpusError,
    dataset_content_sha256,
    load_approved_stacey_dataset,
)
from training.stacey.trainer import (
    StaceyTrainerConfig,
    StaceyTrainingBlockedError,
    StaceyTrainingRunError,
    train_stacey_from_scratch,
    training_config_sha256,
)
from models.stacey.core.tokens import BYTE_VOCABULARY_SIZE


DATA_REVIEWER_KEY = Ed25519PrivateKey.generate()
RUN_AUTHORIZER_KEY = Ed25519PrivateKey.generate()


def _public_key(private_key: Ed25519PrivateKey) -> bytes:
    return private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


GOVERNANCE_VERIFIER = Ed25519GovernanceApprovalVerifier(
    {
        ("fixture-data-reviewer", "DATA_REVIEWER"): _public_key(DATA_REVIEWER_KEY),
        ("fixture-run-authorizer", "RUN_AUTHORIZER"): _public_key(RUN_AUTHORIZER_KEY),
    },
    allowed_roles_by_action={
        "DATASET_APPROVE": ("DATA_REVIEWER",),
        "TRAINING_RUN_APPROVE": ("RUN_AUTHORIZER",),
    },
    audience="stacey-tests",
)


def _signed_approval(action: str, subject_sha256: str) -> SignedGovernanceApproval:
    now = int(time())
    reviewer_id, reviewer_role, private_key = (
        ("fixture-data-reviewer", "DATA_REVIEWER", DATA_REVIEWER_KEY)
        if action == "DATASET_APPROVE"
        else ("fixture-run-authorizer", "RUN_AUTHORIZER", RUN_AUTHORIZER_KEY)
    )
    unsigned = SignedGovernanceApproval(
        approval_id=f"fixture-approval:{action}",
        reviewer_id=reviewer_id,
        reviewer_role=reviewer_role,
        action=action,
        subject_sha256=subject_sha256,
        audience="stacey-tests",
        issued_at=now,
        expires_at=now + 300,
        signature=b"",
    )
    return replace(unsigned, signature=private_key.sign(governance_approval_message(unsigned)))


def load_fixture_dataset(directory: Path):
    manifest_sha256 = hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
    return load_approved_stacey_dataset(
        directory,
        approval_verifier=GOVERNANCE_VERIFIER,
        approval=_signed_approval("DATASET_APPROVE", manifest_sha256),
    )


class HardwareProfileTests(unittest.TestCase):
    def test_probe_returns_serializable_host_snapshot_and_explicit_accelerators(self) -> None:
        profile = probe_hardware_profile()
        payload = profile.to_payload()

        self.assertGreater(profile.logical_cpu_count, 0)
        self.assertGreater(profile.observed_at_ns, 0)
        self.assertEqual(len(payload["accelerators"]), len(profile.accelerators))
        self.assertIsInstance(json.dumps(payload), str)
        for device in profile.accelerators:
            self.assertLessEqual(device.available_memory_bytes, device.total_memory_bytes)

    def test_profile_rejects_available_memory_greater_than_capacity(self) -> None:
        with self.assertRaises(HardwareProbeError):
            AcceleratorSnapshot("cuda:0", "cuda", "test", 10, 11, True, False)

    def test_cpu_profile_maps_to_zero_device_memory_and_from_scratch_only(self) -> None:
        environment = probe_hardware_profile().to_training_environment_evidence(
            device_id="cpu",
            execution_profile_id="cpu-pilot",
            accessible_dataset_ids=("dataset-fixture",),
            approved_run_references=("approved-run-fixture",),
            artifact_store_reference="artifact-store-fixture",
        )

        self.assertEqual(environment.backend_id, "torch.cpu")
        self.assertEqual(environment.available_device_memory_bytes, 0)
        self.assertEqual(environment.supported_methods, (TrainingMethod.FROM_SCRATCH,))

    def test_profile_refuses_an_accelerator_not_in_the_snapshot(self) -> None:
        with self.assertRaisesRegex(HardwareProbeError, "not present"):
            probe_hardware_profile().to_training_environment_evidence(
                device_id="cuda:99",
                execution_profile_id="cuda-profile",
                accessible_dataset_ids=(),
                approved_run_references=(),
                artifact_store_reference="artifact-store-fixture",
            )


def make_ingress() -> UnifiedContextIngress:
    observed_at_ns = 1_800_000_000_000_000_000
    return UnifiedContextIngress(
        protocol_version="1.0",
        transaction_id="tx-corpus-fixture",
        correlation_id="turn-corpus-fixture",
        ingress_timestamp_ns=observed_at_ns,
        intent_vector=IntentVector(IntentKind.USER_LANGUAGE, "Summarize the verified record."),
        scope_vector=ScopeVector("tenant", "user", "project", "workspace"),
        canonical_state_assertions=(),
        hardware_capability_matrix=HardwareCapabilityMatrix(
            resources=(ResourceMeasurement("host-memory", 4_000_000_000, observed_at_ns),),
            active_leases=(),
            available_specialist_slots=(
                SpecialistSlot(
                    block_id="BLOCK_12_LINGUISTIC_COPY",
                    capability_id="text.summarize",
                    status=SpecialistAvailability.AVAILABLE,
                    resource_domain="host-memory",
                    estimated_required_bytes=100_000_000,
                    artifact_sha256="a" * 64,
                    input_modalities=("text",),
                ),
            ),
        ),
    )


def make_target(ingress: UnifiedContextIngress) -> CoreDecisionEnvelope:
    return CoreDecisionEnvelope(
        protocol_version="1.0",
        transaction_id=ingress.transaction_id,
        correlation_id=ingress.correlation_id,
        assigned_block_id="BLOCK_0_CORE",
        predicted_consequences_summary="Summarize the verified record without changing it.",
        calibration_metrics=CalibrationMetrics(0.9),
        declarative_intent_action="DECOMPOSE_TASK_GRAPH",
        scope_vector=ingress.scope_vector,
        task_dependency_graph=TaskDependencyGraph(
            nodes=(TaskGraphNode("summarize", "text.summarize", "BLOCK_12_LINGUISTIC_COPY"),),
            edges=(),
            resource_estimates=(ResourceEstimate("summarize", "host-memory", 100_000_000),),
        ),
        clarification=ClarificationDirective(False, ()),
    )


def build_dataset_directory(
    directory: Path,
    *,
    external_model_generated: bool = False,
    leak_source_group: bool = False,
    duplicate_semantic_example: bool = False,
) -> None:
    split_contents: dict[str, bytes] = {}
    split_entries: dict[str, object] = {}
    for split_name, example_id in (
        ("training", "train-001"),
        ("validation", "validation-001"),
    ):
        ingress = make_ingress()
        ingress_payload = ingress.to_payload()
        ingress_payload["transaction_id"] = f"tx-{example_id}"
        ingress_payload["correlation_id"] = f"turn-{example_id}"
        if not duplicate_semantic_example:
            ingress_payload["intent_vector"]["payload"] += f" Split {split_name}."
        ingress = UnifiedContextIngress.from_payload(ingress_payload)
        record = {
            "example_id": example_id,
            "source_group_id": "shared-scenario" if leak_source_group else f"group-{split_name}",
            "source_id": "human-authored-source",
            "authorization_reference": "source-approval-001",
            "reviewer_reference": "human-review-001",
            "human_reviewed": True,
            "external_model_generated": external_model_generated,
            "ingress": ingress.to_payload(),
            "target_decision_jsonl": make_target(ingress).to_jsonl(),
        }
        filename = f"{split_name}.jsonl"
        content = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        (directory / filename).write_bytes(content)
        split_contents[split_name] = content
        split_entries[split_name] = {
            "file": filename,
            "sha256": hashlib.sha256(content).hexdigest(),
            "record_ids": [example_id],
        }

    holdout_sha256 = hashlib.sha256(b"protected holdout content remains outside the trainer").hexdigest()
    manifest = {
        "schema_version": "stacey.training.dataset.v1",
        "dataset_id": "stacey-task-graph-fixture",
        "version": "test-v1",
        "provenance_reference": "fixture-provenance-record",
        "license_review_reference": "fixture-rights-review",
        "content_sha256": dataset_content_sha256(
            split_contents,
            protected_holdout_sha256=holdout_sha256,
        ),
        "sources": [
            {
                "source_id": "human-authored-source",
                "authorization": TrainingDataAuthorization.APPROVED_DATASET.value,
                "authorization_reference": "source-approval-001",
            }
        ],
        "splits": split_entries,
        "holdout": {
            "sha256": holdout_sha256,
            "record_ids": ["holdout-001"],
            "source_group_ids": ["group-holdout" if not leak_source_group else "shared-scenario"],
            "evaluation_reference": "separate-protected-evaluation-store",
        },
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


class StaceyCorpusTests(unittest.TestCase):
    def test_loader_validates_provenance_contract_targets_and_split_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            build_dataset_directory(directory)

            dataset = load_fixture_dataset(directory)

        self.assertEqual(dataset.manifest.dataset_id, "stacey-task-graph-fixture")
        self.assertEqual(tuple(example.example_id for example in dataset.training_examples), ("train-001",))
        self.assertEqual(dataset.validation_examples[0].decision.scope_vector, make_ingress().scope_vector)
        self.assertEqual(dataset.holdout_record_ids, ("holdout-001",))
        self.assertFalse(hasattr(dataset, "holdout_examples"))
        self.assertEqual(len(dataset.manifest_sha256), 64)

    def test_loader_rejects_external_model_generated_examples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            build_dataset_directory(directory, external_model_generated=True)

            with self.assertRaisesRegex(StaceyCorpusError, "external-model-generated"):
                load_fixture_dataset(directory)

    def test_loader_rejects_source_group_leakage_between_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            build_dataset_directory(directory, leak_source_group=True)

            with self.assertRaisesRegex(StaceyCorpusError, "source groups must not cross"):
                load_fixture_dataset(directory)

    def test_loader_rejects_semantically_duplicate_examples_across_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            build_dataset_directory(directory, duplicate_semantic_example=True)

            with self.assertRaisesRegex(StaceyCorpusError, "semantically duplicate examples"):
                load_fixture_dataset(directory)

    def test_loader_rejects_tampered_split_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            build_dataset_directory(directory)
            (directory / "training.jsonl").write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(StaceyCorpusError, "file digest"):
                load_fixture_dataset(directory)

    def test_loader_rejects_manifest_without_signed_reviewer_approval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            build_dataset_directory(directory)
            with self.assertRaisesRegex(StaceyCorpusError, "signed reviewer approval"):
                load_approved_stacey_dataset(
                    directory,
                    approval_verifier=GOVERNANCE_VERIFIER,
                    approval=_signed_approval("DATASET_APPROVE", "0" * 64),
                )


class StaceyScratchTrainerTests(unittest.TestCase):
    @staticmethod
    def make_configs() -> tuple[StaceyCoreConfig, StaceyTrainerConfig]:
        model_config = StaceyCoreConfig(
            architecture_id=STACEY_CORE_ARCHITECTURE_ID,
            tokenizer_id=STACEY_CORE_TOKENIZER_ID,
            vocabulary_size=BYTE_VOCABULARY_SIZE,
            model_dimension=16,
            attention_heads=2,
            encoder_layers=1,
            decoder_layers=1,
            feedforward_dimension=32,
            maximum_input_tokens=2048,
            maximum_output_tokens=2048,
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
            maximum_hardware_profile_age_seconds=60,
        )
        return model_config, trainer_config

    def make_plan(self, dataset, model_config, trainer_config) -> CoreTrainingRunPlan:
        parameter_count = initialize_stacey_core(model_config, random_seed=23).parameter_count
        return CoreTrainingRunPlan(
            candidate_id="stacey-scratch-core-test",
            candidate_version="test-v1",
            capability_ids=("core.task_graph_planning",),
            method=TrainingMethod.FROM_SCRATCH,
            architecture_reference=model_config.architecture_id,
            tokenizer_reference=model_config.tokenizer_id,
            parameter_count=parameter_count,
            context_length_tokens=2048,
            training_config_sha256=training_config_sha256(model_config, trainer_config),
            random_seed=23,
            dataset=dataset.manifest,
            execution_profile=TrainingExecutionProfile(
                profile_id="cpu-scratch-pilot",
                accelerator_profile_id="cpu",
                required_device_memory_bytes=0,
                maximum_steps=1,
                maximum_wall_time_seconds=120,
                checkpoint_interval_steps=1,
                precision_profile_id="fp32",
            ),
            run_approval_reference="fixture-approval:TRAINING_RUN_APPROVE",
        )

    def test_trainer_runs_one_from_scratch_step_and_saves_bound_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_directory = root / "dataset"
            dataset_directory.mkdir()
            build_dataset_directory(dataset_directory)
            dataset = load_fixture_dataset(dataset_directory)
            model_config, trainer_config = self.make_configs()
            plan = self.make_plan(dataset, model_config, trainer_config)
            artifact_directory = root / "artifacts"
            artifact_directory.mkdir()
            checkpoint_path = artifact_directory / "stacey-scratch-test.pt"
            approval_journal = GovernanceApprovalJournal(
                root / "approvals.sqlite3",
                verifier=GOVERNANCE_VERIFIER,
            )
            self.addCleanup(approval_journal.close)

            report = train_stacey_from_scratch(
                plan=plan,
                dataset=dataset,
                hardware_profile=probe_hardware_profile(),
                model_config=model_config,
                trainer_config=trainer_config,
                approval_verifier=GOVERNANCE_VERIFIER,
                approval_journal=approval_journal,
                run_approval=_signed_approval(
                    "TRAINING_RUN_APPROVE",
                    CoreTrainingReadinessChecker.plan_digest(plan),
                ),
                checkpoint_path=checkpoint_path,
                checkpoint_file_mode=0o600,
            )
            restored = load_model_checkpoint(
                checkpoint_path,
                expected_sha256=report.checkpoint_sha256,
                map_location="cpu",
            )
            periodic_checkpoint_exists = checkpoint_path.with_name(
                "stacey-scratch-test.step-00000001.pt"
            ).is_file()

        self.assertEqual(report.optimizer_steps, 1)
        self.assertTrue(math.isfinite(report.best_validation_loss))
        self.assertTrue(math.isfinite(report.final_training_loss))
        self.assertEqual(restored.parameter_count, plan.parameter_count)
        self.assertEqual(report.dataset_sha256, dataset.manifest.content_sha256)
        self.assertTrue(periodic_checkpoint_exists)

    def test_trainer_resumes_optimizer_rng_and_batch_cursor_from_bound_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_directory = root / "dataset"
            dataset_directory.mkdir()
            build_dataset_directory(dataset_directory)
            dataset = load_fixture_dataset(dataset_directory)
            model_config, trainer_config = self.make_configs()
            trainer_config = replace(trainer_config, validation_interval_steps=2)
            plan = self.make_plan(dataset, model_config, trainer_config)
            plan = replace(
                plan,
                execution_profile=replace(
                    plan.execution_profile,
                    maximum_steps=3,
                    maximum_wall_time_seconds=1,
                ),
            )
            artifact_directory = root / "artifacts"
            artifact_directory.mkdir()
            checkpoint_path = artifact_directory / "resumable-core.pt"
            resume_state_path = artifact_directory / "resumable-core.resume.pt"
            approval_journal = GovernanceApprovalJournal(
                root / "approvals.sqlite3",
                verifier=GOVERNANCE_VERIFIER,
            )
            self.addCleanup(approval_journal.close)
            run_approval = _signed_approval(
                "TRAINING_RUN_APPROVE",
                CoreTrainingReadinessChecker.plan_digest(plan),
            )

            with patch("training.stacey.trainer.time.monotonic", side_effect=(0.0, 0.0, 5.0)):
                interrupted = train_stacey_from_scratch(
                    plan=plan,
                    dataset=dataset,
                    hardware_profile=probe_hardware_profile(),
                    model_config=model_config,
                    trainer_config=trainer_config,
                    approval_verifier=GOVERNANCE_VERIFIER,
                    approval_journal=approval_journal,
                    run_approval=run_approval,
                    checkpoint_path=checkpoint_path,
                    checkpoint_file_mode=0o600,
                    resume_state_path=resume_state_path,
                )

            self.assertEqual(interrupted.optimizer_steps, 1)
            self.assertEqual(interrupted.stop_reason, "maximum_wall_time")
            self.assertTrue(resume_state_path.is_file())
            self.assertEqual(len(interrupted.resume_state_sha256), 64)

            with patch("training.stacey.trainer.time.monotonic", side_effect=(0.0, 0.0, 0.0)):
                resumed = train_stacey_from_scratch(
                    plan=plan,
                    dataset=dataset,
                    hardware_profile=probe_hardware_profile(),
                    model_config=model_config,
                    trainer_config=trainer_config,
                    approval_verifier=GOVERNANCE_VERIFIER,
                    approval_journal=approval_journal,
                    run_approval=run_approval,
                    checkpoint_path=checkpoint_path,
                    checkpoint_file_mode=0o600,
                    resume_state_path=resume_state_path,
                    resume_from_sha256=interrupted.resume_state_sha256,
                )
            checkpoint_exists = Path(resumed.checkpoint_path).is_file()
            resume_bytes = resume_state_path.read_bytes()
            resume_state_path.write_bytes(resume_bytes[:-1] + bytes((resume_bytes[-1] ^ 1,)))
            with patch("training.stacey.trainer.time.monotonic", return_value=0.0):
                with self.assertRaisesRegex(StaceyTrainingRunError, "resume checkpoint could not be verified"):
                    train_stacey_from_scratch(
                        plan=plan,
                        dataset=dataset,
                        hardware_profile=probe_hardware_profile(),
                        model_config=model_config,
                        trainer_config=trainer_config,
                        approval_verifier=GOVERNANCE_VERIFIER,
                        approval_journal=approval_journal,
                        run_approval=run_approval,
                        checkpoint_path=checkpoint_path,
                        checkpoint_file_mode=0o600,
                        resume_state_path=resume_state_path,
                        resume_from_sha256=resumed.resume_state_sha256,
                    )

        self.assertEqual(resumed.resumed_from_step, 1)
        self.assertEqual(resumed.optimizer_steps, 3)
        self.assertEqual(resumed.dataset_sha256, dataset.manifest.content_sha256)
        self.assertTrue(checkpoint_exists)

    def test_trainer_blocks_run_without_explicit_approval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_directory = root / "dataset"
            dataset_directory.mkdir()
            build_dataset_directory(dataset_directory)
            dataset = load_fixture_dataset(dataset_directory)
            model_config, trainer_config = self.make_configs()
            plan = self.make_plan(dataset, model_config, trainer_config)
            artifact_directory = root / "artifacts"
            artifact_directory.mkdir()
            approval_journal = GovernanceApprovalJournal(
                root / "approvals.sqlite3",
                verifier=GOVERNANCE_VERIFIER,
            )
            self.addCleanup(approval_journal.close)

            with self.assertRaises(StaceyTrainingBlockedError):
                train_stacey_from_scratch(
                    plan=plan,
                    dataset=dataset,
                    hardware_profile=probe_hardware_profile(),
                    model_config=model_config,
                    trainer_config=trainer_config,
                    approval_verifier=GOVERNANCE_VERIFIER,
                    approval_journal=approval_journal,
                    run_approval=None,
                    checkpoint_path=artifact_directory / "must-not-exist.pt",
                    checkpoint_file_mode=0o600,
                )


if __name__ == "__main__":
    unittest.main()