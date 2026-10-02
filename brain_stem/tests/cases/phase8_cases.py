from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from models.stacey.core import (
    ActiveResourceLease,
    CalibrationMetrics,
    ClarificationDirective,
    CoreDecisionEnvelope,
    EdgeCondition,
    HardwareCapabilityMatrix,
    IntentKind,
    IntentVector,
    LedgerAssertion,
    ResourceEstimate,
    ResourceMeasurement,
    SpecialistAvailability,
    SpecialistSlot,
    StaceyCore,
    StaceyCoreConfig,
    StaceyCoreInputError,
    StaceyOutputError,
    TaskDependencyEdge,
    TaskDependencyGraph,
    TaskGraphNode,
    UnifiedContextIngress,
    enforce_clarification_threshold,
    initialize_stacey_core,
    load_model_checkpoint,
    parse_decision_jsonl,
    save_model_checkpoint,
)
from models.stacey.core.checkpoint import StaceyCheckpointError
from models.stacey.core.config import STACEY_CORE_ARCHITECTURE_ID, STACEY_CORE_TOKENIZER_ID
from models.stacey.core.inputs import StaceyIngressError
from models.stacey.core.tokens import BYTE_VOCABULARY_SIZE, ByteTokenCodec
from substrate.contracts import ScopeVector
from training.stacey import StaceyTrainingError, run_synthetic_smoke_step, teacher_forcing_loss


SCOPE = ScopeVector("tenant-stacey", "user-stacey", "project-stacey", "workspace-stacey")
SPECIALIST_DIGEST = "a" * 64


def make_config(seed_profile: str = "small-smoke") -> StaceyCoreConfig:
    profiles = {
        "small-smoke": (32, 4, 1, 1, 64, 4096, 4096, 0.0),
        "larger-smoke": (48, 4, 2, 2, 96, 4096, 4096, 0.1),
    }
    dimension, heads, encoder_layers, decoder_layers, feedforward, input_limit, output_limit, dropout = profiles[
        seed_profile
    ]
    return StaceyCoreConfig(
        architecture_id=STACEY_CORE_ARCHITECTURE_ID,
        tokenizer_id=STACEY_CORE_TOKENIZER_ID,
        vocabulary_size=BYTE_VOCABULARY_SIZE,
        model_dimension=dimension,
        attention_heads=heads,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        feedforward_dimension=feedforward,
        maximum_input_tokens=input_limit,
        maximum_output_tokens=output_limit,
        dropout_probability=dropout,
    )


def make_ingress() -> UnifiedContextIngress:
    return UnifiedContextIngress(
        protocol_version="1.0",
        transaction_id="tx-stacey-001",
        correlation_id="turn-stacey-001",
        ingress_timestamp_ns=1_800_000_000_000_000_000,
        intent_vector=IntentVector(
            IntentKind.USER_LANGUAGE,
            "Inspect the records and summarize the result",
        ),
        scope_vector=SCOPE,
        canonical_state_assertions=(
            LedgerAssertion(
                assertion_id="assertion-project-state",
                content_summary="Project contains reviewed records",
                provenance_sha256="b" * 64,
                last_observed_state="VERIFIED",
            ),
        ),
        hardware_capability_matrix=HardwareCapabilityMatrix(
            resources=(ResourceMeasurement("accelerator-memory", 3_000_000_000, 1_800_000_000_000_000_000),),
            active_leases=(ActiveResourceLease("lease-existing", "accelerator-memory", 100_000_000),),
            available_specialist_slots=(
                SpecialistSlot(
                    block_id="BLOCK_10_STRUCTURAL_INGRESS",
                    capability_id="data.extract",
                    status=SpecialistAvailability.AVAILABLE,
                    resource_domain="accelerator-memory",
                    estimated_required_bytes=200_000_000,
                    artifact_sha256=SPECIALIST_DIGEST,
                    input_modalities=("structured-data",),
                ),
                SpecialistSlot(
                    block_id="BLOCK_12_LINGUISTIC_COPY",
                    capability_id="text.summarize",
                    status=SpecialistAvailability.AVAILABLE,
                    resource_domain="accelerator-memory",
                    estimated_required_bytes=150_000_000,
                    artifact_sha256="c" * 64,
                    input_modalities=("text",),
                ),
            ),
        ),
    )


def make_decision(ingress: UnifiedContextIngress) -> CoreDecisionEnvelope:
    return CoreDecisionEnvelope(
        protocol_version="1.0",
        transaction_id=ingress.transaction_id,
        correlation_id=ingress.correlation_id,
        assigned_block_id="BLOCK_0_CORE",
        predicted_consequences_summary="Extract records before preparing the summary.",
        calibration_metrics=CalibrationMetrics(0.8),
        declarative_intent_action="DECOMPOSE_TASK_GRAPH",
        scope_vector=ingress.scope_vector,
        task_dependency_graph=TaskDependencyGraph(
            nodes=(
                TaskGraphNode("extract", "data.extract", "BLOCK_10_STRUCTURAL_INGRESS"),
                TaskGraphNode("summarize", "text.summarize", "BLOCK_12_LINGUISTIC_COPY"),
            ),
            edges=(TaskDependencyEdge("extract", "summarize", EdgeCondition.ON_SUCCESS),),
            resource_estimates=(
                ResourceEstimate("extract", "accelerator-memory", 200_000_000),
                ResourceEstimate("summarize", "accelerator-memory", 150_000_000),
            ),
        ),
        clarification=ClarificationDirective(False, ()),
    )


class StaceyInputContractTests(unittest.TestCase):
    def test_byte_codec_roundtrips_unicode_without_external_vocabulary(self) -> None:
        codec = ByteTokenCodec()
        text = "Stacey: café, 東京"

        self.assertEqual(codec.decode(codec.encode(text)), text)
        self.assertEqual(codec.vocabulary_size, 259)

    def test_ingress_serializes_exactly_the_four_defined_matrices(self) -> None:
        payload = make_ingress().to_payload()

        self.assertEqual(
            set(payload),
            {
                "protocol_version",
                "transaction_id",
                "correlation_id",
                "ingress_timestamp_ns",
                "intent_vector",
                "scope_vector",
                "canonical_state_assertions",
                "hardware_capability_matrix",
            },
        )
        self.assertEqual(
            set(payload["hardware_capability_matrix"]),
            {"resources", "active_leases", "available_specialist_slots"},
        )

    def test_ingress_rejects_incomplete_trusted_scope_and_bad_provenance(self) -> None:
        with self.assertRaises(ValueError):
            UnifiedContextIngress(
                protocol_version="1.0",
                transaction_id="tx",
                correlation_id="turn",
                ingress_timestamp_ns=1,
                intent_vector=IntentVector(IntentKind.USER_LANGUAGE, "Do work"),
                scope_vector=ScopeVector("tenant", "user", "project", " "),
                canonical_state_assertions=(),
                hardware_capability_matrix=HardwareCapabilityMatrix((), (), ()),
            )

        with self.assertRaises(StaceyIngressError):
            LedgerAssertion("assertion", "summary", "not-a-hash", "VERIFIED")


class StaceyOutputContractTests(unittest.TestCase):
    def test_structured_decision_roundtrips_as_one_jsonl_record(self) -> None:
        ingress = make_ingress()
        decision = make_decision(ingress)

        parsed = parse_decision_jsonl(decision.to_jsonl(), ingress)

        self.assertEqual(parsed, decision)
        self.assertEqual(parsed.scope_vector, ingress.scope_vector)
        self.assertEqual(len(parsed.task_dependency_graph.nodes), 2)

    def test_output_rejects_scope_mutation(self) -> None:
        ingress = make_ingress()
        payload = make_decision(ingress).to_payload()
        payload["declarative_intent"]["scope_vector"]["tenant_id"] = "other-tenant"

        with self.assertRaises(StaceyOutputError):
            parse_decision_jsonl(json.dumps(payload), ingress)

    def test_output_rejects_unavailable_capability(self) -> None:
        ingress = make_ingress()
        payload = make_decision(ingress).to_payload()
        payload["declarative_intent"]["task_dependency_graph"]["nodes"][0]["capability_id"] = "unregistered.slot"

        with self.assertRaises(StaceyOutputError):
            parse_decision_jsonl(json.dumps(payload), ingress)

    def test_context_aware_revision_avoids_failed_specialist_and_selects_alternative(self) -> None:
        ingress = make_ingress()
        failed_slot = replace(
            ingress.hardware_capability_matrix.available_specialist_slots[0],
            status=SpecialistAvailability.QUARANTINED,
        )
        alternative_slot = SpecialistSlot(
            block_id="BLOCK_9_ALGORITHMIC_CODER",
            capability_id="data.extract.alternative",
            status=SpecialistAvailability.AVAILABLE,
            resource_domain="accelerator-memory",
            estimated_required_bytes=250_000_000,
            artifact_sha256="d" * 64,
            input_modalities=("structured-data",),
        )
        revised_ingress = replace(
            ingress,
            canonical_state_assertions=(
                *ingress.canonical_state_assertions,
                LedgerAssertion(
                    assertion_id="specialist-failure-event",
                    content_summary="data.extract failed validation and was quarantined",
                    provenance_sha256="e" * 64,
                    last_observed_state="SPECIALIST_FAILED",
                ),
            ),
            hardware_capability_matrix=replace(
                ingress.hardware_capability_matrix,
                available_specialist_slots=(
                    failed_slot,
                    ingress.hardware_capability_matrix.available_specialist_slots[1],
                    alternative_slot,
                ),
            ),
        )
        revised_decision = CoreDecisionEnvelope(
            protocol_version="1.0",
            transaction_id=revised_ingress.transaction_id,
            correlation_id=revised_ingress.correlation_id,
            assigned_block_id="BLOCK_0_CORE",
            predicted_consequences_summary="Use the alternative extractor after the prior failure.",
            calibration_metrics=CalibrationMetrics(0.74),
            declarative_intent_action="DECOMPOSE_TASK_GRAPH",
            scope_vector=revised_ingress.scope_vector,
            task_dependency_graph=TaskDependencyGraph(
                nodes=(
                    TaskGraphNode(
                        "extract-retry",
                        "data.extract.alternative",
                        "BLOCK_9_ALGORITHMIC_CODER",
                    ),
                ),
                edges=(),
                resource_estimates=(
                    ResourceEstimate("extract-retry", "accelerator-memory", 250_000_000),
                ),
            ),
            clarification=ClarificationDirective(False, ()),
        )

        accepted = parse_decision_jsonl(revised_decision.to_jsonl(), revised_ingress)

        self.assertEqual(
            accepted.task_dependency_graph.nodes[0].capability_id,
            "data.extract.alternative",
        )
        self.assertNotIn(
            "data.extract",
            {node.capability_id for node in accepted.task_dependency_graph.nodes},
        )

    def test_context_aware_revision_avoids_failed_specialist_and_uses_available_alternative(self) -> None:
        initial_ingress = make_ingress()
        failed_slot = replace(
            initial_ingress.hardware_capability_matrix.available_specialist_slots[0],
            status=SpecialistAvailability.QUARANTINED,
        )
        alternate_slot = SpecialistSlot(
            block_id="BLOCK_9_ALGORITHMIC_CODER",
            capability_id="data.extract.alternative",
            status=SpecialistAvailability.AVAILABLE,
            resource_domain="accelerator-memory",
            estimated_required_bytes=250_000_000,
            artifact_sha256="d" * 64,
            input_modalities=("structured-data",),
        )
        revised_ingress = replace(
            initial_ingress,
            canonical_state_assertions=(
                *initial_ingress.canonical_state_assertions,
                LedgerAssertion(
                    assertion_id="specialist-failure-event",
                    content_summary="data.extract failed validation and is quarantined",
                    provenance_sha256="e" * 64,
                    last_observed_state="SPECIALIST_FAILED",
                ),
            ),
            hardware_capability_matrix=replace(
                initial_ingress.hardware_capability_matrix,
                available_specialist_slots=(
                    failed_slot,
                    initial_ingress.hardware_capability_matrix.available_specialist_slots[1],
                    alternate_slot,
                ),
            ),
        )
        revised_decision = CoreDecisionEnvelope(
            protocol_version="1.0",
            transaction_id=revised_ingress.transaction_id,
            correlation_id=revised_ingress.correlation_id,
            assigned_block_id="BLOCK_0_CORE",
            predicted_consequences_summary="Use the alternate extractor after the prior capability failure.",
            calibration_metrics=CalibrationMetrics(0.74),
            declarative_intent_action="DECOMPOSE_TASK_GRAPH",
            scope_vector=revised_ingress.scope_vector,
            task_dependency_graph=TaskDependencyGraph(
                nodes=(TaskGraphNode("extract-retry", "data.extract.alternative", "BLOCK_9_ALGORITHMIC_CODER"),),
                edges=(),
                resource_estimates=(ResourceEstimate("extract-retry", "accelerator-memory", 250_000_000),),
            ),
            clarification=ClarificationDirective(False, ()),
        )

        accepted = parse_decision_jsonl(revised_decision.to_jsonl(), revised_ingress)

        self.assertEqual(
            accepted.task_dependency_graph.nodes[0].capability_id,
            "data.extract.alternative",
        )
        self.assertNotIn(
            "data.extract",
            {node.capability_id for node in accepted.task_dependency_graph.nodes},
        )

    def test_output_rejects_cycles_and_missing_resource_estimates(self) -> None:
        ingress = make_ingress()
        payload = make_decision(ingress).to_payload()
        graph = payload["declarative_intent"]["task_dependency_graph"]
        graph["edges"].append(
            {"source_step_id": "summarize", "target_step_id": "extract", "condition": "ON_SUCCESS"}
        )
        with self.assertRaises(StaceyOutputError):
            parse_decision_jsonl(json.dumps(payload), ingress)

        payload = make_decision(ingress).to_payload()
        payload["declarative_intent"]["task_dependency_graph"]["resource_estimates"].pop()
        with self.assertRaises(StaceyOutputError):
            parse_decision_jsonl(json.dumps(payload), ingress)

    def test_clarification_output_cannot_dispatch_work(self) -> None:
        ingress = make_ingress()
        payload = make_decision(ingress).to_payload()
        payload["declarative_intent"]["clarification"] = {
            "required": True,
            "reason_codes": ["MISSING_REQUIRED_CONTEXT"],
        }

        with self.assertRaises(StaceyOutputError):
            parse_decision_jsonl(json.dumps(payload), ingress)

    def test_caller_confidence_threshold_requires_clarification_before_dispatch(self) -> None:
        ingress = make_ingress()
        confident_decision = make_decision(ingress)
        enforce_clarification_threshold(confident_decision, minimum_confidence=0.5)
        low_confidence = CoreDecisionEnvelope(
            protocol_version=confident_decision.protocol_version,
            transaction_id=confident_decision.transaction_id,
            correlation_id=confident_decision.correlation_id,
            assigned_block_id=confident_decision.assigned_block_id,
            predicted_consequences_summary=confident_decision.predicted_consequences_summary,
            calibration_metrics=CalibrationMetrics(0.2),
            declarative_intent_action=confident_decision.declarative_intent_action,
            scope_vector=confident_decision.scope_vector,
            task_dependency_graph=confident_decision.task_dependency_graph,
            clarification=ClarificationDirective(False, ()),
        )

        with self.assertRaises(StaceyOutputError):
            enforce_clarification_threshold(low_confidence, minimum_confidence=0.5)

    def test_output_rejects_multiple_lines_and_duplicate_json_keys(self) -> None:
        ingress = make_ingress()
        line = make_decision(ingress).to_jsonl()

        with self.assertRaises(StaceyOutputError):
            parse_decision_jsonl(line + "\n{}", ingress)
        with self.assertRaises(StaceyOutputError):
            parse_decision_jsonl('{"protocol_version":"1.0","protocol_version":"1.0"}', ingress)

    def test_shared_schema_accepts_stacey_ingress_and_decision(self) -> None:
        if importlib.util.find_spec("jsonschema") is None:
            self.skipTest("jsonschema package not installed")
        import jsonschema

        schema = json.loads(
            Path(__file__).parents[2].joinpath("schemas", "contracts.schema.json").read_text()
        )
        jsonschema.Draft202012Validator.check_schema(schema)
        definitions = schema["$defs"]
        ingress_validator = jsonschema.Draft202012Validator(
            {
                "$schema": schema["$schema"],
                "$defs": definitions,
                "$ref": "#/$defs/unifiedContextIngress",
            }
        )
        decision_validator = jsonschema.Draft202012Validator(
            {
                "$schema": schema["$schema"],
                "$defs": definitions,
                "$ref": "#/$defs/coreDecisionEnvelope",
            }
        )
        ingress = make_ingress()
        self.assertEqual(list(ingress_validator.iter_errors(ingress.to_payload())), [])
        self.assertEqual(list(decision_validator.iter_errors(make_decision(ingress).to_payload())), [])


class StaceyNeuralSkeletonTests(unittest.TestCase):
    def test_configurable_encoder_decoder_forward_loss_and_gradients(self) -> None:
        model = initialize_stacey_core(make_config(), random_seed=19)
        source_ids = model.encode_ingresses((make_ingress(),))
        codec = ByteTokenCodec()
        target_ids = torch.tensor(
            [codec.encode('{"action":"DECOMPOSE_TASK_GRAPH"}')], dtype=torch.long
        )
        logits = model(source_ids, target_ids[:, :-1])
        loss = teacher_forcing_loss(logits, target_ids[:, 1:])
        loss.backward()

        self.assertEqual(logits.shape, (1, target_ids.shape[1] - 1, BYTE_VOCABULARY_SIZE))
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_synthetic_smoke_training_step_updates_random_core_weights(self) -> None:
        model = initialize_stacey_core(make_config(), random_seed=29)
        source_ids = model.encode_ingresses((make_ingress(),))
        target_ids = torch.tensor(
            [ByteTokenCodec().encode('{"action":"DECOMPOSE_TASK_GRAPH"}')], dtype=torch.long
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        report = run_synthetic_smoke_step(model, source_ids, target_ids, optimizer)

        self.assertGreater(report.loss, 0.0)
        self.assertGreater(report.gradient_norm, 0.0)
        self.assertGreater(report.trainable_parameter_count, 0)
        self.assertGreater(report.updated_parameter_count, 0)

    def test_training_objective_rejects_empty_targets_and_shape_mismatch(self) -> None:
        with self.assertRaises(StaceyTrainingError):
            teacher_forcing_loss(torch.zeros((1, 2, BYTE_VOCABULARY_SIZE)), torch.zeros((1, 2), dtype=torch.long))
        with self.assertRaises(StaceyTrainingError):
            teacher_forcing_loss(
                torch.zeros((1, 2, BYTE_VOCABULARY_SIZE)),
                torch.tensor([[3]], dtype=torch.long),
            )

    def test_random_initialization_is_reproducible_given_explicit_seed(self) -> None:
        first = initialize_stacey_core(make_config(), random_seed=23)
        second = initialize_stacey_core(make_config(), random_seed=23)

        for name, parameter in first.state_dict().items():
            self.assertTrue(torch.equal(parameter, second.state_dict()[name]), name)

    def test_candidate_profiles_report_parameter_count_and_raw_weight_estimates(self) -> None:
        compact = initialize_stacey_core(make_config("small-smoke"), random_seed=47)
        wider = initialize_stacey_core(make_config("larger-smoke"), random_seed=47)

        self.assertGreater(wider.parameter_count, compact.parameter_count)
        self.assertEqual(compact.raw_weight_storage_bytes(bits_per_parameter=8), compact.parameter_count)
        self.assertGreater(
            compact.raw_weight_storage_bytes(bits_per_parameter=16),
            compact.raw_weight_storage_bytes(bits_per_parameter=4),
        )
        with self.assertRaises(ValueError):
            compact.raw_weight_storage_bytes(bits_per_parameter=0)

    def test_greedy_generation_is_bounded_and_returns_token_ids(self) -> None:
        model = initialize_stacey_core(make_config(), random_seed=31)
        source_ids = model.encode_ingresses((make_ingress(),))

        generated = model.generate_token_ids(source_ids, maximum_new_tokens=5)

        self.assertEqual(generated.shape[0], 1)
        self.assertLessEqual(generated.shape[1], 5)
        self.assertEqual(generated.dtype, torch.long)

    def test_model_checkpoint_roundtrips_and_binds_digest(self) -> None:
        model = initialize_stacey_core(make_config(), random_seed=37)
        source_ids = model.encode_ingresses((make_ingress(),))
        decoder_ids = torch.tensor([[1, 86, 75]], dtype=torch.long)
        expected = model(source_ids, decoder_ids).detach()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "stacey-core.pt"
            digest = save_model_checkpoint(model, checkpoint_path, file_mode=0o600)
            restored = load_model_checkpoint(
                checkpoint_path,
                expected_sha256=digest,
                map_location="cpu",
            )
            actual = restored(source_ids, decoder_ids).detach()

        self.assertTrue(torch.equal(expected, actual))

    def test_checkpoint_rejects_wrong_digest(self) -> None:
        model = initialize_stacey_core(make_config(), random_seed=41)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "stacey-core.pt"
            save_model_checkpoint(model, checkpoint_path, file_mode=0o600)
            with self.assertRaises(StaceyCheckpointError):
                load_model_checkpoint(checkpoint_path, expected_sha256="f" * 64, map_location="cpu")

    def test_config_does_not_accept_incompatible_shape_or_hidden_architecture(self) -> None:
        with self.assertRaises(ValueError):
            StaceyCoreConfig(
                architecture_id=STACEY_CORE_ARCHITECTURE_ID,
                tokenizer_id=STACEY_CORE_TOKENIZER_ID,
                vocabulary_size=BYTE_VOCABULARY_SIZE,
                model_dimension=30,
                attention_heads=8,
                encoder_layers=1,
                decoder_layers=1,
                feedforward_dimension=64,
                maximum_input_tokens=64,
                maximum_output_tokens=64,
                dropout_probability=0.0,
            )
        with self.assertRaises(ValueError):
            StaceyCoreConfig(
                architecture_id="unreviewed-architecture",
                tokenizer_id=STACEY_CORE_TOKENIZER_ID,
                vocabulary_size=BYTE_VOCABULARY_SIZE,
                model_dimension=32,
                attention_heads=4,
                encoder_layers=1,
                decoder_layers=1,
                feedforward_dimension=64,
                maximum_input_tokens=64,
                maximum_output_tokens=64,
                dropout_probability=0.0,
            )


if __name__ == "__main__":
    unittest.main()