from __future__ import annotations

import hashlib
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from models.stacey.core.inputs import (
    HardwareCapabilityMatrix,
    IntentKind,
    IntentVector,
    UnifiedContextIngress,
)
from models.stacey.core.checkpoint import save_model_checkpoint
from models.stacey.core.config import (
    STACEY_CORE_ARCHITECTURE_ID,
    STACEY_CORE_TOKENIZER_ID,
    StaceyCoreConfig,
)
from models.stacey.core.network import StaceyCore
from models.stacey.core.tokens import BYTE_VOCABULARY_SIZE
from models.stacey.core.outputs import (
    CalibrationMetrics,
    ClarificationDirective,
    CoreDecisionEnvelope,
    EdgeCondition,
    ResourceEstimate,
    SystemInspectionDirective,
    SystemCapabilityRequirement,
    TaskDependencyEdge,
    TaskDependencyGraph,
    TaskGraphNode,
)
from substrate.contracts import Intent, ScopeVector
from src.swarm_core.collective_scheduler import (
    CapabilityContract,
    CapabilityContractRegistry,
    CapabilityResponse,
    CollectiveExecutionReport,
    CollectiveScheduler,
    StepState,
)
from src.swarm_core.collective_runtime import (
    CollectiveRuntimeError,
    StaceyCollectiveRuntime,
)
from src.swarm_core.response_composer import UserResponseStatus
from src.swarm_core.identity import (
    Ed25519ScopeAuthorizationVerifier,
    SignedScopeGrant,
    intent_authorization_digest,
    scope_grant_message,
)
from src.swarm_core.model_catalog import ModelRole
from src.swarm_core.model_lifecycle import ArtifactKind, ModelArtifactManifest
from src.swarm_core.stacey_core_backend import (
    StaceyCoreBackendError,
    StaceyCoreRuntimeBackend,
    stacey_core_config_sha256,
    stacey_core_tokenizer_sha256,
)


SCOPE = ScopeVector("tenant-a", "user-a", "project-a", "workspace-a")
CORE_ARTIFACT_SHA256 = hashlib.sha256(b"approved-core-artifact").hexdigest()
SPECIALIST_ARTIFACT_SHA256 = hashlib.sha256(b"approved-specialist-artifact").hexdigest()


class FakeModelLease:
    def __init__(self, response, events: list[str], artifact_sha256: str = SPECIALIST_ARTIFACT_SHA256) -> None:
        self.response = response
        self.events = events
        self.artifact_sha256 = artifact_sha256

    def infer(self, request):
        self.events.append(f"infer:{request.step_id}")
        return self.response(request)

    def __enter__(self):
        self.events.append("lease-enter")
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.events.append("lease-exit")


class FakeLifecycleManager:
    def __init__(self, responses) -> None:
        self.responses = responses
        self.events: list[str] = []
        self.acquire_calls: list[str] = []
        self.admission_declarations: list[tuple[str, dict[str, object]]] = []
        self.release_id = "release-a"
        self.rotate_after_first_acquire = False

    def active_release(self):
        return SimpleNamespace(release_id=self.release_id)

    def core_artifact_sha256_for_release(self, release_id: str) -> str:
        return CORE_ARTIFACT_SHA256

    def acquire_from_active_release(self, capability_id: str, *, release_id: str, **declaration):
        self.acquire_calls.append(capability_id)
        self.admission_declarations.append((capability_id, {"release_id": release_id, **declaration}))
        if self.rotate_after_first_acquire and len(self.acquire_calls) == 1:
            self.release_id = "release-b"
        return FakeModelLease(self.responses[capability_id], self.events)


class FakeValidator:
    def __init__(self, events: list[str], fail: bool = False) -> None:
        self.events = events
        self.fail = fail

    def validate(self, request, response) -> None:
        self.events.append(f"validate:{request.step_id}")
        if self.fail:
            raise ValueError("contract failure")


class FakeReviewer:
    def __init__(self, events: list[str], *, fail_at: str | None = None) -> None:
        self.events = events
        self.fail_at = fail_at

    def verify_adversarial(self, request, response) -> str:
        self.events.append(f"block-6:{request.step_id}")
        if self.fail_at == "block-6":
            raise ValueError("critic rejected")
        return f"critic:{request.step_id}"

    def verify_epistemic(self, request, response) -> str:
        self.events.append(f"block-8:{request.step_id}")
        if self.fail_at == "block-8":
            raise ValueError("auditor rejected")
        return f"auditor:{request.step_id}"


def make_decision(
    nodes: tuple[TaskGraphNode, ...],
    edges: tuple[TaskDependencyEdge, ...] = (),
    *,
    clarification: bool = False,
) -> CoreDecisionEnvelope:
    return CoreDecisionEnvelope(
        protocol_version="1.0",
        transaction_id="tx-scheduler",
        correlation_id="turn-scheduler",
        assigned_block_id="BLOCK_0_CORE",
        predicted_consequences_summary="Run a reviewed task graph.",
        calibration_metrics=CalibrationMetrics(0.95),
        declarative_intent_action="DECOMPOSE_TASK_GRAPH",
        scope_vector=SCOPE,
        task_dependency_graph=TaskDependencyGraph(
            nodes=nodes,
            edges=edges,
            resource_estimates=tuple(ResourceEstimate(node.step_id, "host-memory", 10) for node in nodes),
        ),
        clarification=ClarificationDirective(
            clarification,
            ("AMBIGUOUS_INTENT",) if clarification else (),
        ),
    )


def make_ingress() -> UnifiedContextIngress:
    return UnifiedContextIngress(
        protocol_version="1.0",
        transaction_id="tx-scheduler",
        correlation_id="turn-scheduler",
        ingress_timestamp_ns=1,
        intent_vector=IntentVector(IntentKind.USER_LANGUAGE, "summarize the notes"),
        scope_vector=SCOPE,
        canonical_state_assertions=(),
        hardware_capability_matrix=HardwareCapabilityMatrix((), (), ()),
    )


def make_runtime_authority(ingress: UnifiedContextIngress):
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    verifier = Ed25519ScopeAuthorizationVerifier(
        public_key,
        audience="collective-runtime-test",
        clock=lambda: 100,
    )
    intent = Intent(
        transaction_id=ingress.transaction_id,
        correlation_id=ingress.correlation_id,
        action="ROUTE_TASK",
        goal=ingress.to_canonical_json(),
        scope=ingress.scope_vector,
    )
    unsigned = SignedScopeGrant(
        scope=ingress.scope_vector,
        action="ROUTE_TASK",
        transaction_id=ingress.transaction_id,
        correlation_id=ingress.correlation_id,
        audience="collective-runtime-test",
        issued_at=90,
        expires_at=110,
        grant_id=f"grant:{ingress.transaction_id}",
        request_sha256=intent_authorization_digest(intent),
        signature=b"",
    )
    grant = replace(unsigned, signature=private_key.sign(scope_grant_message(unsigned)))
    return verifier, grant


class FakeCoreLease:
    def __init__(self, decision, artifact_sha256: str) -> None:
        self.decision = decision
        self.seen_ingresses = []
        self.artifact_sha256 = artifact_sha256
        self.released = False

    def infer(self, ingress):
        self.seen_ingresses.append(ingress)
        return self.decision

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.released = True


class FakeCoreLifecycleManager:
    def __init__(self, decision, artifact_sha256: str = CORE_ARTIFACT_SHA256) -> None:
        self.lease = FakeCoreLease(decision, artifact_sha256)
        self.acquire_calls = []

    def acquire_core_from_release(self, release_id: str):
        self.acquire_calls.append(release_id)
        return self.lease


class FakeCollectiveDispatcher:
    def __init__(self) -> None:
        self.calls = []

    def active_release_snapshot(self):
        return "release-test", CORE_ARTIFACT_SHA256

    def execute(self, decision, *, request_payload, release_id):
        self.calls.append((decision, request_payload, release_id))
        return CollectiveExecutionReport(
            decision.transaction_id,
            decision.correlation_id,
            release_id,
            decision.scope_vector,
            decision.clarification.required,
            (),
        )


def response_for(capability_id: str, block_id: str, payload: bytes = b"ok"):
    def make_response(request):
        return CapabilityResponse(
            transaction_id=request.transaction_id,
            correlation_id=request.correlation_id,
            step_id=request.step_id,
            capability_id=capability_id,
            target_block_id=block_id,
            contract_version="1.0",
            scope=request.scope,
            payload=payload,
            provenance_references=(f"source:{request.step_id}",),
        )
    return make_response


class CollectiveSchedulerTests(unittest.TestCase):
    def make_scheduler(self, definitions, lifecycle, reviewer):
        events = lifecycle.events
        contracts = tuple(
            CapabilityContract(
                capability_id=capability,
                target_block_id=block,
                version="1.0",
                schema_sha256=hashlib.sha256(capability.encode()).hexdigest(),
                maximum_request_bytes=1024,
                maximum_response_bytes=1024,
            )
            for capability, block in definitions
        )
        validators = {
            capability: FakeValidator(events)
            for capability, _ in definitions
        }
        return CollectiveScheduler(
            lifecycle_manager=lifecycle,
            contracts=CapabilityContractRegistry(contracts, validators),
            reviewer=reviewer,
        )

    def test_runs_steps_and_applies_block6_then_block8_before_exposing_outputs(self) -> None:
        lifecycle = FakeLifecycleManager({"cap.alpha": response_for("cap.alpha", "BLOCK_9")})
        scheduler = self.make_scheduler(
            (("cap.alpha", "BLOCK_9"),),
            lifecycle,
            FakeReviewer(lifecycle.events),
        )
        decision = make_decision((TaskGraphNode("step-a", "cap.alpha", "BLOCK_9"),))

        report = scheduler.execute(decision, request_payload=b"request")

        self.assertEqual(report.outputs[0].state, StepState.SUCCEEDED)
        self.assertEqual(report.outputs[0].payload, b"ok")
        self.assertEqual(report.outputs[0].critic_reference, "critic:step-a")
        self.assertEqual(report.outputs[0].auditor_reference, "auditor:step-a")
        self.assertEqual(report.outputs[0].artifact_sha256, SPECIALIST_ARTIFACT_SHA256)
        self.assertEqual(
            lifecycle.admission_declarations,
            [
                (
                    "cap.alpha",
                    {
                        "release_id": "release-a",
                        "contract_version": "1.0",
                        "contract_sha256": hashlib.sha256(b"cap.alpha").hexdigest(),
                        "resource_domain": "host-memory",
                        "estimated_required_bytes": 10,
                    },
                )
            ],
        )
        self.assertEqual(
            lifecycle.events,
            [
                "lease-enter",
                "infer:step-a",
                "lease-exit",
                "validate:step-a",
                "block-6:step-a",
                "block-8:step-a",
            ],
        )

    def test_conditional_edges_run_success_failure_and_always_paths(self) -> None:
        lifecycle = FakeLifecycleManager(
            {
                "cap.root": response_for("wrong-capability", "BLOCK_1"),
                "cap.on-success": response_for("cap.on-success", "BLOCK_9"),
                "cap.on-failure": response_for("cap.on-failure", "BLOCK_6"),
                "cap.always": response_for("cap.always", "BLOCK_8"),
            }
        )
        scheduler = self.make_scheduler(
            (
                ("cap.root", "BLOCK_1"),
                ("cap.on-success", "BLOCK_9"),
                ("cap.on-failure", "BLOCK_6"),
                ("cap.always", "BLOCK_8"),
            ),
            lifecycle,
            FakeReviewer(lifecycle.events),
        )
        lifecycle.rotate_after_first_acquire = True
        decision = make_decision(
            (
                TaskGraphNode("root", "cap.root", "BLOCK_1"),
                TaskGraphNode("success", "cap.on-success", "BLOCK_9"),
                TaskGraphNode("failure", "cap.on-failure", "BLOCK_6"),
                TaskGraphNode("always", "cap.always", "BLOCK_8"),
            ),
            (
                TaskDependencyEdge("root", "success", EdgeCondition.ON_SUCCESS),
                TaskDependencyEdge("root", "failure", EdgeCondition.ON_FAILURE),
                TaskDependencyEdge("root", "always", EdgeCondition.ALWAYS),
            ),
        )

        report = scheduler.execute(decision, request_payload=b"request")
        by_step = {output.step_id: output for output in report.outputs}

        self.assertEqual(by_step["root"].state, StepState.FAILED)
        self.assertEqual(by_step["success"].state, StepState.SKIPPED)
        self.assertEqual(by_step["failure"].state, StepState.SUCCEEDED)
        self.assertEqual(by_step["always"].state, StepState.SUCCEEDED)
        self.assertEqual(
            lifecycle.acquire_calls,
            ["cap.root", "cap.always", "cap.on-failure"],
        )
        self.assertEqual(
            [declaration[1]["release_id"] for declaration in lifecycle.admission_declarations],
            ["release-a", "release-a", "release-a"],
        )

    def test_clarification_does_not_acquire_any_specialist(self) -> None:
        lifecycle = FakeLifecycleManager({})
        scheduler = self.make_scheduler((), lifecycle, FakeReviewer(lifecycle.events))
        decision = make_decision((), clarification=True)

        report = scheduler.execute(decision, request_payload=b"request")

        self.assertTrue(report.clarification_required)
        self.assertEqual(report.outputs, ())
        self.assertEqual(lifecycle.acquire_calls, [])

    def test_block6_failure_prevents_block8_and_lease_always_releases(self) -> None:
        lifecycle = FakeLifecycleManager({"cap.alpha": response_for("cap.alpha", "BLOCK_9")})
        scheduler = self.make_scheduler(
            (("cap.alpha", "BLOCK_9"),),
            lifecycle,
            FakeReviewer(lifecycle.events, fail_at="block-6"),
        )
        decision = make_decision((TaskGraphNode("step-a", "cap.alpha", "BLOCK_9"),))

        report = scheduler.execute(decision, request_payload=b"request")

        self.assertEqual(report.outputs[0].state, StepState.FAILED)
        self.assertIn("lease-exit", lifecycle.events)
        self.assertIn("block-6:step-a", lifecycle.events)
        self.assertNotIn("block-8:step-a", lifecycle.events)

    def test_request_over_limit_fails_before_loading_model(self) -> None:
        lifecycle = FakeLifecycleManager({})
        scheduler = self.make_scheduler(
            (("cap.alpha", "BLOCK_9"),),
            lifecycle,
            FakeReviewer(lifecycle.events),
        )
        decision = make_decision((TaskGraphNode("step-a", "cap.alpha", "BLOCK_9"),))

        report = scheduler.execute(decision, request_payload=b"x" * 2048)

        self.assertEqual(report.outputs[0].state, StepState.FAILED)
        self.assertEqual(lifecycle.acquire_calls, [])


class StaceyCollectiveRuntimeTests(unittest.TestCase):
    def test_core_decision_dispatches_with_canonical_typed_ingress(self) -> None:
        ingress = make_ingress()
        decision = make_decision(())
        core = FakeCoreLifecycleManager(decision)
        scheduler = FakeCollectiveDispatcher()
        verifier, grant = make_runtime_authority(ingress)
        runtime = StaceyCollectiveRuntime(
            core_lifecycle_manager=core,
            scheduler=scheduler,
            authorization_verifier=verifier,
        )

        result = runtime.execute(ingress, grant)

        self.assertIs(result.decision, decision)
        self.assertEqual(result.execution.transaction_id, ingress.transaction_id)
        self.assertEqual(result.response.status, UserResponseStatus.INCOMPLETE)
        self.assertEqual(core.lease.seen_ingresses, [ingress])
        self.assertEqual(core.acquire_calls, ["release-test"])
        self.assertTrue(core.lease.released)
        self.assertEqual(
            scheduler.calls,
            [(decision, ingress.to_canonical_json().encode("utf-8"), "release-test")],
        )

    def test_core_identity_or_scope_mismatch_fails_before_dispatch(self) -> None:
        ingress = make_ingress()
        verifier, grant = make_runtime_authority(ingress)
        valid_decision = make_decision(())
        mismatched_decisions = (
            replace(valid_decision, transaction_id="other-transaction"),
            replace(valid_decision, scope_vector=ScopeVector("tenant-b", "user-a", "project-a", "workspace-a")),
        )

        for decision in mismatched_decisions:
            with self.subTest(decision=decision):
                scheduler = FakeCollectiveDispatcher()
                runtime = StaceyCollectiveRuntime(
                    core_lifecycle_manager=FakeCoreLifecycleManager(decision),
                    scheduler=scheduler,
                    authorization_verifier=verifier,
                )

                with self.assertRaises(CollectiveRuntimeError):
                    runtime.execute(ingress, grant)
                self.assertEqual(scheduler.calls, [])

    def test_core_digest_mismatch_fails_before_inference(self) -> None:
        ingress = make_ingress()
        verifier, grant = make_runtime_authority(ingress)
        core = FakeCoreLifecycleManager(
            make_decision(()),
            hashlib.sha256(b"different-core").hexdigest(),
        )
        scheduler = FakeCollectiveDispatcher()
        runtime = StaceyCollectiveRuntime(
            core_lifecycle_manager=core,
            scheduler=scheduler,
            authorization_verifier=verifier,
        )

        with self.assertRaises(CollectiveRuntimeError):
            runtime.execute(ingress, grant)

        self.assertEqual(core.lease.seen_ingresses, [])
        self.assertTrue(core.lease.released)
        self.assertEqual(scheduler.calls, [])

    def test_tampered_ingress_fails_authorization_before_core_or_dispatch(self) -> None:
        ingress = make_ingress()
        verifier, grant = make_runtime_authority(ingress)
        altered_ingress = replace(
            ingress,
            intent_vector=replace(ingress.intent_vector, payload="different request"),
        )
        core = FakeCoreLifecycleManager(make_decision(()))
        scheduler = FakeCollectiveDispatcher()
        runtime = StaceyCollectiveRuntime(
            core_lifecycle_manager=core,
            scheduler=scheduler,
            authorization_verifier=verifier,
        )

        with self.assertRaises(PermissionError):
            runtime.execute(altered_ingress, grant)

        self.assertEqual(core.lease.seen_ingresses, [])
        self.assertEqual(scheduler.calls, [])

    def test_system_inspection_directive_returns_consent_request_without_dispatch(self) -> None:
        ingress = make_ingress()
        decision = replace(
            make_decision((), clarification=True),
            system_inspection=SystemInspectionDirective(
                "Check the authorized viewport and browser controls needed for this task.",
                ("viewport.capture", "dom.read"),
                (SystemCapabilityRequirement("vision.inspect", ("viewport-access",), (("host-memory", 1000),)),),
            ),
        )
        core = FakeCoreLifecycleManager(decision)
        scheduler = FakeCollectiveDispatcher()
        verifier, grant = make_runtime_authority(ingress)
        runtime = StaceyCollectiveRuntime(
            core_lifecycle_manager=core,
            scheduler=scheduler,
            authorization_verifier=verifier,
        )

        result = runtime.execute(ingress, grant)

        self.assertEqual(result.response.status, UserResponseStatus.SYSTEM_INSPECTION_CONSENT_REQUIRED)
        self.assertEqual(result.pending_inspection.probe_ids, ("viewport.capture", "dom.read"))
        self.assertEqual(result.pending_inspection.active_release_id, "release-test")
        self.assertEqual(result.pending_inspection.scope, ingress.scope_vector)
        self.assertEqual(scheduler.calls, [])


class StaceyCoreBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.directory = Path(self.temporary_directory.name)
        self.config = StaceyCoreConfig(
            architecture_id=STACEY_CORE_ARCHITECTURE_ID,
            tokenizer_id=STACEY_CORE_TOKENIZER_ID,
            vocabulary_size=BYTE_VOCABULARY_SIZE,
            model_dimension=8,
            attention_heads=2,
            encoder_layers=1,
            decoder_layers=1,
            feedforward_dimension=16,
            maximum_input_tokens=1024,
            maximum_output_tokens=64,
            dropout_probability=0.0,
        )
        self.checkpoint_path = self.directory / "core.ckpt"
        self.artifact_sha256 = save_model_checkpoint(
            StaceyCore(self.config),
            self.checkpoint_path,
            file_mode=0o600,
        )

    def make_manifest(self, *, config_sha256: str | None = None, tokenizer_sha256: str | None = None):
        return ModelArtifactManifest(
            artifact_id="stacey-core-test",
            version="0.1.0",
            kind=ArtifactKind.FULL_MODEL,
            role=ModelRole.WORLD_MODEL_CORE,
            relative_path="core.ckpt",
            artifact_sha256=self.artifact_sha256,
            artifact_size_bytes=self.checkpoint_path.stat().st_size,
            capability_ids=("core.reason",),
            backend_id=StaceyCoreRuntimeBackend.backend_id,
            architecture_id=STACEY_CORE_ARCHITECTURE_ID,
            config_sha256=config_sha256 or stacey_core_config_sha256(self.config),
            tokenizer_sha256=tokenizer_sha256 or stacey_core_tokenizer_sha256(),
            license_reference="stacey-owned",
            provenance_reference="scratch-run:test",
            training_lineage="STACEY_SCRATCH",
            evaluation_reference="evaluation:test",
            approval_reference="approval:test",
            resource_profile_name="core-resident",
            resource_domain="host-memory",
            reserved_bytes=1024 * 1024,
        )

    def test_backend_loads_exact_checkpoint_and_confirms_cpu_release(self) -> None:
        backend = StaceyCoreRuntimeBackend(device="cpu")
        manifest = self.make_manifest()

        with self.checkpoint_path.open("rb") as checkpoint:
            handle = backend.load(manifest, checkpoint, base_handle=None)

        self.assertEqual(handle.adapter.artifact_sha256, manifest.artifact_sha256)
        self.assertEqual(handle.adapter.model.config, self.config)
        self.assertTrue(backend.unload(handle))
        self.assertTrue(backend.confirm_released(handle))

    def test_backend_rejects_config_or_tokenizer_digest_mismatch(self) -> None:
        backend = StaceyCoreRuntimeBackend(device="cpu")
        invalid_manifests = (
            self.make_manifest(config_sha256=hashlib.sha256(b"wrong-config").hexdigest()),
            self.make_manifest(tokenizer_sha256=hashlib.sha256(b"wrong-tokenizer").hexdigest()),
        )

        for manifest in invalid_manifests:
            with self.subTest(manifest=manifest):
                with self.checkpoint_path.open("rb") as checkpoint:
                    with self.assertRaises(StaceyCoreBackendError) as error:
                        backend.load(manifest, checkpoint, base_handle=None)
                self.assertTrue(error.exception.resources_released)


if __name__ == "__main__":
    unittest.main()
