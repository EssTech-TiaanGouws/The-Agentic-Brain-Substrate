from __future__ import annotations

import hashlib
import unittest
from dataclasses import replace

from models.stacey.core.inputs import HardwareCapabilityMatrix, IntentKind, IntentVector, UnifiedContextIngress
from models.stacey.core.outputs import (
    CalibrationMetrics,
    ClarificationDirective,
    CoreDecisionEnvelope,
    SystemCapabilityRequirement,
    SystemInspectionDirective,
    TaskDependencyGraph,
    parse_decision_jsonl,
    TaskGraphNode,
)
from substrate.contracts import ScopeVector
from src.swarm_core.collective_scheduler import (
    CapabilityOutput,
    CollectiveExecutionReport,
    StepState,
)
from src.swarm_core.response_composer import (
    ResponseCompositionError,
    StaceyResponseComposer,
    UserResponseStatus,
)


SCOPE = ScopeVector("tenant-a", "user-a", "project-a", "workspace-a")


def make_ingress() -> UnifiedContextIngress:
    return UnifiedContextIngress(
        protocol_version="1.0",
        transaction_id="tx-response",
        correlation_id="turn-response",
        ingress_timestamp_ns=10,
        intent_vector=IntentVector(IntentKind.USER_LANGUAGE, "summarize the report"),
        scope_vector=SCOPE,
        canonical_state_assertions=(),
        hardware_capability_matrix=HardwareCapabilityMatrix((), (), ()),
    )


def make_decision(*, clarification: bool = False) -> CoreDecisionEnvelope:
    ingress = make_ingress()
    nodes = () if clarification else (TaskGraphNode("step-a", "text.summarize", "BLOCK_12"),)
    return CoreDecisionEnvelope(
        protocol_version="1.0",
        transaction_id=ingress.transaction_id,
        correlation_id=ingress.correlation_id,
        assigned_block_id="BLOCK_0_CORE",
        predicted_consequences_summary="Return a verified summary.",
        calibration_metrics=CalibrationMetrics(0.9),
        declarative_intent_action="CLARIFY" if clarification else "DECOMPOSE_TASK_GRAPH",
        scope_vector=SCOPE,
        task_dependency_graph=TaskDependencyGraph(nodes, (), ()),
        clarification=ClarificationDirective(
            clarification,
            ("AMBIGUOUS_INTENT",) if clarification else (),
        ),
    )


class StaceyResponseComposerTests(unittest.TestCase):
    def test_core_inspection_directive_roundtrips_in_the_strict_decision_contract(self) -> None:
        ingress = make_ingress()
        decision = replace(
            make_decision(clarification=True),
            declarative_intent_action="DECOMPOSE_TASK_GRAPH",
            system_inspection=SystemInspectionDirective(
                "Check whether the authorized browser workflow is available.",
                ("viewport.capture", "dom.read"),
                (
                    SystemCapabilityRequirement(
                        "vision.inspect",
                        ("read-only", "viewport-access"),
                        (("host-memory", 4096),),
                    ),
                ),
            ),
        )

        parsed = parse_decision_jsonl(decision.to_jsonl(), ingress)

        self.assertEqual(parsed.system_inspection, decision.system_inspection)
        self.assertTrue(parsed.clarification.required)
        self.assertEqual(parsed.task_dependency_graph.nodes, ())

    def test_complete_response_uses_only_validated_status_not_payload_bytes(self) -> None:
        ingress = make_ingress()
        decision = make_decision()
        output = CapabilityOutput(
            "step-a",
            "text.summarize",
            StepState.SUCCEEDED,
            b"secret untrusted model output",
            None,
            ("source:reviewed",),
            "critic:checked",
            "auditor:checked",
            hashlib.sha256(b"approved-model").hexdigest(),
        )
        report = CollectiveExecutionReport(
            ingress.transaction_id,
            ingress.correlation_id,
            "release-response",
            SCOPE,
            False,
            (output,),
        )

        response = StaceyResponseComposer().compose(ingress, decision, report)

        self.assertEqual(response.status, UserResponseStatus.COMPLETE)
        self.assertEqual(response.completed_step_ids, ("step-a",))
        self.assertEqual(response.verification_references, ("critic:checked", "auditor:checked"))
        self.assertNotIn("secret untrusted model output", response.message)

    def test_clarification_returns_a_bounded_question_without_dispatch_metadata(self) -> None:
        ingress = make_ingress()
        decision = make_decision(clarification=True)
        report = CollectiveExecutionReport(
            ingress.transaction_id,
            ingress.correlation_id,
            "release-response",
            SCOPE,
            True,
            (),
        )

        response = StaceyResponseComposer().compose(ingress, decision, report)

        self.assertEqual(response.status, UserResponseStatus.CLARIFICATION_REQUIRED)
        self.assertEqual(response.message, "What would you like me to focus on?")
        self.assertEqual(response.verification_references, ())

    def test_failed_or_unbound_output_never_claims_completion(self) -> None:
        ingress = make_ingress()
        decision = make_decision()
        failed = CapabilityOutput(
            "step-a", "text.summarize", StepState.FAILED, None, "RuntimeError", (), None, None
        )
        report = CollectiveExecutionReport(
            ingress.transaction_id,
            ingress.correlation_id,
            "release-response",
            SCOPE,
            False,
            (failed,),
        )

        response = StaceyResponseComposer().compose(ingress, decision, report)
        self.assertEqual(response.status, UserResponseStatus.INCOMPLETE)
        self.assertIn("fully verified", response.message)

    def test_mismatched_scope_or_clarification_state_is_rejected(self) -> None:
        ingress = make_ingress()
        decision = make_decision()
        mismatched_scope = CollectiveExecutionReport(
            ingress.transaction_id,
            ingress.correlation_id,
            "release-response",
            ScopeVector("tenant-b", "user-a", "project-a", "workspace-a"),
            False,
            (),
        )
        with self.assertRaises(ResponseCompositionError):
            StaceyResponseComposer().compose(ingress, decision, mismatched_scope)

        mismatched_clarification = CollectiveExecutionReport(
            ingress.transaction_id,
            ingress.correlation_id,
            "release-response",
            SCOPE,
            True,
            (),
        )
        with self.assertRaises(ResponseCompositionError):
            StaceyResponseComposer().compose(ingress, decision, mismatched_clarification)


if __name__ == "__main__":
    unittest.main()