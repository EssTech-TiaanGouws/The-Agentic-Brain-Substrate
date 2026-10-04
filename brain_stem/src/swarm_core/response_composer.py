from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from models.stacey.core.inputs import UnifiedContextIngress
from models.stacey.core.outputs import CoreDecisionEnvelope
from substrate.contracts import ScopeVector

from .collective_scheduler import CapabilityOutput, CollectiveExecutionReport, StepState


class ResponseCompositionError(ValueError):
    pass


class UserResponseStatus(str, Enum):
    CLARIFICATION_REQUIRED = "CLARIFICATION_REQUIRED"
    SYSTEM_INSPECTION_CONSENT_REQUIRED = "SYSTEM_INSPECTION_CONSENT_REQUIRED"
    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"


@dataclass(frozen=True, slots=True)
class UserResponse:
    transaction_id: str
    correlation_id: str
    scope: ScopeVector
    status: UserResponseStatus
    message: str
    verification_references: tuple[str, ...]
    completed_step_ids: tuple[str, ...]
    incomplete_step_ids: tuple[str, ...]


class StaceyResponseComposer:
    """Deterministic Block 12 template formatter over validated workflow metadata."""

    _CLARIFICATION_TEXT = {
        "AMBIGUOUS_INTENT": "What would you like me to focus on?",
        "LOW_CONFIDENCE": "Could you clarify what you would like me to do?",
    }

    def compose(
        self,
        ingress: UnifiedContextIngress,
        decision: CoreDecisionEnvelope,
        report: CollectiveExecutionReport,
    ) -> UserResponse:
        if not isinstance(ingress, UnifiedContextIngress):
            raise TypeError("ingress must be a UnifiedContextIngress")
        if not isinstance(decision, CoreDecisionEnvelope):
            raise TypeError("decision must be a CoreDecisionEnvelope")
        if not isinstance(report, CollectiveExecutionReport):
            raise TypeError("report must be a CollectiveExecutionReport")
        if (
            decision.transaction_id != ingress.transaction_id
            or decision.correlation_id != ingress.correlation_id
            or decision.scope_vector != ingress.scope_vector
            or report.transaction_id != ingress.transaction_id
            or report.correlation_id != ingress.correlation_id
            or report.scope != ingress.scope_vector
        ):
            raise ResponseCompositionError("response inputs do not share one trusted workflow identity and scope")
        if report.clarification_required != decision.clarification.required:
            raise ResponseCompositionError("decision and scheduler disagree about clarification state")
        if report.clarification_required:
            if report.outputs or decision.task_dependency_graph.nodes:
                raise ResponseCompositionError("clarification response cannot include dispatched work")
            if decision.system_inspection is not None:
                return UserResponse(
                    ingress.transaction_id,
                    ingress.correlation_id,
                    ingress.scope_vector,
                    UserResponseStatus.SYSTEM_INSPECTION_CONSENT_REQUIRED,
                    "May I inspect the specified system surfaces before checking whether this model bundle fits?",
                    (),
                    (),
                    (),
                )
            return UserResponse(
                ingress.transaction_id,
                ingress.correlation_id,
                ingress.scope_vector,
                UserResponseStatus.CLARIFICATION_REQUIRED,
                self._clarification_message(decision.clarification.reason_codes),
                (),
                (),
                (),
            )

        nodes_by_id = {node.step_id: node for node in decision.task_dependency_graph.nodes}
        outputs_by_id: dict[str, CapabilityOutput] = {}
        for output in report.outputs:
            if not isinstance(output, CapabilityOutput) or output.step_id in outputs_by_id:
                raise ResponseCompositionError("scheduler report contains invalid or duplicate outputs")
            node = nodes_by_id.get(output.step_id)
            if node is None or node.capability_id != output.capability_id:
                raise ResponseCompositionError("scheduler output is not bound to the Core task graph")
            outputs_by_id[output.step_id] = output
        if set(outputs_by_id) != set(nodes_by_id):
            raise ResponseCompositionError("scheduler report does not cover every task graph node")

        completed = tuple(
            step_id
            for step_id, output in outputs_by_id.items()
            if output.state is StepState.SUCCEEDED
        )
        incomplete = tuple(
            step_id
            for step_id, output in outputs_by_id.items()
            if output.state is not StepState.SUCCEEDED
        )
        references = tuple(
            reference
            for output in outputs_by_id.values()
            for reference in (output.critic_reference, output.auditor_reference)
            if isinstance(reference, str) and reference.strip()
        )
        all_complete = bool(outputs_by_id) and not incomplete and all(
            output.artifact_sha256 is not None for output in outputs_by_id.values()
        )
        if all_complete:
            status = UserResponseStatus.COMPLETE
            message = f"Completed the request and verified {len(completed)} step(s)."
        else:
            status = UserResponseStatus.INCOMPLETE
            message = "I could not complete the request to a fully verified result."
        return UserResponse(
            ingress.transaction_id,
            ingress.correlation_id,
            ingress.scope_vector,
            status,
            message,
            references,
            completed,
            incomplete,
        )

    @classmethod
    def _clarification_message(cls, reason_codes: tuple[str, ...]) -> str:
        for reason_code in reason_codes:
            message = cls._CLARIFICATION_TEXT.get(reason_code)
            if message is not None:
                return message
        return "Could you clarify what you would like me to do?"