from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from models.stacey.core.inputs import UnifiedContextIngress
from models.stacey.core.outputs import CoreDecisionEnvelope
from substrate.contracts import Intent

from .collective_scheduler import CollectiveExecutionReport
from .identity import ScopeAuthorizationVerifier, SignedScopeGrant
from .response_composer import StaceyResponseComposer, UserResponse
from .system_inspection import SystemInspectionRequest


class CollectiveRuntimeError(RuntimeError):
    pass


class CollectiveDispatcher(Protocol):
    def active_release_snapshot(self) -> tuple[str, str] | None: ...

    def execute(
        self,
        decision: CoreDecisionEnvelope,
        *,
        request_payload: bytes,
        release_id: str,
    ) -> CollectiveExecutionReport: ...


@dataclass(frozen=True, slots=True)
class CollectiveRuntimeResult:
    decision: CoreDecisionEnvelope
    execution: CollectiveExecutionReport
    response: UserResponse
    pending_inspection: SystemInspectionRequest | None = None


class StaceyCollectiveRuntime:
    """Connect validated Core inference to proposal-only capability dispatch."""

    def __init__(
        self,
        *,
        core_lifecycle_manager: object,
        scheduler: CollectiveDispatcher,
        authorization_verifier: ScopeAuthorizationVerifier,
        response_composer: StaceyResponseComposer | None = None,
    ) -> None:
        if not callable(getattr(core_lifecycle_manager, "acquire_core_from_release", None)):
            raise ValueError("core_lifecycle_manager must acquire Core from a signed release")
        if not callable(getattr(scheduler, "execute", None)):
            raise ValueError("scheduler must implement execute()")
        if not callable(getattr(authorization_verifier, "authorize", None)):
            raise ValueError("authorization_verifier must implement authorize()")
        if response_composer is not None and not isinstance(response_composer, StaceyResponseComposer):
            raise ValueError("response_composer must be a StaceyResponseComposer")
        self._core_lifecycle_manager = core_lifecycle_manager
        self._scheduler = scheduler
        self._authorization_verifier = authorization_verifier
        self._response_composer = response_composer or StaceyResponseComposer()

    def execute(
        self,
        ingress: UnifiedContextIngress,
        authorization: SignedScopeGrant,
    ) -> CollectiveRuntimeResult:
        if not isinstance(ingress, UnifiedContextIngress):
            raise TypeError("ingress must be a UnifiedContextIngress")
        authorization_intent = Intent(
            transaction_id=ingress.transaction_id,
            correlation_id=ingress.correlation_id,
            action="ROUTE_TASK",
            goal=ingress.to_canonical_json(),
            scope=ingress.scope_vector,
        )
        authorized_scope = self._authorization_verifier.authorize(authorization_intent, authorization)
        if authorized_scope != ingress.scope_vector:
            raise CollectiveRuntimeError("Authorization verifier returned a different trusted scope")
        release_snapshot = self._scheduler.active_release_snapshot()
        if (
            not isinstance(release_snapshot, tuple)
            or len(release_snapshot) != 2
            or any(not isinstance(value, str) or not value.strip() for value in release_snapshot)
        ):
            raise CollectiveRuntimeError("No verified active collective release is available")
        release_id, expected_core_digest = release_snapshot
        with self._core_lifecycle_manager.acquire_core_from_release(release_id) as core:
            if getattr(core, "artifact_sha256", None) != expected_core_digest:
                raise CollectiveRuntimeError("Loaded Core digest differs from the signed active release")
            decision = core.infer(ingress)
        if not isinstance(decision, CoreDecisionEnvelope):
            raise CollectiveRuntimeError("Core returned an untyped decision")
        if (
            decision.transaction_id != ingress.transaction_id
            or decision.correlation_id != ingress.correlation_id
            or decision.scope_vector != ingress.scope_vector
        ):
            raise CollectiveRuntimeError("Core decision changed ingress identity or trusted scope")

        if decision.system_inspection is not None:
            if not decision.clarification.required or decision.task_dependency_graph.nodes:
                raise CollectiveRuntimeError("system inspection must pause task dispatch for consent")
            pending_inspection = SystemInspectionRequest(
                inspection_id=f"inspection:{ingress.transaction_id}",
                transaction_id=ingress.transaction_id,
                correlation_id=ingress.correlation_id,
                purpose=decision.system_inspection.purpose,
                active_release_id=release_id,
                scope=ingress.scope_vector,
                probe_ids=decision.system_inspection.probe_ids,
                required_capabilities=decision.system_inspection.required_capabilities,
            )
            execution = CollectiveExecutionReport(
                ingress.transaction_id,
                ingress.correlation_id,
                release_id,
                ingress.scope_vector,
                True,
                (),
            )
            response = self._response_composer.compose(ingress, decision, execution)
            return CollectiveRuntimeResult(decision, execution, response, pending_inspection)

        execution = self._scheduler.execute(
            decision,
            request_payload=ingress.to_canonical_json().encode("utf-8"),
            release_id=release_id,
        )
        if not isinstance(execution, CollectiveExecutionReport):
            raise CollectiveRuntimeError("Scheduler returned an untyped execution report")
        if (
            execution.transaction_id != ingress.transaction_id
            or execution.correlation_id != ingress.correlation_id
            or execution.release_id != release_id
            or execution.scope != ingress.scope_vector
            or execution.clarification_required != decision.clarification.required
        ):
            raise CollectiveRuntimeError("Scheduler report changed workflow identity, scope, or clarification state")
        response = self._response_composer.compose(ingress, decision, execution)
        return CollectiveRuntimeResult(decision, execution, response)