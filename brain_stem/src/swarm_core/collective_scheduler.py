from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from types import MappingProxyType
from typing import Callable, Mapping, Protocol

from models.stacey.core.outputs import (
    CoreDecisionEnvelope,
    EdgeCondition,
    ResourceEstimate,
    TaskGraphNode,
)
from substrate.contracts import ScopeVector

from .model_lifecycle import ModelLifecycleManager


class CollectiveScheduleError(RuntimeError):
    pass


class CapabilityContractError(ValueError):
    pass


class StepState(str, Enum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass(frozen=True, slots=True)
class CapabilityContract:
    capability_id: str
    target_block_id: str
    version: str
    schema_sha256: str
    maximum_request_bytes: int
    maximum_response_bytes: int

    def __post_init__(self) -> None:
        for field_name in ("capability_id", "target_block_id", "version"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise CapabilityContractError(f"{field_name} must be a non-empty string")
        if (
            not isinstance(self.schema_sha256, str)
            or len(self.schema_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.schema_sha256)
        ):
            raise CapabilityContractError("schema_sha256 must be a lowercase SHA-256 digest")
        for field_name in ("maximum_request_bytes", "maximum_response_bytes"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise CapabilityContractError(f"{field_name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class CapabilityRequest:
    transaction_id: str
    correlation_id: str
    step_id: str
    capability_id: str
    target_block_id: str
    contract_version: str
    scope: ScopeVector
    payload: bytes
    dependencies: tuple[CapabilityOutput, ...]


@dataclass(frozen=True, slots=True)
class CapabilityResponse:
    transaction_id: str
    correlation_id: str
    step_id: str
    capability_id: str
    target_block_id: str
    contract_version: str
    scope: ScopeVector
    payload: bytes
    provenance_references: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CapabilityOutput:
    step_id: str
    capability_id: str
    state: StepState
    payload: bytes | None
    error_code: str | None
    provenance_references: tuple[str, ...]
    critic_reference: str | None
    auditor_reference: str | None
    artifact_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class CollectiveExecutionReport:
    transaction_id: str
    correlation_id: str
    release_id: str
    scope: ScopeVector
    clarification_required: bool
    outputs: tuple[CapabilityOutput, ...]


class CapabilityResponseValidator(Protocol):
    def validate(self, request: CapabilityRequest, response: CapabilityResponse) -> None: ...


class CollectiveReviewer(Protocol):
    def verify_adversarial(self, request: CapabilityRequest, response: CapabilityResponse) -> str: ...

    def verify_epistemic(self, request: CapabilityRequest, response: CapabilityResponse) -> str: ...


class CapabilityContractRegistry:
    def __init__(
        self,
        contracts: tuple[CapabilityContract, ...],
        validators: Mapping[str, CapabilityResponseValidator],
    ) -> None:
        by_capability: dict[str, CapabilityContract] = {}
        for contract in contracts:
            if not isinstance(contract, CapabilityContract):
                raise CapabilityContractError("contracts must contain CapabilityContract values")
            if contract.capability_id in by_capability:
                raise CapabilityContractError("duplicate capability contract")
            if not callable(getattr(validators.get(contract.capability_id), "validate", None)):
                raise CapabilityContractError("every contract requires a response validator")
            by_capability[contract.capability_id] = contract
        if set(validators) != set(by_capability):
            raise CapabilityContractError("validators must exactly match registered capability contracts")
        self._contracts = MappingProxyType(by_capability)
        self._validators = MappingProxyType(dict(validators))

    def resolve(self, capability_id: str) -> tuple[CapabilityContract, CapabilityResponseValidator]:
        try:
            return self._contracts[capability_id], self._validators[capability_id]
        except KeyError as error:
            raise CapabilityContractError(f"No contract is registered for {capability_id}") from error


class CollectiveScheduler:
    """Deterministic sequential DAG executor; learned models cannot authorize mutations."""

    def __init__(
        self,
        *,
        lifecycle_manager: ModelLifecycleManager,
        contracts: CapabilityContractRegistry,
        reviewer: CollectiveReviewer,
    ) -> None:
        if (
            not callable(getattr(lifecycle_manager, "acquire_from_active_release", None))
            or not callable(getattr(lifecycle_manager, "active_release", None))
            or not callable(getattr(lifecycle_manager, "core_artifact_sha256_for_release", None))
        ):
            raise ValueError("lifecycle_manager must resolve and pin signed release artifacts")
        if not callable(getattr(reviewer, "verify_adversarial", None)) or not callable(
            getattr(reviewer, "verify_epistemic", None)
        ):
            raise ValueError("reviewer must implement Block 6 and Block 8 checks")
        self._lifecycle_manager = lifecycle_manager
        self._contracts = contracts
        self._reviewer = reviewer

    def active_release_snapshot(self) -> tuple[str, str] | None:
        release = self._lifecycle_manager.active_release()
        if release is None:
            return None
        release_id = getattr(release, "release_id", None)
        if not isinstance(release_id, str) or not release_id.strip():
            raise CollectiveScheduleError("active release has no valid ID")
        core_digest = self._lifecycle_manager.core_artifact_sha256_for_release(release_id)
        if not isinstance(core_digest, str) or len(core_digest) != 64:
            raise CollectiveScheduleError("active release has no valid Core artifact digest")
        return release_id, core_digest

    def execute(
        self,
        decision: CoreDecisionEnvelope,
        *,
        request_payload: bytes,
        release_id: str | None = None,
    ) -> CollectiveExecutionReport:
        if not isinstance(decision, CoreDecisionEnvelope):
            raise TypeError("decision must be a validated CoreDecisionEnvelope")
        if not isinstance(request_payload, bytes):
            raise TypeError("request_payload must be bytes")
        scope = decision.scope_vector
        if not isinstance(scope, ScopeVector) or not scope.is_complete():
            raise CollectiveScheduleError("decision has no complete trusted scope")
        if release_id is None:
            snapshot = self.active_release_snapshot()
            if snapshot is None:
                raise CollectiveScheduleError("no signed collective release is active")
            release_id = snapshot[0]
        elif not isinstance(release_id, str) or not release_id.strip():
            raise CollectiveScheduleError("release_id must be a non-empty string")
        else:
            self._lifecycle_manager.core_artifact_sha256_for_release(release_id)
        if decision.clarification.required:
            return CollectiveExecutionReport(
                decision.transaction_id,
                decision.correlation_id,
                release_id,
                scope,
                True,
                (),
            )

        graph = decision.task_dependency_graph
        nodes = {node.step_id: node for node in graph.nodes}
        edges_by_target: dict[str, list[object]] = {step_id: [] for step_id in nodes}
        outgoing: dict[str, list[object]] = {step_id: [] for step_id in nodes}
        for edge in graph.edges:
            if edge.source_step_id not in nodes or edge.target_step_id not in nodes:
                raise CollectiveScheduleError("task graph edge references an unknown node")
            edges_by_target[edge.target_step_id].append(edge)
            outgoing[edge.source_step_id].append(edge)
        estimate_by_step = {item.step_id: item for item in graph.resource_estimates}
        if set(estimate_by_step) != set(nodes):
            raise CollectiveScheduleError("each task graph node must have exactly one resource estimate")
        self._validate_acyclic(nodes, outgoing)

        states: dict[str, StepState] = {}
        outputs: dict[str, CapabilityOutput] = {}
        pending = set(nodes)
        while pending:
            ready = sorted(
                step_id
                for step_id in pending
                if all(edge.source_step_id in states for edge in edges_by_target[step_id])
            )
            if not ready:
                raise CollectiveScheduleError("task graph cannot make deterministic progress")
            for step_id in ready:
                node = nodes[step_id]
                incoming = edges_by_target[step_id]
                if incoming and not all(self._edge_matches(edge.condition, states[edge.source_step_id]) for edge in incoming):
                    output = CapabilityOutput(
                        step_id,
                        node.capability_id,
                        StepState.SKIPPED,
                        None,
                        "CONDITION_NOT_MET",
                        (),
                        None,
                        None,
                    )
                    states[step_id] = output.state
                    outputs[step_id] = output
                    pending.remove(step_id)
                    continue
                dependencies = tuple(outputs[edge.source_step_id] for edge in incoming)
                output = self._execute_step(
                    decision,
                    node,
                    request_payload,
                    dependencies,
                    estimate_by_step[step_id],
                    release_id,
                )
                states[step_id] = output.state
                outputs[step_id] = output
                pending.remove(step_id)

        return CollectiveExecutionReport(
            decision.transaction_id,
            decision.correlation_id,
            release_id,
            scope,
            False,
            tuple(outputs[step_id] for step_id in nodes),
        )

    def _execute_step(
        self,
        decision: CoreDecisionEnvelope,
        node: TaskGraphNode,
        request_payload: bytes,
        dependencies: tuple[CapabilityOutput, ...],
        resource_estimate: ResourceEstimate,
        release_id: str,
    ) -> CapabilityOutput:
        try:
            contract, validator = self._contracts.resolve(node.capability_id)
            if contract.target_block_id != node.target_block_id:
                raise CapabilityContractError("task graph target block differs from its capability contract")
            if len(request_payload) > contract.maximum_request_bytes:
                raise CapabilityContractError("request exceeds the capability contract byte limit")
            request = CapabilityRequest(
                transaction_id=decision.transaction_id,
                correlation_id=decision.correlation_id,
                step_id=node.step_id,
                capability_id=node.capability_id,
                target_block_id=node.target_block_id,
                contract_version=contract.version,
                scope=decision.scope_vector,
                payload=request_payload,
                dependencies=dependencies,
            )
            with self._lifecycle_manager.acquire_from_active_release(
                node.capability_id,
                release_id=release_id,
                contract_version=contract.version,
                contract_sha256=contract.schema_sha256,
                resource_domain=resource_estimate.resource_domain,
                estimated_required_bytes=resource_estimate.estimated_required_bytes,
            ) as model:
                response = model.infer(request)
                artifact_sha256 = getattr(model, "artifact_sha256", None)
            if not isinstance(artifact_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", artifact_sha256) is None:
                raise CollectiveScheduleError("leased model did not expose its signed artifact digest")
            if not isinstance(response, CapabilityResponse):
                raise CapabilityContractError("specialist returned an untyped response")
            self._validate_response(request, response, contract)
            validator.validate(request, response)
            critic_reference = self._reviewer.verify_adversarial(request, response)
            if not isinstance(critic_reference, str) or not critic_reference.strip():
                raise CollectiveScheduleError("Block 6 did not return a verification reference")
            auditor_reference = self._reviewer.verify_epistemic(request, response)
            if not isinstance(auditor_reference, str) or not auditor_reference.strip():
                raise CollectiveScheduleError("Block 8 did not return a verification reference")
            return CapabilityOutput(
                node.step_id,
                node.capability_id,
                StepState.SUCCEEDED,
                response.payload,
                None,
                response.provenance_references,
                critic_reference,
                auditor_reference,
                artifact_sha256,
            )
        except Exception as error:
            return CapabilityOutput(
                node.step_id,
                node.capability_id,
                StepState.FAILED,
                None,
                type(error).__name__,
                (),
                None,
                None,
            )

    @staticmethod
    def _validate_response(
        request: CapabilityRequest,
        response: CapabilityResponse,
        contract: CapabilityContract,
    ) -> None:
        if (
            response.transaction_id != request.transaction_id
            or response.correlation_id != request.correlation_id
            or response.step_id != request.step_id
            or response.capability_id != request.capability_id
            or response.target_block_id != request.target_block_id
            or response.contract_version != contract.version
            or response.scope != request.scope
        ):
            raise CapabilityContractError("specialist response changed request identity, scope, or contract")
        if not isinstance(response.payload, bytes) or len(response.payload) > contract.maximum_response_bytes:
            raise CapabilityContractError("specialist response violates the capability byte limit")
        if not isinstance(response.provenance_references, tuple) or any(
            not isinstance(reference, str) or not reference.strip()
            for reference in response.provenance_references
        ):
            raise CapabilityContractError("specialist response provenance is malformed")

    @staticmethod
    def _edge_matches(condition: EdgeCondition, source_state: StepState) -> bool:
        if condition is EdgeCondition.ALWAYS:
            return True
        if condition is EdgeCondition.ON_SUCCESS:
            return source_state is StepState.SUCCEEDED
        if condition is EdgeCondition.ON_FAILURE:
            return source_state is StepState.FAILED
        return False

    @staticmethod
    def _validate_acyclic(nodes: Mapping[str, TaskGraphNode], outgoing: Mapping[str, list[object]]) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(step_id: str) -> None:
            if step_id in visiting:
                raise CollectiveScheduleError("task graph contains a cycle")
            if step_id in visited:
                return
            visiting.add(step_id)
            for edge in outgoing[step_id]:
                visit(edge.target_step_id)
            visiting.remove(step_id)
            visited.add(step_id)

        for step_id in nodes:
            visit(step_id)