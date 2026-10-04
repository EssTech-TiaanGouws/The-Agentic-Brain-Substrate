from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from enum import Enum

from substrate.contracts import ScopeVector

from .inputs import SpecialistAvailability, UnifiedContextIngress


class StaceyOutputError(ValueError):
    pass


class EdgeCondition(str, Enum):
    ON_SUCCESS = "ON_SUCCESS"
    ON_FAILURE = "ON_FAILURE"
    ALWAYS = "ALWAYS"


_CLARIFICATION_REASONS = frozenset(
    {
        "AMBIGUOUS_INTENT",
        "MISSING_REQUIRED_CONTEXT",
        "CONFLICTING_CONSTRAINTS",
        "LOW_CONFIDENCE",
        "CAPABILITY_UNAVAILABLE",
    }
)


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StaceyOutputError(f"{field_name} must be a non-empty string")
    return value


def _nonnegative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= sys.maxsize:
        raise StaceyOutputError(f"{field_name} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class TaskGraphNode:
    step_id: str
    capability_id: str
    target_block_id: str

    def __post_init__(self) -> None:
        _text(self.step_id, "step_id")
        _text(self.capability_id, "capability_id")
        _text(self.target_block_id, "target_block_id")


@dataclass(frozen=True, slots=True)
class TaskDependencyEdge:
    source_step_id: str
    target_step_id: str
    condition: EdgeCondition

    def __post_init__(self) -> None:
        _text(self.source_step_id, "source_step_id")
        _text(self.target_step_id, "target_step_id")
        if not isinstance(self.condition, EdgeCondition):
            raise StaceyOutputError("condition must be an EdgeCondition")
        if self.source_step_id == self.target_step_id:
            raise StaceyOutputError("task graph self-dependencies are not allowed")


@dataclass(frozen=True, slots=True)
class ResourceEstimate:
    step_id: str
    resource_domain: str
    estimated_required_bytes: int

    def __post_init__(self) -> None:
        _text(self.step_id, "resource estimate step_id")
        _text(self.resource_domain, "resource_domain")
        _nonnegative_int(self.estimated_required_bytes, "estimated_required_bytes")


@dataclass(frozen=True, slots=True)
class TaskDependencyGraph:
    nodes: tuple[TaskGraphNode, ...]
    edges: tuple[TaskDependencyEdge, ...]
    resource_estimates: tuple[ResourceEstimate, ...]

    def __post_init__(self) -> None:
        for name, values, expected_type in (
            ("nodes", self.nodes, TaskGraphNode),
            ("edges", self.edges, TaskDependencyEdge),
            ("resource_estimates", self.resource_estimates, ResourceEstimate),
        ):
            if not isinstance(values, tuple) or any(not isinstance(value, expected_type) for value in values):
                raise StaceyOutputError(f"{name} contains invalid entries")


@dataclass(frozen=True, slots=True)
class ClarificationDirective:
    required: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.required, bool):
            raise StaceyOutputError("clarification.required must be a boolean")
        if not isinstance(self.reason_codes, tuple):
            raise StaceyOutputError("clarification.reason_codes must be a tuple")
        for reason in self.reason_codes:
            if not isinstance(reason, str) or reason not in _CLARIFICATION_REASONS:
                raise StaceyOutputError(f"unsupported clarification reason code: {reason}")
        if len(set(self.reason_codes)) != len(self.reason_codes):
            raise StaceyOutputError("clarification reason codes must be unique")
        if self.required != bool(self.reason_codes):
            raise StaceyOutputError("clarification.required must match whether reason codes are present")


@dataclass(frozen=True, slots=True)
class SystemCapabilityRequirement:
    capability_id: str
    required_control_ids: tuple[str, ...]
    minimum_resources: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        _text(self.capability_id, "system_inspection.capability_id")
        if not isinstance(self.required_control_ids, tuple):
            raise StaceyOutputError("system_inspection.required_control_ids must be a tuple")
        for control_id in self.required_control_ids:
            _text(control_id, "system_inspection.required_control_id")
        if len(set(self.required_control_ids)) != len(self.required_control_ids):
            raise StaceyOutputError("system inspection control IDs must be unique")
        if not isinstance(self.minimum_resources, tuple):
            raise StaceyOutputError("system_inspection.minimum_resources must be a tuple")
        seen_domains: set[str] = set()
        for entry in self.minimum_resources:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise StaceyOutputError("system inspection resources must be (domain, bytes) pairs")
            domain, required_bytes = entry
            _text(domain, "system_inspection.resource_domain")
            if isinstance(required_bytes, bool) or not isinstance(required_bytes, int) or required_bytes <= 0:
                raise StaceyOutputError("system_inspection.required_bytes must be a positive integer")
            if domain in seen_domains:
                raise StaceyOutputError("system inspection resource domains must be unique")
            seen_domains.add(domain)


@dataclass(frozen=True, slots=True)
class SystemInspectionDirective:
    purpose: str
    probe_ids: tuple[str, ...]
    required_capabilities: tuple[SystemCapabilityRequirement, ...]

    def __post_init__(self) -> None:
        _text(self.purpose, "system_inspection.purpose")
        if not isinstance(self.probe_ids, tuple) or not self.probe_ids:
            raise StaceyOutputError("system_inspection.probe_ids must be a non-empty tuple")
        for probe_id in self.probe_ids:
            _text(probe_id, "system_inspection.probe_id")
        if len(set(self.probe_ids)) != len(self.probe_ids):
            raise StaceyOutputError("system_inspection probe IDs must be unique")
        if not isinstance(self.required_capabilities, tuple) or not self.required_capabilities:
            raise StaceyOutputError("system_inspection.required_capabilities must be a non-empty tuple")
        if any(not isinstance(item, SystemCapabilityRequirement) for item in self.required_capabilities):
            raise StaceyOutputError("system_inspection.required_capabilities contains an invalid entry")
        capability_ids = tuple(item.capability_id for item in self.required_capabilities)
        if len(set(capability_ids)) != len(capability_ids):
            raise StaceyOutputError("system inspection capability IDs must be unique")


@dataclass(frozen=True, slots=True)
class CalibrationMetrics:
    confidence_coefficient: float

    def __post_init__(self) -> None:
        value = self.confidence_coefficient
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise StaceyOutputError("confidence_coefficient must be numeric")
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise StaceyOutputError("confidence_coefficient must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class CoreDecisionEnvelope:
    protocol_version: str
    transaction_id: str
    correlation_id: str
    assigned_block_id: str
    predicted_consequences_summary: str
    calibration_metrics: CalibrationMetrics
    declarative_intent_action: str
    scope_vector: ScopeVector
    task_dependency_graph: TaskDependencyGraph
    clarification: ClarificationDirective
    system_inspection: SystemInspectionDirective | None = None

    def __post_init__(self) -> None:
        if self.system_inspection is not None and not isinstance(self.system_inspection, SystemInspectionDirective):
            raise StaceyOutputError("system_inspection must be a SystemInspectionDirective or None")
        if self.system_inspection is not None and (
            not self.clarification.required or self.task_dependency_graph.nodes
        ):
            raise StaceyOutputError("system inspection must request clarification before task dispatch")

    def to_payload(self) -> dict[str, object]:
        inspection_payload = None
        if self.system_inspection is not None:
            inspection_payload = {
                "purpose": self.system_inspection.purpose,
                "probe_ids": list(self.system_inspection.probe_ids),
                "required_capabilities": [
                    {
                        "capability_id": requirement.capability_id,
                        "required_control_ids": list(requirement.required_control_ids),
                        "minimum_resources": [list(resource) for resource in requirement.minimum_resources],
                    }
                    for requirement in self.system_inspection.required_capabilities
                ],
            }
        return {
            "protocol_version": self.protocol_version,
            "transaction_id": self.transaction_id,
            "correlation_id": self.correlation_id,
            "assigned_block_id": self.assigned_block_id,
            "predicted_consequences_summary": self.predicted_consequences_summary,
            "calibration_metrics": {
                "confidence_coefficient": self.calibration_metrics.confidence_coefficient,
            },
            "declarative_intent": {
                "action": self.declarative_intent_action,
                "scope_vector": {
                    "tenant_id": self.scope_vector.tenant_id,
                    "user_id": self.scope_vector.user_id,
                    "project_id": self.scope_vector.project_id,
                    "workspace_id": self.scope_vector.workspace_id,
                },
                "task_dependency_graph": {
                    "nodes": [
                        {
                            "step_id": node.step_id,
                            "capability_id": node.capability_id,
                            "target_block_id": node.target_block_id,
                        }
                        for node in self.task_dependency_graph.nodes
                    ],
                    "edges": [
                        {
                            "source_step_id": edge.source_step_id,
                            "target_step_id": edge.target_step_id,
                            "condition": edge.condition.value,
                        }
                        for edge in self.task_dependency_graph.edges
                    ],
                    "resource_estimates": [
                        {
                            "step_id": estimate.step_id,
                            "resource_domain": estimate.resource_domain,
                            "estimated_required_bytes": estimate.estimated_required_bytes,
                        }
                        for estimate in self.task_dependency_graph.resource_estimates
                    ],
                },
                "clarification": {
                    "required": self.clarification.required,
                    "reason_codes": list(self.clarification.reason_codes),
                },
                "system_inspection": inspection_payload,
            },
        }

    def to_jsonl(self) -> str:
        return json.dumps(self.to_payload(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise StaceyOutputError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_decision_jsonl(line: str, ingress: UnifiedContextIngress) -> CoreDecisionEnvelope:
    if not isinstance(line, str) or not line.strip() or "\n" in line.rstrip("\n") or "\r" in line.rstrip("\n"):
        raise StaceyOutputError("Core output must be exactly one non-empty JSON Lines record")
    if not isinstance(ingress, UnifiedContextIngress):
        raise StaceyOutputError("ingress must be a UnifiedContextIngress")
    try:
        payload = json.loads(line, object_pairs_hook=_unique_object, parse_constant=lambda token: (_ for _ in ()).throw(StaceyOutputError(f"invalid numeric constant: {token}")))
    except (json.JSONDecodeError, UnicodeError) as error:
        raise StaceyOutputError("Core output is not valid JSON") from error
    if not isinstance(payload, dict) or set(payload) != {
        "protocol_version",
        "transaction_id",
        "correlation_id",
        "assigned_block_id",
        "predicted_consequences_summary",
        "calibration_metrics",
        "declarative_intent",
    }:
        raise StaceyOutputError("Core decision envelope has missing or unknown top-level fields")
    if payload["protocol_version"] != "1.0":
        raise StaceyOutputError("unsupported Core decision protocol version")
    if payload["transaction_id"] != ingress.transaction_id or payload["correlation_id"] != ingress.correlation_id:
        raise StaceyOutputError("Core decision transaction identity does not match ingress")
    if payload["assigned_block_id"] != "BLOCK_0_CORE":
        raise StaceyOutputError("Core decision must be assigned to BLOCK_0_CORE")
    summary = _text(payload["predicted_consequences_summary"], "predicted_consequences_summary")

    calibration = payload["calibration_metrics"]
    if not isinstance(calibration, dict) or set(calibration) != {"confidence_coefficient"}:
        raise StaceyOutputError("calibration_metrics has missing or unknown fields")
    calibration_metrics = CalibrationMetrics(calibration["confidence_coefficient"])

    intent = payload["declarative_intent"]
    base_intent_fields = {
        "action",
        "scope_vector",
        "task_dependency_graph",
        "clarification",
    }
    if not isinstance(intent, dict) or frozenset(intent) not in {
        frozenset(base_intent_fields),
        frozenset(base_intent_fields | {"system_inspection"}),
    }:
        raise StaceyOutputError("declarative_intent has missing or unknown fields")
    if intent["action"] != "DECOMPOSE_TASK_GRAPH":
        raise StaceyOutputError("unsupported Core declarative action")

    scope_data = intent["scope_vector"]
    expected_scope = {
        "tenant_id": ingress.scope_vector.tenant_id,
        "user_id": ingress.scope_vector.user_id,
        "project_id": ingress.scope_vector.project_id,
        "workspace_id": ingress.scope_vector.workspace_id,
    }
    if scope_data != expected_scope:
        raise StaceyOutputError("Core output scope differs from trusted ingress scope")
    scope = ScopeVector(**scope_data)

    graph_data = intent["task_dependency_graph"]
    if not isinstance(graph_data, dict) or set(graph_data) != {"nodes", "edges", "resource_estimates"}:
        raise StaceyOutputError("task_dependency_graph has missing or unknown fields")
    if not all(isinstance(graph_data[name], list) for name in ("nodes", "edges", "resource_estimates")):
        raise StaceyOutputError("task graph collections must be arrays")

    nodes: list[TaskGraphNode] = []
    for node_data in graph_data["nodes"]:
        if not isinstance(node_data, dict) or set(node_data) != {"step_id", "capability_id", "target_block_id"}:
            raise StaceyOutputError("task graph node has missing or unknown fields")
        nodes.append(TaskGraphNode(**node_data))
    node_ids = [node.step_id for node in nodes]
    if len(node_ids) != len(set(node_ids)):
        raise StaceyOutputError("task graph node IDs must be unique")
    available_slots = {
        (slot.capability_id, slot.block_id)
        for slot in ingress.hardware_capability_matrix.available_specialist_slots
        if slot.status is SpecialistAvailability.AVAILABLE
    }
    if any((node.capability_id, node.target_block_id) not in available_slots for node in nodes):
        raise StaceyOutputError("task graph selects an unavailable or unregistered specialist slot")

    edges: list[TaskDependencyEdge] = []
    for edge_data in graph_data["edges"]:
        if not isinstance(edge_data, dict) or set(edge_data) != {
            "source_step_id", "target_step_id", "condition"
        }:
            raise StaceyOutputError("task graph edge has missing or unknown fields")
        try:
            edges.append(
                TaskDependencyEdge(
                    source_step_id=edge_data["source_step_id"],
                    target_step_id=edge_data["target_step_id"],
                    condition=EdgeCondition(edge_data["condition"]),
                )
            )
        except ValueError as error:
            raise StaceyOutputError("task graph edge contains an invalid condition") from error
    if len({(edge.source_step_id, edge.target_step_id, edge.condition) for edge in edges}) != len(edges):
        raise StaceyOutputError("task graph edges must be unique")
    if any(edge.source_step_id not in node_ids or edge.target_step_id not in node_ids for edge in edges):
        raise StaceyOutputError("task graph edge references an unknown step")

    adjacency: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    for edge in edges:
        adjacency[edge.source_step_id].append(edge.target_step_id)
    visited: set[str] = set()
    visiting: set[str] = set()

    def has_cycle(node_id: str) -> bool:
        if node_id in visiting:
            return True
        if node_id in visited:
            return False
        visiting.add(node_id)
        if any(has_cycle(target) for target in adjacency[node_id]):
            return True
        visiting.remove(node_id)
        visited.add(node_id)
        return False

    if any(has_cycle(node_id) for node_id in node_ids):
        raise StaceyOutputError("task dependency graph must be acyclic")

    estimates: list[ResourceEstimate] = []
    for estimate_data in graph_data["resource_estimates"]:
        if not isinstance(estimate_data, dict) or set(estimate_data) != {
            "step_id", "resource_domain", "estimated_required_bytes"
        }:
            raise StaceyOutputError("resource estimate has missing or unknown fields")
        estimates.append(ResourceEstimate(**estimate_data))
    if len({estimate.step_id for estimate in estimates}) != len(estimates):
        raise StaceyOutputError("resource estimates must have unique step IDs")
    if set(estimate.step_id for estimate in estimates) != set(node_ids):
        raise StaceyOutputError("every graph step must have exactly one resource estimate")

    clarification_data = intent["clarification"]
    if not isinstance(clarification_data, dict) or set(clarification_data) != {"required", "reason_codes"}:
        raise StaceyOutputError("clarification directive has missing or unknown fields")
    if not isinstance(clarification_data["reason_codes"], list):
        raise StaceyOutputError("clarification reason_codes must be an array")
    clarification = ClarificationDirective(
        required=clarification_data["required"],
        reason_codes=tuple(clarification_data["reason_codes"]),
    )
    if clarification.required and nodes:
        raise StaceyOutputError("Core must not dispatch specialist work while clarification is required")

    inspection_data = intent.get("system_inspection")
    system_inspection = None
    if inspection_data is not None:
        if not isinstance(inspection_data, dict) or set(inspection_data) != {
            "purpose", "probe_ids", "required_capabilities"
        }:
            raise StaceyOutputError("system_inspection has missing or unknown fields")
        probe_ids = inspection_data["probe_ids"]
        requirements_data = inspection_data["required_capabilities"]
        if not isinstance(probe_ids, list) or not isinstance(requirements_data, list):
            raise StaceyOutputError("system_inspection probes and required_capabilities must be arrays")
        requirements: list[SystemCapabilityRequirement] = []
        for requirement_data in requirements_data:
            if not isinstance(requirement_data, dict) or set(requirement_data) != {
                "capability_id", "required_control_ids", "minimum_resources"
            }:
                raise StaceyOutputError("system capability requirement has missing or unknown fields")
            control_ids = requirement_data["required_control_ids"]
            resources_data = requirement_data["minimum_resources"]
            if not isinstance(control_ids, list) or not isinstance(resources_data, list):
                raise StaceyOutputError("system capability controls/resources must be arrays")
            resources: list[tuple[str, int]] = []
            for resource in resources_data:
                if not isinstance(resource, list) or len(resource) != 2:
                    raise StaceyOutputError("system resource requirement must be a [domain, bytes] pair")
                resources.append((resource[0], resource[1]))
            requirements.append(
                SystemCapabilityRequirement(
                    capability_id=requirement_data["capability_id"],
                    required_control_ids=tuple(control_ids),
                    minimum_resources=tuple(resources),
                )
            )
        system_inspection = SystemInspectionDirective(
            purpose=inspection_data["purpose"],
            probe_ids=tuple(probe_ids),
            required_capabilities=tuple(requirements),
        )
        if not clarification.required or nodes:
            raise StaceyOutputError("system inspection must request clarification before task dispatch")

    return CoreDecisionEnvelope(
        protocol_version=payload["protocol_version"],
        transaction_id=payload["transaction_id"],
        correlation_id=payload["correlation_id"],
        assigned_block_id=payload["assigned_block_id"],
        predicted_consequences_summary=summary,
        calibration_metrics=calibration_metrics,
        declarative_intent_action=intent["action"],
        scope_vector=scope,
        task_dependency_graph=TaskDependencyGraph(tuple(nodes), tuple(edges), tuple(estimates)),
        clarification=clarification,
        system_inspection=system_inspection,
    )


def enforce_clarification_threshold(
    decision: CoreDecisionEnvelope,
    *,
    minimum_confidence: float,
) -> None:
    """Apply a caller-injected confidence threshold; model confidence is never authority."""
    if not isinstance(decision, CoreDecisionEnvelope):
        raise StaceyOutputError("decision must be a validated CoreDecisionEnvelope")
    if (
        isinstance(minimum_confidence, bool)
        or not isinstance(minimum_confidence, (int, float))
        or not math.isfinite(minimum_confidence)
        or not 0.0 <= minimum_confidence <= 1.0
    ):
        raise StaceyOutputError("minimum_confidence must be a finite value in [0, 1]")
    if (
        decision.calibration_metrics.confidence_coefficient < minimum_confidence
        and not decision.clarification.required
    ):
        raise StaceyOutputError("low-confidence decision must request clarification before specialist dispatch")