from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum

from substrate.contracts import ScopeVector


class StaceyIngressError(ValueError):
    pass


class IntentKind(str, Enum):
    USER_LANGUAGE = "USER_LANGUAGE"
    STRUCTURED_EVENT = "STRUCTURED_EVENT"


class SpecialistAvailability(str, Enum):
    AVAILABLE = "AVAILABLE"
    BUSY = "BUSY"
    UNAVAILABLE = "UNAVAILABLE"
    QUARANTINED = "QUARANTINED"


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StaceyIngressError(f"{name} must be a non-empty string")
    return value


def _nonnegative(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StaceyIngressError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class IntentVector:
    kind: IntentKind
    payload: str
    source_event_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, IntentKind):
            raise StaceyIngressError("kind must be an IntentKind")
        _text(self.payload, "intent payload")
        if self.source_event_id is not None:
            _text(self.source_event_id, "source_event_id")


@dataclass(frozen=True, slots=True)
class LedgerAssertion:
    assertion_id: str
    content_summary: str
    provenance_sha256: str
    last_observed_state: str

    def __post_init__(self) -> None:
        for name in ("assertion_id", "content_summary", "last_observed_state"):
            _text(getattr(self, name), name)
        if not isinstance(self.provenance_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", self.provenance_sha256) is None:
            raise StaceyIngressError("provenance_sha256 must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class ResourceMeasurement:
    resource_domain: str
    measured_available_bytes: int
    observed_at_ns: int

    def __post_init__(self) -> None:
        _text(self.resource_domain, "resource_domain")
        _nonnegative(self.measured_available_bytes, "measured_available_bytes")
        _nonnegative(self.observed_at_ns, "observed_at_ns")


@dataclass(frozen=True, slots=True)
class ActiveResourceLease:
    lease_id: str
    resource_domain: str
    reserved_bytes: int

    def __post_init__(self) -> None:
        _text(self.lease_id, "lease_id")
        _text(self.resource_domain, "resource_domain")
        _nonnegative(self.reserved_bytes, "reserved_bytes")


@dataclass(frozen=True, slots=True)
class SpecialistSlot:
    block_id: str
    capability_id: str
    status: SpecialistAvailability
    resource_domain: str
    estimated_required_bytes: int
    artifact_sha256: str
    input_modalities: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("block_id", "capability_id", "resource_domain"):
            _text(getattr(self, name), name)
        if not isinstance(self.status, SpecialistAvailability):
            raise StaceyIngressError("status must be a SpecialistAvailability")
        _nonnegative(self.estimated_required_bytes, "estimated_required_bytes")
        if not isinstance(self.artifact_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", self.artifact_sha256) is None:
            raise StaceyIngressError("artifact_sha256 must be a lowercase SHA-256 digest")
        if not isinstance(self.input_modalities, tuple):
            raise StaceyIngressError("input_modalities must be a tuple")
        for modality in self.input_modalities:
            _text(modality, "input modality")
        if len(set(self.input_modalities)) != len(self.input_modalities):
            raise StaceyIngressError("input_modalities must not contain duplicates")


@dataclass(frozen=True, slots=True)
class HardwareCapabilityMatrix:
    resources: tuple[ResourceMeasurement, ...]
    active_leases: tuple[ActiveResourceLease, ...]
    available_specialist_slots: tuple[SpecialistSlot, ...]

    def __post_init__(self) -> None:
        for name, values, expected_type in (
            ("resources", self.resources, ResourceMeasurement),
            ("active_leases", self.active_leases, ActiveResourceLease),
            ("available_specialist_slots", self.available_specialist_slots, SpecialistSlot),
        ):
            if not isinstance(values, tuple) or any(not isinstance(value, expected_type) for value in values):
                raise StaceyIngressError(f"{name} has invalid entries")
        domains = [resource.resource_domain for resource in self.resources]
        if len(domains) != len(set(domains)):
            raise StaceyIngressError("resource domains must be unique")
        capabilities = [slot.capability_id for slot in self.available_specialist_slots]
        if len(capabilities) != len(set(capabilities)):
            raise StaceyIngressError("specialist capability IDs must be unique in an ingress snapshot")


@dataclass(frozen=True, slots=True)
class UnifiedContextIngress:
    protocol_version: str
    transaction_id: str
    correlation_id: str
    ingress_timestamp_ns: int
    intent_vector: IntentVector
    scope_vector: ScopeVector
    canonical_state_assertions: tuple[LedgerAssertion, ...]
    hardware_capability_matrix: HardwareCapabilityMatrix

    def __post_init__(self) -> None:
        for name in ("protocol_version", "transaction_id", "correlation_id"):
            _text(getattr(self, name), name)
        _nonnegative(self.ingress_timestamp_ns, "ingress_timestamp_ns")
        if not isinstance(self.intent_vector, IntentVector):
            raise StaceyIngressError("intent_vector must be an IntentVector")
        if not isinstance(self.scope_vector, ScopeVector) or not self.scope_vector.is_complete():
            raise StaceyIngressError("scope_vector must contain all four trusted fields")
        if not isinstance(self.canonical_state_assertions, tuple) or any(
            not isinstance(assertion, LedgerAssertion) for assertion in self.canonical_state_assertions
        ):
            raise StaceyIngressError("canonical_state_assertions has invalid entries")
        if not isinstance(self.hardware_capability_matrix, HardwareCapabilityMatrix):
            raise StaceyIngressError("hardware_capability_matrix must be a HardwareCapabilityMatrix")

    @classmethod
    def from_payload(cls, payload: object) -> UnifiedContextIngress:
        def object_with_keys(value: object, expected: set[str], name: str) -> dict[str, object]:
            if not isinstance(value, dict) or set(value) != expected:
                raise StaceyIngressError(f"{name} has missing or unknown fields")
            return value

        try:
            top = object_with_keys(
                payload,
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
                "ingress",
            )
            intent = object_with_keys(
                top["intent_vector"],
                {"kind", "payload", "source_event_id"},
                "intent_vector",
            )
            scope = object_with_keys(
                top["scope_vector"],
                {"tenant_id", "user_id", "project_id", "workspace_id"},
                "scope_vector",
            )
            assertions_data = top["canonical_state_assertions"]
            if not isinstance(assertions_data, list):
                raise StaceyIngressError("canonical_state_assertions must be an array")
            assertions: list[LedgerAssertion] = []
            for assertion_data in assertions_data:
                assertion = object_with_keys(
                    assertion_data,
                    {"assertion_id", "content_summary", "provenance_sha256", "last_observed_state"},
                    "ledger assertion",
                )
                assertions.append(LedgerAssertion(**assertion))

            matrix = object_with_keys(
                top["hardware_capability_matrix"],
                {"resources", "active_leases", "available_specialist_slots"},
                "hardware_capability_matrix",
            )
            for field_name in ("resources", "active_leases", "available_specialist_slots"):
                if not isinstance(matrix[field_name], list):
                    raise StaceyIngressError(f"{field_name} must be an array")

            resources: list[ResourceMeasurement] = []
            for resource_data in matrix["resources"]:
                resource = object_with_keys(
                    resource_data,
                    {"resource_domain", "measured_available_bytes", "observed_at_ns"},
                    "resource measurement",
                )
                resources.append(ResourceMeasurement(**resource))

            leases: list[ActiveResourceLease] = []
            for lease_data in matrix["active_leases"]:
                lease = object_with_keys(
                    lease_data,
                    {"lease_id", "resource_domain", "reserved_bytes"},
                    "active resource lease",
                )
                leases.append(ActiveResourceLease(**lease))

            slots: list[SpecialistSlot] = []
            for slot_data in matrix["available_specialist_slots"]:
                slot = object_with_keys(
                    slot_data,
                    {
                        "block_id",
                        "capability_id",
                        "status",
                        "resource_domain",
                        "estimated_required_bytes",
                        "artifact_sha256",
                        "input_modalities",
                    },
                    "specialist slot",
                )
                modalities = slot["input_modalities"]
                if not isinstance(modalities, list):
                    raise StaceyIngressError("input_modalities must be an array")
                slots.append(
                    SpecialistSlot(
                        block_id=slot["block_id"],
                        capability_id=slot["capability_id"],
                        status=SpecialistAvailability(slot["status"]),
                        resource_domain=slot["resource_domain"],
                        estimated_required_bytes=slot["estimated_required_bytes"],
                        artifact_sha256=slot["artifact_sha256"],
                        input_modalities=tuple(modalities),
                    )
                )

            return cls(
                protocol_version=top["protocol_version"],
                transaction_id=top["transaction_id"],
                correlation_id=top["correlation_id"],
                ingress_timestamp_ns=top["ingress_timestamp_ns"],
                intent_vector=IntentVector(
                    kind=IntentKind(intent["kind"]),
                    payload=intent["payload"],
                    source_event_id=intent["source_event_id"],
                ),
                scope_vector=ScopeVector(**scope),
                canonical_state_assertions=tuple(assertions),
                hardware_capability_matrix=HardwareCapabilityMatrix(
                    resources=tuple(resources),
                    active_leases=tuple(leases),
                    available_specialist_slots=tuple(slots),
                ),
            )
        except StaceyIngressError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise StaceyIngressError("ingress payload contains invalid field values") from error

    def to_payload(self) -> dict[str, object]:
        return {
            "protocol_version": self.protocol_version,
            "transaction_id": self.transaction_id,
            "correlation_id": self.correlation_id,
            "ingress_timestamp_ns": self.ingress_timestamp_ns,
            "intent_vector": {
                "kind": self.intent_vector.kind.value,
                "payload": self.intent_vector.payload,
                "source_event_id": self.intent_vector.source_event_id,
            },
            "scope_vector": {
                "tenant_id": self.scope_vector.tenant_id,
                "user_id": self.scope_vector.user_id,
                "project_id": self.scope_vector.project_id,
                "workspace_id": self.scope_vector.workspace_id,
            },
            "canonical_state_assertions": [
                {
                    "assertion_id": assertion.assertion_id,
                    "content_summary": assertion.content_summary,
                    "provenance_sha256": assertion.provenance_sha256,
                    "last_observed_state": assertion.last_observed_state,
                }
                for assertion in self.canonical_state_assertions
            ],
            "hardware_capability_matrix": {
                "resources": [
                    {
                        "resource_domain": resource.resource_domain,
                        "measured_available_bytes": resource.measured_available_bytes,
                        "observed_at_ns": resource.observed_at_ns,
                    }
                    for resource in self.hardware_capability_matrix.resources
                ],
                "active_leases": [
                    {
                        "lease_id": lease.lease_id,
                        "resource_domain": lease.resource_domain,
                        "reserved_bytes": lease.reserved_bytes,
                    }
                    for lease in self.hardware_capability_matrix.active_leases
                ],
                "available_specialist_slots": [
                    {
                        "block_id": slot.block_id,
                        "capability_id": slot.capability_id,
                        "status": slot.status.value,
                        "resource_domain": slot.resource_domain,
                        "estimated_required_bytes": slot.estimated_required_bytes,
                        "artifact_sha256": slot.artifact_sha256,
                        "input_modalities": list(slot.input_modalities),
                    }
                    for slot in self.hardware_capability_matrix.available_specialist_slots
                ],
            },
        }

    def to_canonical_json(self) -> str:
        return json.dumps(self.to_payload(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))