from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from time import time_ns
from typing import Callable, Protocol

from substrate.contracts import Intent, ScopeVector
from models.stacey.core.outputs import SystemCapabilityRequirement

from .identity import ScopeAuthorizationVerifier, SignedScopeGrant
from .hardware_profile import HardwareProfile, probe_hardware_profile
from .model_lifecycle import ModelArtifactRegistry


class SystemInspectionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SystemInspectionRequest:
    inspection_id: str
    transaction_id: str
    correlation_id: str
    purpose: str
    active_release_id: str
    scope: ScopeVector
    probe_ids: tuple[str, ...]
    required_capabilities: tuple[SystemCapabilityRequirement, ...]

    def __post_init__(self) -> None:
        for field_name in ("inspection_id", "transaction_id", "correlation_id", "purpose", "active_release_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if not isinstance(self.scope, ScopeVector) or not self.scope.is_complete():
            raise ValueError("system inspection requires a complete trusted scope")
        if not isinstance(self.probe_ids, tuple) or not self.probe_ids:
            raise ValueError("probe_ids must be a non-empty tuple")
        if any(not isinstance(probe, str) or not probe.strip() for probe in self.probe_ids):
            raise ValueError("probe_ids must contain non-empty strings")
        if len(set(self.probe_ids)) != len(self.probe_ids):
            raise ValueError("probe_ids must not contain duplicates")
        if not isinstance(self.required_capabilities, tuple) or not self.required_capabilities:
            raise ValueError("required_capabilities must be a non-empty tuple")
        if any(not isinstance(item, SystemCapabilityRequirement) for item in self.required_capabilities):
            raise ValueError("required_capabilities must contain SystemCapabilityRequirement values")
        capability_ids = tuple(item.capability_id for item in self.required_capabilities)
        if len(set(capability_ids)) != len(capability_ids):
            raise ValueError("required capability IDs must not contain duplicates")

    def canonical_json(self) -> str:
        return json.dumps(
            {
                "inspection_id": self.inspection_id,
                "transaction_id": self.transaction_id,
                "correlation_id": self.correlation_id,
                "purpose": self.purpose,
                "active_release_id": self.active_release_id,
                "scope": {
                    "tenant_id": self.scope.tenant_id,
                    "user_id": self.scope.user_id,
                    "project_id": self.scope.project_id,
                    "workspace_id": self.scope.workspace_id,
                },
                "probe_ids": list(self.probe_ids),
                "required_capabilities": [
                    {
                        "capability_id": need.capability_id,
                        "required_control_ids": list(need.required_control_ids),
                        "minimum_resources": [list(resource) for resource in need.minimum_resources],
                    }
                    for need in self.required_capabilities
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


CapabilityNeed = SystemCapabilityRequirement


@dataclass(frozen=True, slots=True)
class ApprovedCapabilityComponent:
    capability_id: str
    artifact_id: str
    artifact_sha256: str
    resource_domain: str
    reserved_bytes: int

    def __post_init__(self) -> None:
        for field_name in ("capability_id", "artifact_id", "resource_domain"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if not isinstance(self.artifact_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", self.artifact_sha256) is None:
            raise ValueError("artifact_sha256 must be a lowercase SHA-256 digest")
        if isinstance(self.reserved_bytes, bool) or not isinstance(self.reserved_bytes, int) or self.reserved_bytes <= 0:
            raise ValueError("reserved_bytes must be a positive integer")


@dataclass(frozen=True, slots=True)
class SystemProbeEvidence:
    probe_id: str
    scope: ScopeVector
    available: bool
    source_ref: str
    source_sha256: str
    observed_at_ns: int
    controls: tuple[str, ...]
    available_resources: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        for field_name in ("probe_id", "source_ref"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if not isinstance(self.available, bool):
            raise ValueError("available must be a boolean")
        if not isinstance(self.scope, ScopeVector) or not self.scope.is_complete():
            raise ValueError("probe evidence must contain a complete four-field scope")
        if not isinstance(self.source_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", self.source_sha256) is None:
            raise ValueError("source_sha256 must be a lowercase SHA-256 digest")
        if isinstance(self.observed_at_ns, bool) or not isinstance(self.observed_at_ns, int) or self.observed_at_ns < 0:
            raise ValueError("observed_at_ns must be a non-negative integer")
        if not isinstance(self.controls, tuple) or any(not isinstance(value, str) or not value.strip() for value in self.controls):
            raise ValueError("controls must contain non-empty strings")
        if len(set(self.controls)) != len(self.controls):
            raise ValueError("controls must not contain duplicates")
        if not isinstance(self.available_resources, tuple):
            raise ValueError("available_resources must be a tuple")
        domains: set[str] = set()
        for entry in self.available_resources:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise ValueError("available resources must be (domain, bytes) pairs")
            domain, available_bytes = entry
            if not isinstance(domain, str) or not domain.strip():
                raise ValueError("resource domain must be a non-empty string")
            if isinstance(available_bytes, bool) or not isinstance(available_bytes, int) or available_bytes < 0:
                raise ValueError("available resource bytes must be a non-negative integer")
            if domain in domains:
                raise ValueError("available_resources cannot repeat a domain")
            domains.add(domain)


class SystemProbeProvider(Protocol):
    def inspect(
        self,
        scope: ScopeVector,
        probe_ids: tuple[str, ...],
    ) -> tuple[SystemProbeEvidence, ...]: ...


class HostHardwareProbeProvider:
    """Read-only adapter from the measured local host profile to an approved probe ID."""

    probe_id = "host.hardware"

    def __init__(
        self,
        *,
        profile_provider: Callable[[], HardwareProfile] = probe_hardware_profile,
    ) -> None:
        if not callable(profile_provider):
            raise ValueError("profile_provider must be callable")
        self._profile_provider = profile_provider

    def inspect(
        self,
        scope: ScopeVector,
        probe_ids: tuple[str, ...],
    ) -> tuple[SystemProbeEvidence, ...]:
        if not isinstance(scope, ScopeVector) or not scope.is_complete():
            raise SystemInspectionError("host hardware probe requires a complete trusted scope")
        if not isinstance(probe_ids, tuple) or any(probe_id != self.probe_id for probe_id in probe_ids):
            raise SystemInspectionError("host hardware provider received an unimplemented probe ID")
        profile = self._profile_provider()
        if not isinstance(profile, HardwareProfile):
            raise SystemInspectionError("hardware profile provider returned an untyped profile")
        payload = json.dumps(
            profile.to_payload(),
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        resources: list[tuple[str, int]] = []
        if profile.host_memory_available_bytes is not None:
            resources.append(("host-memory", profile.host_memory_available_bytes))
        for accelerator in profile.accelerators:
            if accelerator.available_memory_bytes is not None:
                resources.append((f"device-memory:{accelerator.device_id}", accelerator.available_memory_bytes))
        controls = [f"cpu-architecture:{profile.machine_architecture}"]
        if profile.tensor_runtime is not None:
            controls.append(f"tensor-runtime:{profile.tensor_runtime}")
        controls.extend(
            f"accelerator-backend:{accelerator.backend_id}"
            for accelerator in profile.accelerators
        )
        return (
            SystemProbeEvidence(
                probe_id=self.probe_id,
                scope=scope,
                available=True,
                source_ref=f"host-hardware-profile:{profile.observed_at_ns}",
                source_sha256=hashlib.sha256(payload).hexdigest(),
                observed_at_ns=profile.observed_at_ns,
                controls=tuple(sorted(set(controls))),
                available_resources=tuple(sorted(resources)),
            ),
        )


@dataclass(frozen=True, slots=True)
class ResourceDeficit:
    resource_domain: str
    required_bytes: int
    available_bytes: int


@dataclass(frozen=True, slots=True)
class SystemFitReport:
    inspection_id: str
    scope: ScopeVector
    active_release_id: str
    available_capability_ids: tuple[str, ...]
    missing_capability_ids: tuple[str, ...]
    missing_probe_ids: tuple[str, ...]
    missing_control_ids: tuple[str, ...]
    resource_deficits: tuple[ResourceDeficit, ...]
    evidence: tuple[SystemProbeEvidence, ...]
    build_request_required: bool


class SystemInspectionService:
    """Consent-gated, read-only system fit assessment for one Stacey model bundle."""

    def __init__(
        self,
        *,
        authorization_verifier: ScopeAuthorizationVerifier,
        artifact_registry: ModelArtifactRegistry,
        probe_provider: SystemProbeProvider,
        allowed_probe_ids: tuple[str, ...],
        maximum_evidence_age_ns: int,
        clock: Callable[[], int] = time_ns,
    ) -> None:
        if not callable(getattr(authorization_verifier, "authorize", None)):
            raise ValueError("authorization_verifier must implement authorize()")
        if not callable(getattr(probe_provider, "inspect", None)):
            raise ValueError("probe_provider must implement inspect()")
        if not isinstance(artifact_registry, ModelArtifactRegistry):
            raise ValueError("artifact_registry must be a signature-verifying ModelArtifactRegistry")
        if not isinstance(allowed_probe_ids, tuple) or not allowed_probe_ids or any(
            not isinstance(value, str) or not value.strip() for value in allowed_probe_ids
        ):
            raise ValueError("allowed_probe_ids must be a non-empty tuple")
        if len(set(allowed_probe_ids)) != len(allowed_probe_ids):
            raise ValueError("allowed_probe_ids must not contain duplicates")
        if isinstance(maximum_evidence_age_ns, bool) or not isinstance(maximum_evidence_age_ns, int) or maximum_evidence_age_ns <= 0:
            raise ValueError("maximum_evidence_age_ns must be a positive integer")
        if not callable(clock):
            raise ValueError("clock must be callable")
        self._authorization_verifier = authorization_verifier
        self._artifact_registry = artifact_registry
        self._probe_provider = probe_provider
        self._allowed_probe_ids = frozenset(allowed_probe_ids)
        self._maximum_evidence_age_ns = maximum_evidence_age_ns
        self._clock = clock

    def inspect_and_assess(
        self,
        request: SystemInspectionRequest,
        authorization: SignedScopeGrant,
        *,
        active_release_id: str,
    ) -> SystemFitReport:
        if not isinstance(request, SystemInspectionRequest):
            raise TypeError("request must be a SystemInspectionRequest")
        if not isinstance(authorization, SignedScopeGrant):
            raise SystemInspectionError("signed operator inspection authorization is required")
        if not isinstance(active_release_id, str) or not active_release_id.strip():
            raise SystemInspectionError("active_release_id must be a non-empty signed release ID")
        if request.active_release_id != active_release_id:
            raise SystemInspectionError("inspection request is bound to a different signed release")
        if any(probe_id not in self._allowed_probe_ids for probe_id in request.probe_ids):
            raise SystemInspectionError("request contains a probe outside the operator-approved allowlist")
        required_capabilities = request.required_capabilities
        if not isinstance(required_capabilities, tuple) or any(
            not isinstance(item, SystemCapabilityRequirement) for item in required_capabilities
        ):
            raise TypeError("required_capabilities must contain SystemCapabilityRequirement values")

        intent = Intent(
            transaction_id=request.transaction_id,
            correlation_id=request.correlation_id,
            action="INSPECT_SYSTEM",
            goal=request.canonical_json(),
            scope=request.scope,
        )
        try:
            authorized_scope = self._authorization_verifier.authorize(intent, authorization)
        except Exception as error:
            raise SystemInspectionError("system inspection lacks exact signed authorization") from error
        if authorized_scope != request.scope:
            raise SystemInspectionError("inspection authorization changed the trusted scope")

        try:
            active_release = self._artifact_registry.active_release()
            if active_release is None or active_release.release_id != active_release_id:
                raise SystemInspectionError("requested system-fit assessment is not bound to the active signed release")
            active_components = tuple(
                ApprovedCapabilityComponent(
                    binding.capability_id,
                    binding.artifact_id,
                    self._artifact_registry.get(binding.artifact_id).artifact_sha256,
                    self._artifact_registry.get(binding.artifact_id).resource_domain,
                    self._artifact_registry.get(binding.artifact_id).reserved_bytes,
                )
                for binding in active_release.capability_bindings
            )
        except SystemInspectionError:
            raise
        except Exception as error:
            raise SystemInspectionError("active capability inventory failed signed-release verification") from error

        try:
            evidence = self._probe_provider.inspect(request.scope, request.probe_ids)
        except Exception as error:
            raise SystemInspectionError("approved system probe failed") from error
        if not isinstance(evidence, tuple) or any(not isinstance(item, SystemProbeEvidence) for item in evidence):
            raise SystemInspectionError("system probe returned untyped evidence")
        evidence_by_id = {item.probe_id: item for item in evidence}
        if set(evidence_by_id) != set(request.probe_ids) or len(evidence_by_id) != len(evidence):
            raise SystemInspectionError("system probe response must exactly match the approved probe set")
        now_ns = self._clock()
        if isinstance(now_ns, bool) or not isinstance(now_ns, int) or now_ns < 0:
            raise SystemInspectionError("clock must return a non-negative integer timestamp")
        for item in evidence:
            if item.scope != request.scope:
                raise SystemInspectionError("system probe evidence crossed the authorized scope")
            age_ns = now_ns - item.observed_at_ns
            if age_ns < 0 or age_ns > self._maximum_evidence_age_ns:
                raise SystemInspectionError("system probe evidence is stale or from the future")

        active_ids = {component.capability_id for component in active_components}
        requested_ids = tuple(item.capability_id for item in required_capabilities)
        if len(set(requested_ids)) != len(requested_ids):
            raise SystemInspectionError("required capability IDs must not contain duplicates")
        missing_capabilities = tuple(sorted(set(requested_ids) - active_ids))
        available_controls = {
            control
            for item in evidence
            if item.available
            for control in item.controls
        }
        missing_probes = tuple(
            sorted(
                {
                    probe_id
                    for probe_id in request.probe_ids
                    if probe_id not in evidence_by_id or not evidence_by_id[probe_id].available
                }
            )
        )
        missing_controls = tuple(
            sorted(
                {
                    control_id
                    for need in required_capabilities
                    for control_id in need.required_control_ids
                    if control_id not in available_controls
                }
            )
        )
        required_resources: dict[str, int] = {}
        for need in required_capabilities:
            for domain, amount in need.minimum_resources:
                required_resources[domain] = required_resources.get(domain, 0) + amount
        available_resources: dict[str, int] = {}
        for item in evidence:
            if item.available:
                for domain, amount in item.available_resources:
                    available_resources[domain] = max(available_resources.get(domain, 0), amount)
        deficits = tuple(
            ResourceDeficit(domain, required, available_resources.get(domain, 0))
            for domain, required in sorted(required_resources.items())
            if available_resources.get(domain, 0) < required
        )
        needs_build = bool(missing_capabilities or missing_probes or missing_controls or deficits)
        return SystemFitReport(
            request.inspection_id,
            request.scope,
            active_release_id,
            tuple(sorted(active_ids)),
            missing_capabilities,
            missing_probes,
            missing_controls,
            deficits,
            evidence,
            needs_build,
        )


def signed_inspection_goal_sha256(request: SystemInspectionRequest) -> str:
    if not isinstance(request, SystemInspectionRequest):
        raise TypeError("request must be a SystemInspectionRequest")
    return hashlib.sha256(request.canonical_json().encode("utf-8")).hexdigest()