from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from substrate.contracts import Intent, ScopeVector
from src.swarm_core.identity import (
    Ed25519ScopeAuthorizationVerifier,
    SignedScopeGrant,
    intent_authorization_digest,
    scope_grant_message,
)
from src.swarm_core.hardware_profile import AcceleratorSnapshot, HardwareProfile
from src.swarm_core.model_catalog import ModelRole
from src.swarm_core.model_lifecycle import (
    ArtifactKind,
    CapabilityArtifactBinding,
    CollectiveReleaseManifest,
    Ed25519ArtifactManifestVerifier,
    ModelArtifactManifest,
    ModelArtifactRegistry,
    artifact_manifest_message,
    collective_release_message,
    release_activation_message,
)
from src.swarm_core.system_inspection import (
    CapabilityNeed,
    HostHardwareProbeProvider,
    SystemInspectionError,
    SystemInspectionRequest,
    SystemInspectionService,
    SystemProbeEvidence,
)


SCOPE = ScopeVector("tenant-a", "user-a", "project-a", "workspace-a")


class FakeProbeProvider:
    def __init__(self, evidence_factory) -> None:
        self.evidence_factory = evidence_factory
        self.calls = []

    def inspect(self, scope, probe_ids):
        self.calls.append((scope, probe_ids))
        return self.evidence_factory(scope, probe_ids)


class SystemInspectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        root = Path(self.temporary_directory.name)
        artifact_root = root / "artifacts"
        artifact_root.mkdir()
        self.private_key = Ed25519PrivateKey.generate()
        public_key = self.private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        self.registry = ModelArtifactRegistry(
            root / "registry.sqlite3",
            artifact_root,
            manifest_verifier=Ed25519ArtifactManifestVerifier(public_key),
        )
        self.addCleanup(self.registry.close)
        self._register_artifact(
            artifact_root,
            artifact_id="stacey-core-test",
            relative_path="core.bin",
            content=b"scratch-trained-core",
            role=ModelRole.WORLD_MODEL_CORE,
            capability_ids=("core.reason",),
            training_lineage="STACEY_SCRATCH",
        )
        self.specialist = self._register_artifact(
            artifact_root,
            artifact_id="vision-specialist-test",
            relative_path="vision.bin",
            content=b"approved-vision-specialist",
            role=ModelRole.SPECIALIST,
            capability_ids=("vision.inspect",),
            training_lineage="EXTERNAL_LICENSED:review-test",
        )
        contract_digest = hashlib.sha256(b"vision-contract-v1").hexdigest()
        release = CollectiveReleaseManifest(
            release_id="release-system-fit",
            version="1.0",
            core_artifact_id="stacey-core-test",
            required_capability_ids=("vision.inspect",),
            capability_bindings=(
                CapabilityArtifactBinding(
                    "vision.inspect",
                    self.specialist.artifact_id,
                    "1.0",
                    contract_digest,
                ),
            ),
            approval_reference="release-review:1",
            rollback_release_id=None,
        )
        self.registry.register_release(release, self.private_key.sign(collective_release_message(release)))
        self.registry.activate_release(
            release.release_id,
            self.private_key.sign(release_activation_message(release.release_id, None, "ACTIVATE")),
        )

        operator_key = Ed25519PrivateKey.generate()
        operator_public_key = operator_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        self.operator_key = operator_key
        self.verifier = Ed25519ScopeAuthorizationVerifier(
            operator_public_key,
            audience="system-inspection-test",
            clock=lambda: 100,
        )
        self.now_ns = 1_000

    def _register_artifact(
        self,
        artifact_root: Path,
        *,
        artifact_id: str,
        relative_path: str,
        content: bytes,
        role: ModelRole,
        capability_ids: tuple[str, ...],
        training_lineage: str,
    ) -> ModelArtifactManifest:
        (artifact_root / relative_path).write_bytes(content)
        manifest = ModelArtifactManifest(
            artifact_id=artifact_id,
            version="1.0",
            kind=ArtifactKind.FULL_MODEL,
            role=role,
            relative_path=relative_path,
            artifact_sha256=hashlib.sha256(content).hexdigest(),
            artifact_size_bytes=len(content),
            capability_ids=capability_ids,
            backend_id="backend.test",
            architecture_id="architecture.test",
            config_sha256=hashlib.sha256(b"config").hexdigest(),
            tokenizer_sha256=hashlib.sha256(b"tokenizer").hexdigest(),
            license_reference="license:test",
            provenance_reference="provenance:test",
            training_lineage=training_lineage,
            evaluation_reference="evaluation:test",
            approval_reference="artifact-review:test",
            resource_profile_name="model-memory",
            resource_domain="host-memory",
            reserved_bytes=100,
        )
        self.registry.register(manifest, self.private_key.sign(artifact_manifest_message(manifest)))
        return manifest

    def make_request(
        self,
        probe_ids: tuple[str, ...] = ("viewport.capture",),
        required_capabilities: tuple[CapabilityNeed, ...] | None = None,
    ):
        return SystemInspectionRequest(
            inspection_id="inspection-1",
            transaction_id="tx-inspect",
            correlation_id="turn-inspect",
            purpose="Check whether the approved visual workflow can run.",
            active_release_id="release-system-fit",
            scope=SCOPE,
            probe_ids=probe_ids,
            required_capabilities=required_capabilities
            or (CapabilityNeed("vision.inspect", ("read-only",), (("host-memory", 500),)),),
        )

    def sign_request(self, request: SystemInspectionRequest) -> SignedScopeGrant:
        intent = Intent(
            transaction_id=request.transaction_id,
            correlation_id=request.correlation_id,
            action="INSPECT_SYSTEM",
            goal=request.canonical_json(),
            scope=request.scope,
        )
        unsigned = SignedScopeGrant(
            scope=request.scope,
            action=intent.action,
            transaction_id=intent.transaction_id,
            correlation_id=intent.correlation_id,
            audience="system-inspection-test",
            issued_at=90,
            expires_at=110,
            grant_id="grant-inspection-1",
            request_sha256=intent_authorization_digest(intent),
            signature=b"",
        )
        return replace(unsigned, signature=self.operator_key.sign(scope_grant_message(unsigned)))

    def make_evidence(
        self,
        scope,
        probe_ids,
        *,
        controls=("read-only", "viewport-access"),
        resources=(("host-memory", 2_000),),
        unavailable_probe_ids=(),
    ):
        return tuple(
            SystemProbeEvidence(
                probe_id,
                scope,
                probe_id not in unavailable_probe_ids,
                f"local-probe:{probe_id}",
                hashlib.sha256(probe_id.encode("utf-8")).hexdigest(),
                self.now_ns,
                controls,
                resources,
            )
            for probe_id in probe_ids
        )

    def make_service(self, provider: FakeProbeProvider):
        return SystemInspectionService(
            authorization_verifier=self.verifier,
            artifact_registry=self.registry,
            probe_provider=provider,
            allowed_probe_ids=("viewport.capture", "dom.read", "audio.input"),
            maximum_evidence_age_ns=100,
            clock=lambda: self.now_ns,
        )

    def test_signed_inspection_uses_active_registry_and_reports_missing_bundle_parts(self) -> None:
        needs = (
            CapabilityNeed(
                "vision.inspect",
                ("read-only",),
                (("host-memory", 1_000),),
            ),
            CapabilityNeed(
                "audio.transcribe",
                ("microphone-consent",),
                (("host-memory", 4_000),),
            ),
        )
        request = self.make_request(("viewport.capture", "audio.input"), needs)
        provider = FakeProbeProvider(
            lambda scope, probes: self.make_evidence(
                scope,
                probes,
                unavailable_probe_ids=("audio.input",),
                controls=("read-only", "viewport-access"),
                resources=(("host-memory", 2_000),),
            )
        )
        service = self.make_service(provider)

        fit = service.inspect_and_assess(
            request,
            self.sign_request(request),
            active_release_id="release-system-fit",
        )

        self.assertEqual(fit.available_capability_ids, ("vision.inspect",))
        self.assertEqual(fit.missing_capability_ids, ("audio.transcribe",))
        self.assertIn("audio.input", fit.missing_probe_ids)
        self.assertEqual(fit.missing_control_ids, ("microphone-consent",))
        self.assertEqual(fit.resource_deficits[0].resource_domain, "host-memory")
        self.assertTrue(fit.build_request_required)
        self.assertEqual(len(provider.calls), 1)

    def test_complete_fit_does_not_request_a_build(self) -> None:
        request = self.make_request(
            required_capabilities=(CapabilityNeed("vision.inspect", ("read-only",), (("host-memory", 500),)),)
        )
        provider = FakeProbeProvider(self.make_evidence)
        service = self.make_service(provider)

        fit = service.inspect_and_assess(
            request,
            self.sign_request(request),
            active_release_id="release-system-fit",
        )

        self.assertFalse(fit.build_request_required)
        self.assertEqual(fit.missing_capability_ids, ())
        self.assertEqual(fit.resource_deficits, ())

    def test_invalid_grant_stale_scope_mismatch_and_unapproved_probe_are_denied(self) -> None:
        request = self.make_request()
        provider = FakeProbeProvider(self.make_evidence)
        service = self.make_service(provider)
        grant = self.sign_request(request)
        with self.assertRaises(SystemInspectionError):
            service.inspect_and_assess(
                request,
                replace(grant, signature=b""),
                active_release_id="release-system-fit",
            )
        self.assertEqual(provider.calls, [])

        stale_provider = FakeProbeProvider(
            lambda scope, probes: tuple(
                SystemProbeEvidence(
                    probe,
                    scope,
                    True,
                    "local-probe:stale",
                    hashlib.sha256(probe.encode()).hexdigest(),
                    0,
                    ("read-only",),
                    (),
                )
                for probe in probes
            )
        )
        with self.assertRaisesRegex(SystemInspectionError, "stale or from the future"):
            self.make_service(stale_provider).inspect_and_assess(
                request,
                grant,
                active_release_id="release-system-fit",
            )

        with self.assertRaisesRegex(SystemInspectionError, "outside the operator-approved allowlist"):
            bad_request = self.make_request(("arbitrary.shell",))
            self.make_service(provider).inspect_and_assess(
                bad_request,
                self.sign_request(bad_request),
                active_release_id="release-system-fit",
            )

    def test_probe_evidence_cannot_cross_scope_or_release(self) -> None:
        request = self.make_request()
        other_scope = ScopeVector("tenant-b", "user-a", "project-a", "workspace-a")
        provider = FakeProbeProvider(
            lambda probes_scope, probe_ids: self.make_evidence(other_scope, probe_ids)
        )
        with self.assertRaisesRegex(SystemInspectionError, "crossed the authorized scope"):
            self.make_service(provider).inspect_and_assess(
                request,
                self.sign_request(request),
                active_release_id="release-system-fit",
            )

        provider = FakeProbeProvider(self.make_evidence)
        with self.assertRaisesRegex(SystemInspectionError, "different signed release"):
            self.make_service(provider).inspect_and_assess(
                request,
                self.sign_request(request),
                active_release_id="another-release",
            )
        self.assertEqual(provider.calls, [])

    def test_host_hardware_probe_emits_measured_scoped_resource_evidence(self) -> None:
        profile = HardwareProfile(
            observed_at_ns=self.now_ns,
            operating_system="test-os",
            machine_architecture="arm64",
            logical_cpu_count=8,
            host_memory_total_bytes=16_000,
            host_memory_available_bytes=12_000,
            tensor_runtime="torch-test",
            cuda_runtime=None,
            accelerators=(
                AcceleratorSnapshot(
                    "cuda:0",
                    "cuda",
                    "test-gpu",
                    8_000,
                    6_000,
                    True,
                    False,
                ),
            ),
        )
        provider = HostHardwareProbeProvider(profile_provider=lambda: profile)

        evidence = provider.inspect(SCOPE, ("host.hardware",))

        self.assertEqual(evidence[0].scope, SCOPE)
        self.assertEqual(
            evidence[0].available_resources,
            (("device-memory:cuda:0", 6_000), ("host-memory", 12_000)),
        )
        self.assertIn("cpu-architecture:arm64", evidence[0].controls)
        self.assertIn("accelerator-backend:cuda", evidence[0].controls)
        self.assertEqual(evidence[0].source_sha256, hashlib.sha256(
            json.dumps(profile.to_payload(), allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest())
        with self.assertRaisesRegex(SystemInspectionError, "unimplemented probe"):
            provider.inspect(SCOPE, ("host.hardware", "arbitrary.shell"))


if __name__ == "__main__":
    unittest.main()