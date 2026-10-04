from __future__ import annotations

import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path
from time import time_ns

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from src.swarm_core.lease_manager import (
    ResourceAdmissionDenied,
    ResourceLeaseManager,
    ResourcePolicy,
    ResourceSnapshot,
)
from src.swarm_core.model_catalog import ModelRole
from src.swarm_core.model_lifecycle import (
    ArtifactApprovalError,
    ArtifactCompatibilityError,
    ArtifactIntegrityError,
    ArtifactKind,
    CapabilityArtifactBinding,
    CollectiveReleaseManifest,
    Ed25519ArtifactManifestVerifier,
    ModelArtifactManifest,
    ModelArtifactRegistry,
    ModelLifecycleError,
    ModelLifecycleManager,
    artifact_manifest_message,
    collective_release_message,
    release_activation_message,
)


class FakeBackend:
    backend_id = "backend.test"
    max_concurrent_leases = 8

    def __init__(self) -> None:
        self.unload_result = True
        self.confirm_result = True
        self.load_error: Exception | None = None
        self.load_calls: list[tuple[str, bytes, object | None]] = []
        self.unload_calls: list[object] = []

    def load(self, manifest, artifact, *, base_handle):
        if self.load_error is not None:
            raise self.load_error
        handle = {"artifact_id": manifest.artifact_id, "base_handle": base_handle}
        self.load_calls.append((manifest.artifact_id, artifact.read(), base_handle))
        return handle

    def infer(self, handle, request):
        return (handle["artifact_id"], request)

    def unload(self, handle):
        self.unload_calls.append(handle)
        return self.unload_result

    def confirm_released(self, handle):
        return self.confirm_result


def make_manifest(
    artifact_id: str,
    content: bytes,
    *,
    relative_path: str,
    role: ModelRole = ModelRole.SPECIALIST,
    kind: ArtifactKind = ArtifactKind.FULL_MODEL,
    capability_ids: tuple[str, ...] = ("text.summarize",),
    reserved_bytes: int = 100,
    training_lineage: str = "EXTERNAL_LICENSED:review-001",
    base_artifact_sha256: str | None = None,
    base_architecture_id: str | None = None,
    base_config_sha256: str | None = None,
    base_tokenizer_sha256: str | None = None,
) -> ModelArtifactManifest:
    return ModelArtifactManifest(
        artifact_id=artifact_id,
        version="1.0.0",
        kind=kind,
        role=role,
        relative_path=relative_path,
        artifact_sha256=hashlib.sha256(content).hexdigest(),
        artifact_size_bytes=len(content),
        capability_ids=capability_ids,
        backend_id=FakeBackend.backend_id,
        architecture_id="arch.test",
        config_sha256=hashlib.sha256(b"config").hexdigest(),
        tokenizer_sha256=hashlib.sha256(b"tokenizer").hexdigest(),
        license_reference="license-review-001",
        provenance_reference="provenance-001",
        training_lineage=training_lineage,
        evaluation_reference="evaluation-001",
        approval_reference="operator-approval-001",
        resource_profile_name="model-load",
        resource_domain="device-memory",
        reserved_bytes=reserved_bytes,
        base_artifact_sha256=base_artifact_sha256,
        base_architecture_id=base_architecture_id,
        base_config_sha256=base_config_sha256,
        base_tokenizer_sha256=base_tokenizer_sha256,
    )


class ModelLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.artifact_root = self.root / "artifacts"
        self.artifact_root.mkdir()
        self.private_key = Ed25519PrivateKey.generate()
        public_key = self.private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        self.manifest_verifier = Ed25519ArtifactManifestVerifier(public_key)
        self.registry = ModelArtifactRegistry(
            self.root / "registry.sqlite3",
            self.artifact_root,
            manifest_verifier=self.manifest_verifier,
        )
        self.addCleanup(self.registry.close)
        self.backend = FakeBackend()

    def write_artifact(self, relative_path: str, content: bytes) -> None:
        path = self.artifact_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def sign_and_register(self, manifest: ModelArtifactManifest) -> ModelArtifactManifest:
        signature = self.private_key.sign(artifact_manifest_message(manifest))
        return self.registry.register(manifest, signature)

    def make_manager(self, available_bytes: int = 1000) -> ModelLifecycleManager:
        leases = ResourceLeaseManager(
            ResourcePolicy("model-load", "device-memory", 0, 5_000_000_000),
            lambda: ResourceSnapshot("device-memory", available_bytes, time_ns()),
        )
        self.lease_manager = leases
        return ModelLifecycleManager(
            registry=self.registry,
            backends={FakeBackend.backend_id: self.backend},
            lease_manager=leases,
        )

    def add_specialist(
        self,
        artifact_id: str,
        content: bytes,
        *,
        reserved_bytes: int = 100,
        capability: str = "text.summarize",
    ) -> ModelArtifactManifest:
        relative_path = f"{artifact_id}.bin"
        self.write_artifact(relative_path, content)
        manifest = make_manifest(
            artifact_id,
            content,
            relative_path=relative_path,
            reserved_bytes=reserved_bytes,
            capability_ids=(capability,),
        )
        return self.sign_and_register(manifest)

    def test_registry_requires_valid_signature_and_exact_artifact_bytes(self) -> None:
        content = b"approved model bytes"
        self.write_artifact("model.bin", content)
        manifest = make_manifest("model-1", content, relative_path="model.bin")
        with self.assertRaises(ArtifactApprovalError):
            self.registry.register(manifest, b"x" * 64)

        self.write_artifact("model.bin", b"tampered model bytes")
        signature = self.private_key.sign(artifact_manifest_message(manifest))
        with self.assertRaises(ArtifactIntegrityError):
            self.registry.register(manifest, signature)

    def test_registry_persists_approved_manifest_and_detects_file_tampering(self) -> None:
        manifest = self.add_specialist("model-1", b"model bytes")
        self.assertEqual(self.registry.get("model-1"), manifest)
        self.registry.close()
        self.registry = ModelArtifactRegistry(
            self.root / "registry.sqlite3",
            self.artifact_root,
            manifest_verifier=self.manifest_verifier,
        )
        self.addCleanup(self.registry.close)
        self.assertEqual(self.registry.get("model-1"), manifest)

        self.write_artifact("model-1.bin", b"replaced bytes")
        with self.assertRaises(ArtifactIntegrityError):
            with self.registry.open_verified("model-1"):
                pass

    def test_core_manifest_requires_stacey_scratch_lineage(self) -> None:
        content = b"core"
        with self.assertRaisesRegex(ValueError, "scratch-training"):
            make_manifest(
                "core-invalid",
                content,
                relative_path="core.bin",
                role=ModelRole.WORLD_MODEL_CORE,
                capability_ids=("core.reason",),
                training_lineage="EXTERNAL_LICENSED:review-001",
            )

    def test_lora_manifest_must_match_exact_approved_core_base(self) -> None:
        core_bytes = b"scratch-trained-core"
        self.write_artifact("core.bin", core_bytes)
        core = make_manifest(
            "core-1",
            core_bytes,
            relative_path="core.bin",
            role=ModelRole.WORLD_MODEL_CORE,
            capability_ids=("core.reason",),
            reserved_bytes=300,
            training_lineage="STACEY_SCRATCH",
        )
        self.sign_and_register(core)
        adapter_bytes = b"adapter"
        self.write_artifact("adapter.bin", adapter_bytes)
        incompatible = make_manifest(
            "adapter-bad",
            adapter_bytes,
            relative_path="adapter.bin",
            kind=ArtifactKind.LORA_ADAPTER,
            capability_ids=("reason.specialized",),
            base_artifact_sha256=core.artifact_sha256,
            base_architecture_id="other-arch",
            base_config_sha256=core.config_sha256,
            base_tokenizer_sha256=core.tokenizer_sha256,
        )
        signature = self.private_key.sign(artifact_manifest_message(incompatible))
        with self.assertRaises(ArtifactCompatibilityError):
            self.registry.register(incompatible, signature)

    def test_adapter_lease_loads_with_exact_core_and_holds_its_lease(self) -> None:
        core_bytes = b"scratch-trained-core"
        self.write_artifact("core.bin", core_bytes)
        core = make_manifest(
            "core-1",
            core_bytes,
            relative_path="core.bin",
            role=ModelRole.WORLD_MODEL_CORE,
            capability_ids=("core.reason",),
            reserved_bytes=300,
            training_lineage="STACEY_SCRATCH",
        )
        self.sign_and_register(core)
        adapter_bytes = b"adapter"
        self.write_artifact("adapter.bin", adapter_bytes)
        adapter = make_manifest(
            "adapter-1",
            adapter_bytes,
            relative_path="adapter.bin",
            kind=ArtifactKind.LORA_ADAPTER,
            capability_ids=("reason.specialized",),
            reserved_bytes=20,
            base_artifact_sha256=core.artifact_sha256,
            base_architecture_id=core.architecture_id,
            base_config_sha256=core.config_sha256,
            base_tokenizer_sha256=core.tokenizer_sha256,
        )
        self.sign_and_register(adapter)
        manager = self.make_manager()

        with manager.acquire("reason.specialized") as lease:
            self.assertEqual(lease.infer("query"), ("adapter-1", "query"))
            adapter_load = next(call for call in self.backend.load_calls if call[0] == "adapter-1")
            self.assertIsNotNone(adapter_load[2])
            self.assertIn("core-1", manager.resident_artifact_ids)
        self.assertEqual(manager.resident_artifact_ids, ("adapter-1", "core-1"))

    def test_active_lease_prevents_eviction_and_idle_specialist_is_evicted_for_capacity(self) -> None:
        self.add_specialist("specialist-a", b"a" * 40, reserved_bytes=80)
        self.add_specialist("specialist-b", b"b" * 40, reserved_bytes=80, capability="text.translate")
        manager = self.make_manager(available_bytes=100)

        first = manager.acquire("text.summarize")
        with self.assertRaises(ModelLifecycleError):
            manager.evict("specialist-a")
        first.release()

        with manager.acquire("text.translate"):
            pass

        self.assertEqual(manager.resident_artifact_ids, ("specialist-b",))
        self.assertEqual(len(self.backend.unload_calls), 1)

    def test_resident_sharing_respects_backend_concurrency_limit(self) -> None:
        self.add_specialist("specialist-a", b"a" * 40)
        self.backend.max_concurrent_leases = 1
        manager = self.make_manager()
        first = manager.acquire("text.summarize")
        with self.assertRaises(ModelLifecycleError):
            manager.acquire("text.summarize")
        first.release()
        with manager.acquire("text.summarize"):
            pass

    def test_failed_unload_quarantines_capacity_until_release_is_confirmed(self) -> None:
        self.add_specialist("specialist-a", b"a" * 40, reserved_bytes=80)
        manager = self.make_manager(available_bytes=100)
        lease = manager.acquire("text.summarize")
        lease.release()
        self.backend.unload_result = False
        self.backend.confirm_result = False

        self.assertFalse(manager.evict("specialist-a"))
        self.assertEqual(manager.quarantined_artifact_ids, ("specialist-a",))
        self.assertEqual(self.lease_manager.reserved_bytes, 80)
        with self.assertRaises(ModelLifecycleError):
            manager.acquire("text.summarize")

        self.backend.confirm_result = True
        self.assertTrue(manager.reconcile_quarantine("specialist-a"))
        self.assertEqual(self.lease_manager.reserved_bytes, 0)

    def test_load_failure_with_uncertain_cleanup_is_quarantined(self) -> None:
        self.add_specialist("specialist-a", b"a" * 40, reserved_bytes=80)
        manager = self.make_manager(available_bytes=100)

        class UncertainLoadError(RuntimeError):
            resources_released = False
            handle = object()

        self.backend.load_error = UncertainLoadError("partial allocation")
        with self.assertRaises(ModelLifecycleError):
            manager.acquire("text.summarize")
        self.assertEqual(manager.quarantined_artifact_ids, ("specialist-a",))
        self.assertEqual(self.lease_manager.reserved_bytes, 80)

    def _register_release_pair(self, release_id: str, rollback_release_id: str | None):
        core_bytes = f"scratch-core-{release_id}".encode("utf-8")
        core_path = f"{release_id}-core.bin"
        self.write_artifact(core_path, core_bytes)
        core = make_manifest(
            f"{release_id}-core",
            core_bytes,
            relative_path=core_path,
            role=ModelRole.WORLD_MODEL_CORE,
            capability_ids=("core.reason",),
            reserved_bytes=300,
            training_lineage="STACEY_SCRATCH",
        )
        self.sign_and_register(core)
        specialist_bytes = f"specialist-{release_id}".encode("utf-8")
        specialist_path = f"{release_id}-specialist.bin"
        self.write_artifact(specialist_path, specialist_bytes)
        specialist = make_manifest(
            f"{release_id}-specialist",
            specialist_bytes,
            relative_path=specialist_path,
            capability_ids=("text.summarize",),
        )
        self.sign_and_register(specialist)
        release = CollectiveReleaseManifest(
            release_id=release_id,
            version="1.0.0",
            core_artifact_id=core.artifact_id,
            required_capability_ids=("text.summarize",),
            capability_bindings=(
                CapabilityArtifactBinding(
                    "text.summarize",
                    specialist.artifact_id,
                    "1.0",
                    hashlib.sha256(b"text-summarize-contract-v1").hexdigest(),
                ),
            ),
            approval_reference=f"release-approval:{release_id}",
            rollback_release_id=rollback_release_id,
        )
        signature = self.private_key.sign(collective_release_message(release))
        self.registry.register_release(release, signature)
        return release

    def test_release_requires_complete_capability_bindings_and_signed_artifacts(self) -> None:
        with self.assertRaisesRegex(ValueError, "every required capability"):
            CollectiveReleaseManifest(
                release_id="incomplete-release",
                version="1.0.0",
                core_artifact_id="missing-core",
                required_capability_ids=("text.summarize", "image.inspect"),
                capability_bindings=(
                    CapabilityArtifactBinding(
                        "text.summarize",
                        "missing-specialist",
                        "1.0",
                        hashlib.sha256(b"contract").hexdigest(),
                    ),
                ),
                approval_reference="approval",
            )

    def test_signed_release_activation_and_rollback_are_durable(self) -> None:
        first = self._register_release_pair("release-1", None)
        activate_first = self.private_key.sign(
            release_activation_message(first.release_id, None, "ACTIVATE")
        )
        self.registry.activate_release(first.release_id, activate_first)

        second = self._register_release_pair("release-2", first.release_id)
        activate_second = self.private_key.sign(
            release_activation_message(second.release_id, first.release_id, "ACTIVATE")
        )
        self.registry.activate_release(second.release_id, activate_second)
        self.assertEqual(self.registry.active_release(), second)

        rollback = self.private_key.sign(
            release_activation_message(first.release_id, second.release_id, "ROLLBACK")
        )
        self.registry.activate_release(first.release_id, rollback, rollback=True)
        self.assertEqual(self.registry.active_release(), first)

    def test_active_release_pins_capability_to_its_exact_artifact(self) -> None:
        release = self._register_release_pair("release-1", None)
        self.add_specialist(
            "aaa-unreleased-competitor",
            b"x",
            reserved_bytes=10,
            capability="text.summarize",
        )
        activation = self.private_key.sign(
            release_activation_message(release.release_id, None, "ACTIVATE")
        )
        self.registry.activate_release(release.release_id, activation)
        manager = self.make_manager()

        with manager.acquire_from_active_release(
            "text.summarize",
            contract_version="1.0",
            contract_sha256=hashlib.sha256(b"text-summarize-contract-v1").hexdigest(),
            resource_domain="device-memory",
            estimated_required_bytes=10,
        ) as lease:
            self.assertEqual(lease.infer("query"), ("release-1-specialist", "query"))
        self.assertEqual(
            tuple(call[0] for call in self.backend.load_calls),
            ("release-1-specialist",),
        )

    def test_active_release_rejects_contract_or_resource_drift_before_loading(self) -> None:
        release = self._register_release_pair("release-1", None)
        activation = self.private_key.sign(
            release_activation_message(release.release_id, None, "ACTIVATE")
        )
        self.registry.activate_release(release.release_id, activation)
        manager = self.make_manager()
        valid = {
            "contract_version": "1.0",
            "contract_sha256": hashlib.sha256(b"text-summarize-contract-v1").hexdigest(),
            "resource_domain": "device-memory",
            "estimated_required_bytes": 10,
        }
        invalid_declarations = (
            {**valid, "contract_version": "2.0"},
            {**valid, "contract_sha256": hashlib.sha256(b"different-schema").hexdigest()},
            {**valid, "resource_domain": "host-memory"},
            {**valid, "estimated_required_bytes": 101},
        )

        for declaration in invalid_declarations:
            with self.subTest(declaration=declaration):
                with self.assertRaises(ArtifactCompatibilityError):
                    manager.acquire_from_active_release("text.summarize", **declaration)

        self.assertEqual(self.backend.load_calls, [])
        with manager.acquire_from_active_release("text.summarize", **valid):
            pass
        self.assertEqual(len(self.backend.load_calls), 1)

    def test_pinned_release_uses_its_core_and_specialist_after_activation_changes(self) -> None:
        first = self._register_release_pair("release-1", None)
        activate_first = self.private_key.sign(
            release_activation_message(first.release_id, None, "ACTIVATE")
        )
        self.registry.activate_release(first.release_id, activate_first)
        second = self._register_release_pair("release-2", first.release_id)
        activate_second = self.private_key.sign(
            release_activation_message(second.release_id, first.release_id, "ACTIVATE")
        )
        self.registry.activate_release(second.release_id, activate_second)
        manager = self.make_manager()

        self.assertEqual(
            manager.core_artifact_sha256_for_release(first.release_id),
            self.registry.get(first.core_artifact_id).artifact_sha256,
        )
        with manager.acquire_from_active_release(
            "text.summarize",
            release_id=first.release_id,
            contract_version="1.0",
            contract_sha256=hashlib.sha256(b"text-summarize-contract-v1").hexdigest(),
            resource_domain="device-memory",
            estimated_required_bytes=10,
        ) as lease:
            self.assertEqual(lease.infer("query"), ("release-1-specialist", "query"))
        self.assertEqual(
            tuple(call[0] for call in self.backend.load_calls),
            ("release-1-specialist",),
        )

    def test_core_acquisition_is_pinned_and_core_cannot_be_evicted(self) -> None:
        release = self._register_release_pair("release-1", None)
        activation = self.private_key.sign(
            release_activation_message(release.release_id, None, "ACTIVATE")
        )
        self.registry.activate_release(release.release_id, activation)
        manager = self.make_manager()
        core_manifest = self.registry.get(release.core_artifact_id)

        with manager.acquire_core_from_release(release.release_id) as core:
            self.assertEqual(core.artifact_id, core_manifest.artifact_id)
            self.assertEqual(core.artifact_sha256, core_manifest.artifact_sha256)
            self.assertEqual(core.infer("decision"), (core_manifest.artifact_id, "decision"))
        with self.assertRaises(ModelLifecycleError):
            manager.evict(core_manifest.artifact_id)

    def test_release_activation_requires_current_rollback_target_and_valid_signature(self) -> None:
        first = self._register_release_pair("release-1", None)
        with self.assertRaises(ArtifactApprovalError):
            self.registry.activate_release(first.release_id, b"invalid")

        second = self._register_release_pair("release-2", "not-active")
        with self.assertRaises(ArtifactCompatibilityError):
            self.registry.activate_release(
                second.release_id,
                self.private_key.sign(
                    release_activation_message(second.release_id, None, "ACTIVATE")
                ),
            )

    def test_tampered_activation_history_is_rejected(self) -> None:
        release = self._register_release_pair("release-1", None)
        signature = self.private_key.sign(
            release_activation_message(release.release_id, None, "ACTIVATE")
        )
        self.registry.activate_release(release.release_id, signature)
        connection = sqlite3.connect(self.root / "registry.sqlite3")
        connection.execute("DROP TRIGGER release_activations_no_update")
        connection.execute("UPDATE release_activations SET event_hash = ?", ("0" * 64,))
        connection.commit()
        connection.close()
        with self.assertRaises(ArtifactIntegrityError):
            self.registry.active_release()


if __name__ == "__main__":
    unittest.main()