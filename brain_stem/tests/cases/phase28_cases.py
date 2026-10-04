from __future__ import annotations

import base64
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from src.swarm_core.model_catalog import ModelRole
from src.swarm_core.model_lifecycle import (
    ArtifactKind,
    ArtifactIntegrityError,
    CapabilityArtifactBinding,
    CollectiveReleaseManifest,
    Ed25519ArtifactManifestVerifier,
    ModelArtifactManifest,
    ModelArtifactRegistry,
    artifact_manifest_message,
    collective_release_message,
    release_activation_message,
)
from src.swarm_core.stacey_bundle import StaceyBundleBuilder, StaceyBundleError, StaceyBundleVerifier


class StaceyBundleBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        root = Path(self.temporary_directory.name)
        self.root = root
        self.artifact_root = root / "artifacts"
        self.artifact_root.mkdir()
        self.private_key = Ed25519PrivateKey.generate()
        self.public_key = self.private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        self.registry = ModelArtifactRegistry(
            root / "registry.sqlite3",
            self.artifact_root,
            manifest_verifier=Ed25519ArtifactManifestVerifier(self.public_key),
        )
        self.addCleanup(self.registry.close)
        self.core = self.register_artifact(
            "core-test",
            "core.bin",
            b"scratch core weights",
            role=ModelRole.WORLD_MODEL_CORE,
            capability_ids=("core.reason",),
            training_lineage="STACEY_SCRATCH",
        )
        self.specialist = self.register_artifact(
            "vision-test",
            "vision.bin",
            b"approved vision weights",
            role=ModelRole.SPECIALIST,
            capability_ids=("vision.inspect",),
            training_lineage="EXTERNAL_LICENSED:review-test",
        )
        release = CollectiveReleaseManifest(
            release_id="release-bundle-test",
            version="1.0.0",
            core_artifact_id=self.core.artifact_id,
            required_capability_ids=("vision.inspect",),
            capability_bindings=(
                CapabilityArtifactBinding(
                    "vision.inspect",
                    self.specialist.artifact_id,
                    "1.0",
                    hashlib.sha256(b"vision-contract").hexdigest(),
                ),
            ),
            approval_reference="release-approval:test",
            rollback_release_id=None,
        )
        self.registry.register_release(release, self.private_key.sign(collective_release_message(release)))
        self.registry.activate_release(
            release.release_id,
            self.private_key.sign(release_activation_message(release.release_id, None, "ACTIVATE")),
        )

    def register_artifact(
        self,
        artifact_id: str,
        relative_path: str,
        content: bytes,
        *,
        role: ModelRole,
        capability_ids: tuple[str, ...],
        training_lineage: str,
    ) -> ModelArtifactManifest:
        (self.artifact_root / relative_path).write_bytes(content)
        manifest = ModelArtifactManifest(
            artifact_id=artifact_id,
            version="1.0.0",
            kind=ArtifactKind.FULL_MODEL,
            role=role,
            relative_path=relative_path,
            artifact_sha256=hashlib.sha256(content).hexdigest(),
            artifact_size_bytes=len(content),
            capability_ids=capability_ids,
            backend_id="backend.test",
            architecture_id="architecture.test",
            config_sha256=hashlib.sha256(f"{artifact_id}:config".encode()).hexdigest(),
            tokenizer_sha256=hashlib.sha256(f"{artifact_id}:tokenizer".encode()).hexdigest(),
            license_reference="license:test",
            provenance_reference=f"provenance:{artifact_id}",
            training_lineage=training_lineage,
            evaluation_reference=f"evaluation:{artifact_id}",
            approval_reference=f"approval:{artifact_id}",
            resource_profile_name="model-resident",
            resource_domain="host-memory",
            reserved_bytes=1024,
        )
        signature = self.private_key.sign(artifact_manifest_message(manifest))
        self.registry.register(manifest, signature)
        return manifest

    def make_builder(self, maximum_bundle_bytes: int = 1_000_000) -> StaceyBundleBuilder:
        return StaceyBundleBuilder(
            registry=self.registry,
            maximum_component_bytes=100_000,
            maximum_bundle_bytes=maximum_bundle_bytes,
        )

    def test_bundle_is_deterministic_and_contains_verified_active_release_components(self) -> None:
        first_path = self.root / "first.stacey"
        second_path = self.root / "second.stacey"
        first = self.make_builder().build_active(first_path)
        second = self.make_builder().build_active(second_path)

        self.assertEqual(first.bundle_sha256, second.bundle_sha256)
        self.assertEqual(first.artifact_ids, ("core-test", "vision-test"))
        with zipfile.ZipFile(first_path, "r") as archive:
            self.assertEqual(
                archive.namelist(),
                [
                    "manifest.json",
                    f"components/{self.core.artifact_sha256}.bin",
                    f"components/{self.specialist.artifact_sha256}.bin",
                ],
            )
            manifest_payload = json.loads(archive.read("manifest.json"))
            release = CollectiveReleaseManifest.from_payload(manifest_payload["release"])
            release_signature = base64.b64decode(manifest_payload["release_signature_base64"], validate=True)
            verifier = Ed25519ArtifactManifestVerifier(self.public_key)
            self.assertTrue(verifier.verify_message(collective_release_message(release), release_signature))
            for component in manifest_payload["components"]:
                component_manifest = ModelArtifactManifest.from_payload(component["manifest"])
                signature = base64.b64decode(component["manifest_signature_base64"], validate=True)
                self.assertTrue(verifier.verify(component_manifest, signature))
                artifact_bytes = archive.read(component["bundle_path"])
                self.assertEqual(hashlib.sha256(artifact_bytes).hexdigest(), component_manifest.artifact_sha256)

        self.assertEqual(first.release_id, "release-bundle-test")
        self.assertEqual(first_path.stat().st_size, first.bundle_size_bytes)
        verified = StaceyBundleVerifier(
            signature_verifier=Ed25519ArtifactManifestVerifier(self.public_key),
            maximum_bundle_bytes=1_000_000,
            maximum_component_bytes=100_000,
        ).verify(first_path)
        self.assertEqual(verified.release.release_id, "release-bundle-test")
        self.assertEqual(
            tuple(component.manifest.artifact_id for component in verified.components),
            ("core-test", "vision-test"),
        )

    def test_tampered_registered_bytes_fail_and_leave_no_bundle(self) -> None:
        (self.artifact_root / "vision.bin").write_bytes(b"changed bytes")
        destination = self.root / "rejected.stacey"

        with self.assertRaises(ArtifactIntegrityError):
            self.make_builder().build_active(destination)

        self.assertFalse(destination.exists())

    def test_bundle_size_limit_and_existing_targets_are_enforced(self) -> None:
        with self.assertRaisesRegex(StaceyBundleError, "size limit"):
            self.make_builder(maximum_bundle_bytes=100).build_active(self.root / "too-large.stacey")
        target = self.root / "already-exists.stacey"
        target.write_bytes(b"user file")
        with self.assertRaisesRegex(StaceyBundleError, "already exists"):
            self.make_builder().build_active(target)
        self.assertEqual(target.read_bytes(), b"user file")

    def test_bundle_verifier_rejects_component_byte_tampering(self) -> None:
        bundle_path = self.root / "original.stacey"
        self.make_builder().build_active(bundle_path)
        tampered_path = self.root / "tampered.stacey"
        with zipfile.ZipFile(bundle_path, "r") as source, zipfile.ZipFile(tampered_path, "w") as target:
            for name in source.namelist():
                content = source.read(name)
                if name.endswith(f"{self.specialist.artifact_sha256}.bin"):
                    content = b"tampered signed specialist"
                target.writestr(name, content)

        verifier = StaceyBundleVerifier(
            signature_verifier=Ed25519ArtifactManifestVerifier(self.public_key),
            maximum_bundle_bytes=1_000_000,
            maximum_component_bytes=100_000,
        )
        with self.assertRaisesRegex(StaceyBundleError, "component size|component bytes"):
            verifier.verify(tampered_path)


if __name__ == "__main__":
    unittest.main()