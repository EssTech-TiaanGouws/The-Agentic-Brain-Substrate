from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from .model_lifecycle import (
    ArtifactIntegrityError,
    ArtifactKind,
    ArtifactManifestVerifier,
    CollectiveReleaseManifest,
    ModelRole,
    ModelArtifactManifest,
    ModelArtifactRegistry,
    collective_release_message,
    artifact_manifest_message,
)


class StaceyBundleError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class StaceyBundleReport:
    bundle_path: str
    bundle_sha256: str
    bundle_size_bytes: int
    release_id: str
    artifact_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class VerifiedBundleComponent:
    manifest: ModelArtifactManifest
    signature: bytes
    content: bytes


@dataclass(frozen=True, slots=True)
class VerifiedStaceyBundle:
    release: CollectiveReleaseManifest
    release_signature: bytes
    components: tuple[VerifiedBundleComponent, ...]
    bundle_sha256: str
    bundle_size_bytes: int


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise StaceyBundleError("bundle manifest contains duplicate JSON keys")
        result[key] = value
    return result


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o600) << 16
    info.flag_bits |= 0x800
    return info


class StaceyBundleBuilder:
    """Builds one portable bundle file from the exact active signed Stacey release."""

    def __init__(
        self,
        *,
        registry: ModelArtifactRegistry,
        maximum_component_bytes: int,
        maximum_bundle_bytes: int,
        file_mode: int = 0o600,
    ) -> None:
        if not isinstance(registry, ModelArtifactRegistry):
            raise ValueError("registry must be a ModelArtifactRegistry")
        for name, value in (
            ("maximum_component_bytes", maximum_component_bytes),
            ("maximum_bundle_bytes", maximum_bundle_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(file_mode, bool) or not isinstance(file_mode, int):
            raise ValueError("file_mode must be an integer permission mode")
        allowed = stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO
        if file_mode < 0 or file_mode & ~allowed:
            raise ValueError("file_mode contains unsupported permission bits")
        self._registry = registry
        self._maximum_component_bytes = maximum_component_bytes
        self._maximum_bundle_bytes = maximum_bundle_bytes
        self._file_mode = file_mode

    def build_active(self, destination: str | os.PathLike[str]) -> StaceyBundleReport:
        target = Path(destination)
        if target.is_symlink() or not target.name or not target.parent.is_dir():
            raise StaceyBundleError("bundle target must be a non-symlink file in an existing directory")
        if target.exists():
            raise StaceyBundleError("bundle target already exists")
        release = self._registry.active_release()
        if release is None:
            raise StaceyBundleError("no signed active collective release is available")
        release, release_signature = self._get_signed_release(release.release_id)
        artifact_ids = tuple(sorted({release.core_artifact_id, *(binding.artifact_id for binding in release.capability_bindings)}))
        signed_artifacts = tuple(self._registry.get_signed(artifact_id) for artifact_id in artifact_ids)
        total_declared_bytes = sum(manifest.artifact_size_bytes for manifest, _ in signed_artifacts)
        if any(manifest.artifact_size_bytes > self._maximum_component_bytes for manifest, _ in signed_artifacts):
            raise StaceyBundleError("a release component exceeds the configured size limit")
        if total_declared_bytes > self._maximum_bundle_bytes:
            raise StaceyBundleError("release components exceed the configured bundle-size limit")

        manifest_payload = {
            "format_id": "stacey.model.bundle.v1",
            "model_id": "stacey",
            "release": release.to_payload(),
            "release_signature_base64": base64.b64encode(release_signature).decode("ascii"),
            "components": [
                {
                    "artifact_id": manifest.artifact_id,
                    "artifact_sha256": manifest.artifact_sha256,
                    "manifest": manifest.to_payload(),
                    "manifest_signature_base64": base64.b64encode(signature).decode("ascii"),
                    "bundle_path": f"components/{manifest.artifact_sha256}.bin",
                }
                for manifest, signature in signed_artifacts
            ],
        }
        temp_descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".pending",
            dir=target.parent,
        )
        os.close(temp_descriptor)
        temp_path = Path(temp_name)
        try:
            os.chmod(temp_path, self._file_mode)
            with zipfile.ZipFile(temp_path, mode="w", allowZip64=True) as archive:
                archive.writestr(_zip_info("manifest.json"), _canonical_json(manifest_payload))
                for manifest, _ in signed_artifacts:
                    self._write_verified_component(archive, manifest)
            if temp_path.stat().st_size > self._maximum_bundle_bytes:
                raise StaceyBundleError("final bundle archive exceeds the configured size limit")
            current_release = self._registry.active_release()
            if current_release is None or current_release.release_id != release.release_id:
                raise StaceyBundleError("active release changed while the bundle was being assembled")
            digest = hashlib.sha256()
            size = 0
            with temp_path.open("rb") as bundle_file:
                while chunk := bundle_file.read(1024 * 1024):
                    digest.update(chunk)
                    size += len(chunk)
                os.fsync(bundle_file.fileno())
            try:
                os.link(temp_path, target, follow_symlinks=False)
            except FileExistsError as error:
                raise StaceyBundleError("bundle target appeared during assembly") from error
            directory_descriptor = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
            temp_path.unlink()
            return StaceyBundleReport(
                str(target),
                digest.hexdigest(),
                size,
                release.release_id,
                artifact_ids,
            )
        except (StaceyBundleError, ArtifactIntegrityError):
            temp_path.unlink(missing_ok=True)
            raise
        except Exception as error:
            temp_path.unlink(missing_ok=True)
            raise StaceyBundleError("active release bundle could not be assembled") from error

    def _get_signed_release(self, release_id: str) -> tuple[CollectiveReleaseManifest, bytes]:
        try:
            return self._registry.get_release_signed(release_id)
        except Exception as error:
            raise StaceyBundleError("active release signature could not be verified") from error

    def _write_verified_component(
        self,
        archive: zipfile.ZipFile,
        manifest: ModelArtifactManifest,
    ) -> None:
        name = f"components/{manifest.artifact_sha256}.bin"
        try:
            with self._registry.open_verified(manifest.artifact_id) as (verified_manifest, stream):
                if verified_manifest != manifest:
                    raise ArtifactIntegrityError("artifact manifest changed during bundle assembly")
                info = _zip_info(name)
                with archive.open(info, mode="w", force_zip64=True) as component:
                    digest = hashlib.sha256()
                    size = 0
                    while chunk := stream.read(1024 * 1024):
                        digest.update(chunk)
                        size += len(chunk)
                        if size > self._maximum_component_bytes:
                            raise StaceyBundleError("artifact exceeded its configured component limit")
                        component.write(chunk)
                if digest.hexdigest() != manifest.artifact_sha256 or size != manifest.artifact_size_bytes:
                    raise ArtifactIntegrityError("component bytes changed while entering the bundle")
        except (StaceyBundleError, ArtifactIntegrityError):
            raise
        except Exception as error:
            raise StaceyBundleError("signed release component could not be read") from error


class StaceyBundleVerifier:
    """Verifies one portable bundle without extraction, activation, or model loading."""

    _BUNDLE_FIELDS = frozenset(
        {"format_id", "model_id", "release", "release_signature_base64", "components"}
    )
    _COMPONENT_FIELDS = frozenset(
        {"artifact_id", "artifact_sha256", "manifest", "manifest_signature_base64", "bundle_path"}
    )

    def __init__(
        self,
        *,
        signature_verifier: ArtifactManifestVerifier,
        maximum_bundle_bytes: int,
        maximum_manifest_bytes: int = 1_000_000,
        maximum_component_bytes: int = 8_000_000_000,
        maximum_components: int = 64,
    ) -> None:
        if not callable(getattr(signature_verifier, "verify", None)) or not callable(
            getattr(signature_verifier, "verify_message", None)
        ):
            raise ValueError("signature_verifier must verify artifact manifests and signed messages")
        for name, value in (
            ("maximum_bundle_bytes", maximum_bundle_bytes),
            ("maximum_manifest_bytes", maximum_manifest_bytes),
            ("maximum_component_bytes", maximum_component_bytes),
            ("maximum_components", maximum_components),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self._signature_verifier = signature_verifier
        self._maximum_bundle_bytes = maximum_bundle_bytes
        self._maximum_manifest_bytes = maximum_manifest_bytes
        self._maximum_component_bytes = maximum_component_bytes
        self._maximum_components = maximum_components

    def verify(self, bundle_path: str | os.PathLike[str]) -> VerifiedStaceyBundle:
        path = Path(bundle_path)
        if path.is_symlink() or not path.is_file():
            raise StaceyBundleError("bundle must be a regular non-symlink file")
        bundle_size = path.stat().st_size
        if bundle_size <= 0 or bundle_size > self._maximum_bundle_bytes:
            raise StaceyBundleError("bundle file size is outside configured limits")
        bundle_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            with zipfile.ZipFile(path, mode="r") as archive:
                infos = archive.infolist()
                names = [info.filename for info in infos]
                if len(names) != len(set(names)) or "manifest.json" not in names:
                    raise StaceyBundleError("bundle contains duplicate members or no manifest")
                manifest_info = next(info for info in infos if info.filename == "manifest.json")
                if manifest_info.file_size > self._maximum_manifest_bytes:
                    raise StaceyBundleError("bundle manifest exceeds its configured limit")
                if manifest_info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
                    raise StaceyBundleError("bundle manifest uses an unsupported compression method")
                payload = json.loads(
                    archive.read(manifest_info),
                    object_pairs_hook=_strict_object,
                    parse_constant=lambda token: (_ for _ in ()).throw(
                        StaceyBundleError(f"bundle manifest contains invalid numeric constant {token}")
                    ),
                )
                if not isinstance(payload, dict) or set(payload) != self._BUNDLE_FIELDS:
                    raise StaceyBundleError("bundle manifest has missing or unknown fields")
                if payload["format_id"] != "stacey.model.bundle.v1" or payload["model_id"] != "stacey":
                    raise StaceyBundleError("unsupported model bundle format or identity")
                release = CollectiveReleaseManifest.from_payload(payload["release"])
                release_signature = base64.b64decode(payload["release_signature_base64"], validate=True)
                if not self._signature_verifier.verify_message(
                    collective_release_message(release),
                    release_signature,
                ):
                    raise StaceyBundleError("collective release signature is invalid")
                components_payload = payload["components"]
                if (
                    not isinstance(components_payload, list)
                    or not components_payload
                    or len(components_payload) > self._maximum_components
                ):
                    raise StaceyBundleError("bundle component count is outside configured limits")
                expected_ids = {release.core_artifact_id, *(binding.artifact_id for binding in release.capability_bindings)}
                components: list[VerifiedBundleComponent] = []
                seen_ids: set[str] = set()
                expected_paths = {"manifest.json"}
                total_uncompressed = manifest_info.file_size
                for entry in components_payload:
                    if not isinstance(entry, dict) or set(entry) != self._COMPONENT_FIELDS:
                        raise StaceyBundleError("bundle component entry has missing or unknown fields")
                    manifest = ModelArtifactManifest.from_payload(entry["manifest"])
                    if (
                        entry["artifact_id"] != manifest.artifact_id
                        or entry["artifact_sha256"] != manifest.artifact_sha256
                        or manifest.artifact_id in seen_ids
                    ):
                        raise StaceyBundleError("bundle component identity is inconsistent or duplicated")
                    seen_ids.add(manifest.artifact_id)
                    expected_path = f"components/{manifest.artifact_sha256}.bin"
                    if entry["bundle_path"] != expected_path:
                        raise StaceyBundleError("bundle component path is not canonical")
                    component_info = archive.getinfo(expected_path)
                    if component_info.is_dir() or component_info.file_size != manifest.artifact_size_bytes:
                        raise StaceyBundleError("bundle component size or member type differs from its manifest")
                    if component_info.file_size > self._maximum_component_bytes:
                        raise StaceyBundleError("bundle component exceeds its configured size limit")
                    if component_info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
                        raise StaceyBundleError("bundle component uses an unsupported compression method")
                    if component_info.compress_size == 0 or component_info.file_size / component_info.compress_size > 1000:
                        raise StaceyBundleError("bundle component compression ratio exceeds its safety limit")
                    component_signature = base64.b64decode(entry["manifest_signature_base64"], validate=True)
                    if not self._signature_verifier.verify(manifest, component_signature):
                        raise StaceyBundleError("component manifest signature is invalid")
                    content = archive.read(component_info)
                    if hashlib.sha256(content).hexdigest() != manifest.artifact_sha256:
                        raise StaceyBundleError("bundle component bytes do not match their signed manifest")
                    total_uncompressed += component_info.file_size
                    if total_uncompressed > self._maximum_bundle_bytes:
                        raise StaceyBundleError("total uncompressed bundle size exceeds its configured limit")
                    expected_paths.add(expected_path)
                    components.append(VerifiedBundleComponent(manifest, component_signature, content))
                if seen_ids != expected_ids or set(names) != expected_paths:
                    raise StaceyBundleError("bundle components do not exactly cover the signed release")
                self._validate_release_components(release, tuple(component.manifest for component in components))
                return VerifiedStaceyBundle(
                    release,
                    release_signature,
                    tuple(sorted(components, key=lambda component: component.manifest.artifact_id)),
                    bundle_digest,
                    bundle_size,
                )
        except StaceyBundleError:
            raise
        except (OSError, zipfile.BadZipFile, KeyError, ValueError, TypeError) as error:
            raise StaceyBundleError("bundle archive is malformed or invalid") from error

    @staticmethod
    def _validate_release_components(
        release: CollectiveReleaseManifest,
        components: tuple[ModelArtifactManifest, ...],
    ) -> None:
        by_id = {component.artifact_id: component for component in components}
        core = by_id.get(release.core_artifact_id)
        if (
            core is None
            or core.kind is not ArtifactKind.FULL_MODEL
            or core.role is not ModelRole.WORLD_MODEL_CORE
            or core.training_lineage != "STACEY_SCRATCH"
        ):
            raise StaceyBundleError("bundle does not contain a full scratch-trained Stacey Core")
        for binding in release.capability_bindings:
            component = by_id.get(binding.artifact_id)
            if component is None or binding.capability_id not in component.capability_ids:
                raise StaceyBundleError("bundle release capability does not match its signed artifact")
            if component.kind is ArtifactKind.LORA_ADAPTER and component.base_artifact_sha256 != core.artifact_sha256:
                raise StaceyBundleError("bundle adapter is bound to a different Core artifact")
            if component.kind is ArtifactKind.FULL_MODEL and component.role is not ModelRole.SPECIALIST:
                raise StaceyBundleError("bundle capability full-model artifact is not a specialist")