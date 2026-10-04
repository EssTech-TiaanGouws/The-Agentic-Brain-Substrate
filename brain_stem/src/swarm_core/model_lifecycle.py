from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from threading import RLock
from time import time_ns
from typing import BinaryIO, Callable, Iterator, Mapping, Protocol

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .lease_manager import (
    ResourceAdmissionDenied,
    ResourceLease,
    ResourceLeaseManager,
    ResourceProfile,
)
from .model_catalog import ModelRole


class ArtifactRegistryError(RuntimeError):
    pass


class ArtifactIntegrityError(ArtifactRegistryError):
    pass


class ArtifactApprovalError(ArtifactRegistryError):
    pass


class ArtifactNotFoundError(ArtifactRegistryError):
    pass


class ArtifactCompatibilityError(ArtifactRegistryError):
    pass


class ModelLifecycleError(RuntimeError):
    pass


class ModelLoadError(ModelLifecycleError):
    pass


class ModelQuarantinedError(ModelLifecycleError):
    pass


class ArtifactKind(str, Enum):
    FULL_MODEL = "FULL_MODEL"
    LORA_ADAPTER = "LORA_ADAPTER"


class ResidentState(str, Enum):
    RESIDENT = "RESIDENT"
    QUARANTINED = "QUARANTINED"


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _digest(value: object, field_name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("Artifact manifest must contain finite JSON-compatible values") from error


@dataclass(frozen=True, slots=True)
class ModelArtifactManifest:
    artifact_id: str
    version: str
    kind: ArtifactKind
    role: ModelRole
    relative_path: str
    artifact_sha256: str
    artifact_size_bytes: int
    capability_ids: tuple[str, ...]
    backend_id: str
    architecture_id: str
    config_sha256: str
    tokenizer_sha256: str
    license_reference: str
    provenance_reference: str
    training_lineage: str
    evaluation_reference: str
    approval_reference: str
    resource_profile_name: str
    resource_domain: str
    reserved_bytes: int
    base_artifact_sha256: str | None = None
    base_architecture_id: str | None = None
    base_config_sha256: str | None = None
    base_tokenizer_sha256: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "artifact_id",
            "version",
            "backend_id",
            "architecture_id",
            "license_reference",
            "provenance_reference",
            "training_lineage",
            "evaluation_reference",
            "approval_reference",
            "resource_profile_name",
            "resource_domain",
        ):
            _required_text(getattr(self, field_name), field_name)
        if not isinstance(self.kind, ArtifactKind) or not isinstance(self.role, ModelRole):
            raise ValueError("Artifact kind and role must use their declared enums")
        _digest(self.artifact_sha256, "artifact_sha256")
        _digest(self.config_sha256, "config_sha256")
        _digest(self.tokenizer_sha256, "tokenizer_sha256")
        for field_name in ("artifact_size_bytes", "reserved_bytes"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if not isinstance(self.capability_ids, tuple) or not self.capability_ids:
            raise ValueError("capability_ids must be a non-empty tuple")
        if any(not isinstance(item, str) or not item.strip() for item in self.capability_ids):
            raise ValueError("capability_ids must contain non-empty strings")
        if len(set(self.capability_ids)) != len(self.capability_ids):
            raise ValueError("capability_ids must not contain duplicates")
        if self.role is ModelRole.WORLD_MODEL_CORE and self.training_lineage != "STACEY_SCRATCH":
            raise ValueError("The Stacey Core artifact must declare Stacey scratch-training lineage")
        path = PurePosixPath(self.relative_path)
        if (
            not isinstance(self.relative_path, str)
            or not self.relative_path
            or "\\" in self.relative_path
            or path.is_absolute()
            or path.as_posix() != self.relative_path
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("relative_path must be a normalized path within the artifact store")
        base_fields = (
            self.base_artifact_sha256,
            self.base_architecture_id,
            self.base_config_sha256,
            self.base_tokenizer_sha256,
        )
        if self.kind is ArtifactKind.LORA_ADAPTER:
            if any(value is None for value in base_fields):
                raise ValueError("LoRA adapters must bind an exact base artifact/config/tokenizer")
            _digest(self.base_artifact_sha256, "base_artifact_sha256")
            _digest(self.base_config_sha256, "base_config_sha256")
            _digest(self.base_tokenizer_sha256, "base_tokenizer_sha256")
            _required_text(self.base_architecture_id, "base_architecture_id")
        elif any(value is not None for value in base_fields):
            raise ValueError("Full models cannot declare adapter base compatibility fields")

    def to_payload(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "version": self.version,
            "kind": self.kind.value,
            "role": self.role.value,
            "relative_path": self.relative_path,
            "artifact_sha256": self.artifact_sha256,
            "artifact_size_bytes": self.artifact_size_bytes,
            "capability_ids": list(self.capability_ids),
            "backend_id": self.backend_id,
            "architecture_id": self.architecture_id,
            "config_sha256": self.config_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "license_reference": self.license_reference,
            "provenance_reference": self.provenance_reference,
            "training_lineage": self.training_lineage,
            "evaluation_reference": self.evaluation_reference,
            "approval_reference": self.approval_reference,
            "resource_profile_name": self.resource_profile_name,
            "resource_domain": self.resource_domain,
            "reserved_bytes": self.reserved_bytes,
            "base_artifact_sha256": self.base_artifact_sha256,
            "base_architecture_id": self.base_architecture_id,
            "base_config_sha256": self.base_config_sha256,
            "base_tokenizer_sha256": self.base_tokenizer_sha256,
        }

    @classmethod
    def from_payload(cls, payload: object) -> ModelArtifactManifest:
        if not isinstance(payload, dict) or set(payload) != {
            "artifact_id", "version", "kind", "role", "relative_path", "artifact_sha256",
            "artifact_size_bytes", "capability_ids", "backend_id", "architecture_id",
            "config_sha256", "tokenizer_sha256", "license_reference", "provenance_reference",
            "training_lineage", "evaluation_reference", "approval_reference", "resource_profile_name",
            "resource_domain", "reserved_bytes", "base_artifact_sha256", "base_architecture_id",
            "base_config_sha256", "base_tokenizer_sha256",
        }:
            raise ArtifactIntegrityError("Stored artifact manifest has an unexpected shape")
        try:
            capability_ids = payload["capability_ids"]
            if not isinstance(capability_ids, list):
                raise ValueError("capability_ids must be an array")
            return cls(
                artifact_id=payload["artifact_id"],
                version=payload["version"],
                kind=ArtifactKind(payload["kind"]),
                role=ModelRole(payload["role"]),
                relative_path=payload["relative_path"],
                artifact_sha256=payload["artifact_sha256"],
                artifact_size_bytes=payload["artifact_size_bytes"],
                capability_ids=tuple(capability_ids),
                backend_id=payload["backend_id"],
                architecture_id=payload["architecture_id"],
                config_sha256=payload["config_sha256"],
                tokenizer_sha256=payload["tokenizer_sha256"],
                license_reference=payload["license_reference"],
                provenance_reference=payload["provenance_reference"],
                training_lineage=payload["training_lineage"],
                evaluation_reference=payload["evaluation_reference"],
                approval_reference=payload["approval_reference"],
                resource_profile_name=payload["resource_profile_name"],
                resource_domain=payload["resource_domain"],
                reserved_bytes=payload["reserved_bytes"],
                base_artifact_sha256=payload["base_artifact_sha256"],
                base_architecture_id=payload["base_architecture_id"],
                base_config_sha256=payload["base_config_sha256"],
                base_tokenizer_sha256=payload["base_tokenizer_sha256"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ArtifactIntegrityError("Stored artifact manifest is invalid") from error


def artifact_manifest_message(manifest: ModelArtifactManifest) -> bytes:
    if not isinstance(manifest, ModelArtifactManifest):
        raise TypeError("manifest must be a ModelArtifactManifest")
    return b"STACEY-ARTIFACT-MANIFEST-v1\0" + _canonical_json(manifest.to_payload())


class ArtifactManifestVerifier(Protocol):
    def verify(self, manifest: ModelArtifactManifest, signature: bytes) -> bool: ...

    def verify_message(self, message: bytes, signature: bytes) -> bool: ...


class Ed25519ArtifactManifestVerifier:
    def __init__(self, public_key: bytes) -> None:
        try:
            self._public_key = Ed25519PublicKey.from_public_bytes(public_key)
        except (TypeError, ValueError) as error:
            raise ValueError("public_key must contain 32 raw Ed25519 bytes") from error

    def verify(self, manifest: ModelArtifactManifest, signature: bytes) -> bool:
        return self.verify_message(artifact_manifest_message(manifest), signature)

    def verify_message(self, message: bytes, signature: bytes) -> bool:
        if not isinstance(message, bytes) or not isinstance(signature, bytes) or len(signature) != 64:
            return False
        try:
            self._public_key.verify(signature, message)
        except InvalidSignature:
            return False
        return True


@dataclass(frozen=True, slots=True)
class CapabilityArtifactBinding:
    capability_id: str
    artifact_id: str
    contract_version: str
    contract_sha256: str

    def __post_init__(self) -> None:
        for field_name in ("capability_id", "artifact_id", "contract_version"):
            _required_text(getattr(self, field_name), field_name)
        _digest(self.contract_sha256, "contract_sha256")

    def to_payload(self) -> dict[str, str]:
        return {
            "capability_id": self.capability_id,
            "artifact_id": self.artifact_id,
            "contract_version": self.contract_version,
            "contract_sha256": self.contract_sha256,
        }


@dataclass(frozen=True, slots=True)
class CollectiveReleaseManifest:
    release_id: str
    version: str
    core_artifact_id: str
    required_capability_ids: tuple[str, ...]
    capability_bindings: tuple[CapabilityArtifactBinding, ...]
    approval_reference: str
    rollback_release_id: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("release_id", "version", "core_artifact_id", "approval_reference"):
            _required_text(getattr(self, field_name), field_name)
        if self.rollback_release_id is not None:
            _required_text(self.rollback_release_id, "rollback_release_id")
            if self.rollback_release_id == self.release_id:
                raise ValueError("A release cannot be its own rollback target")
        if not isinstance(self.required_capability_ids, tuple) or not self.required_capability_ids:
            raise ValueError("required_capability_ids must be a non-empty tuple")
        if any(not isinstance(item, str) or not item.strip() for item in self.required_capability_ids):
            raise ValueError("required_capability_ids must contain non-empty strings")
        if len(set(self.required_capability_ids)) != len(self.required_capability_ids):
            raise ValueError("required_capability_ids must not contain duplicates")
        if not isinstance(self.capability_bindings, tuple) or any(
            not isinstance(binding, CapabilityArtifactBinding) for binding in self.capability_bindings
        ):
            raise ValueError("capability_bindings must contain CapabilityArtifactBinding records")
        binding_ids = tuple(binding.capability_id for binding in self.capability_bindings)
        if len(set(binding_ids)) != len(binding_ids) or set(binding_ids) != set(self.required_capability_ids):
            raise ValueError("Release must bind every required capability exactly once")

    def to_payload(self) -> dict[str, object]:
        return {
            "release_id": self.release_id,
            "version": self.version,
            "core_artifact_id": self.core_artifact_id,
            "required_capability_ids": list(self.required_capability_ids),
            "capability_bindings": [binding.to_payload() for binding in self.capability_bindings],
            "approval_reference": self.approval_reference,
            "rollback_release_id": self.rollback_release_id,
        }

    @classmethod
    def from_payload(cls, payload: object) -> CollectiveReleaseManifest:
        if not isinstance(payload, dict) or set(payload) != {
            "release_id", "version", "core_artifact_id", "required_capability_ids",
            "capability_bindings", "approval_reference", "rollback_release_id",
        }:
            raise ArtifactIntegrityError("Stored collective release has an unexpected shape")
        try:
            required = payload["required_capability_ids"]
            bindings = payload["capability_bindings"]
            if not isinstance(required, list) or not isinstance(bindings, list):
                raise ValueError("Release capability data must be arrays")
            parsed_bindings: list[CapabilityArtifactBinding] = []
            for binding in bindings:
                if not isinstance(binding, dict) or set(binding) != {
                    "capability_id", "artifact_id", "contract_version", "contract_sha256"
                }:
                    raise ValueError("Release capability binding has an unexpected shape")
                parsed_bindings.append(CapabilityArtifactBinding(**binding))
            return cls(
                release_id=payload["release_id"],
                version=payload["version"],
                core_artifact_id=payload["core_artifact_id"],
                required_capability_ids=tuple(required),
                capability_bindings=tuple(parsed_bindings),
                approval_reference=payload["approval_reference"],
                rollback_release_id=payload["rollback_release_id"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ArtifactIntegrityError("Stored collective release is invalid") from error


def collective_release_message(manifest: CollectiveReleaseManifest) -> bytes:
    if not isinstance(manifest, CollectiveReleaseManifest):
        raise TypeError("manifest must be a CollectiveReleaseManifest")
    return b"STACEY-COLLECTIVE-RELEASE-v1\0" + _canonical_json(manifest.to_payload())


def release_activation_message(
    release_id: str,
    prior_release_id: str | None,
    action: str,
) -> bytes:
    _required_text(release_id, "release_id")
    if prior_release_id is not None:
        _required_text(prior_release_id, "prior_release_id")
    if action not in {"ACTIVATE", "ROLLBACK"}:
        raise ValueError("action must be ACTIVATE or ROLLBACK")
    return b"STACEY-RELEASE-ACTIVATION-v1\0" + _canonical_json(
        {"release_id": release_id, "prior_release_id": prior_release_id, "action": action}
    )


class ModelArtifactRegistry:
    def __init__(
        self,
        database_path: str | os.PathLike[str],
        artifact_root: str | os.PathLike[str],
        *,
        manifest_verifier: ArtifactManifestVerifier,
        file_mode: int = 0o600,
    ) -> None:
        database = Path(database_path)
        root = Path(artifact_root)
        if not database.parent.is_dir() or database.is_symlink():
            raise ValueError("database_path must be a non-symlink file in an existing directory")
        if root.is_symlink() or not root.is_dir():
            raise ValueError("artifact_root must be an existing non-symlink directory")
        if not callable(getattr(manifest_verifier, "verify", None)):
            raise ValueError("manifest_verifier must implement verify()")
        if isinstance(file_mode, bool) or not isinstance(file_mode, int) or file_mode < 0 or file_mode & ~0o777:
            raise ValueError("file_mode must contain only permission bits")
        self._root = root.resolve(strict=True)
        self._verifier = manifest_verifier
        self._lock = RLock()
        self._closed = False
        self._create_database(database.resolve(), file_mode)
        self._connection = sqlite3.connect(database.resolve(), timeout=30, isolation_level=None, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._initialize_schema()

    @staticmethod
    def _create_database(path: Path, file_mode: int) -> None:
        flags = os.O_CREAT | os.O_EXCL | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags, file_mode)
        except FileExistsError:
            if path.is_symlink() or not path.is_file():
                raise ValueError("database_path must identify a regular non-symlink file")
        else:
            os.fsync(descriptor)
            os.close(descriptor)
        os.chmod(path, file_mode, follow_symlinks=False)

    def _initialize_schema(self) -> None:
        version = self._connection.execute("PRAGMA user_version").fetchone()[0]
        if version == 0:
            existing = self._connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
            if existing:
                raise ArtifactIntegrityError("Unversioned artifact registry contains unexpected tables")
            self._connection.executescript(
                """
                CREATE TABLE artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    manifest_json TEXT NOT NULL,
                    signature BLOB NOT NULL
                );
                CREATE TRIGGER artifacts_no_update BEFORE UPDATE ON artifacts
                    BEGIN SELECT RAISE(ABORT, 'artifact manifests are immutable'); END;
                CREATE TRIGGER artifacts_no_delete BEFORE DELETE ON artifacts
                    BEGIN SELECT RAISE(ABORT, 'artifact manifests are immutable'); END;
                PRAGMA user_version=1;
                """
            )
            version = 1
        if version == 1:
            self._connection.executescript(
                """
                CREATE TABLE collective_releases (
                    release_id TEXT PRIMARY KEY,
                    manifest_json TEXT NOT NULL,
                    signature BLOB NOT NULL
                );
                CREATE TRIGGER collective_releases_no_update BEFORE UPDATE ON collective_releases
                    BEGIN SELECT RAISE(ABORT, 'collective releases are immutable'); END;
                CREATE TRIGGER collective_releases_no_delete BEFORE DELETE ON collective_releases
                    BEGIN SELECT RAISE(ABORT, 'collective releases are immutable'); END;
                CREATE TABLE release_activations (
                    sequence INTEGER PRIMARY KEY,
                    release_id TEXT NOT NULL,
                    prior_release_id TEXT,
                    action TEXT NOT NULL,
                    occurred_at_ns INTEGER NOT NULL,
                    signature BLOB NOT NULL,
                    previous_hash TEXT,
                    event_hash TEXT NOT NULL UNIQUE
                );
                CREATE TRIGGER release_activations_no_update BEFORE UPDATE ON release_activations
                    BEGIN SELECT RAISE(ABORT, 'release activation history is append-only'); END;
                CREATE TRIGGER release_activations_no_delete BEFORE DELETE ON release_activations
                    BEGIN SELECT RAISE(ABORT, 'release activation history is append-only'); END;
                PRAGMA user_version=2;
                """
            )
            version = 2
        if version != 2:
            raise ArtifactIntegrityError("Unsupported artifact registry schema version")

    def register(self, manifest: ModelArtifactManifest, signature: bytes) -> ModelArtifactManifest:
        if not isinstance(manifest, ModelArtifactManifest):
            raise TypeError("manifest must be a ModelArtifactManifest")
        if not self._verifier.verify(manifest, signature):
            raise ArtifactApprovalError("Artifact manifest lacks a valid operator signature")
        self._verify_artifact_file(manifest)
        if manifest.kind is ArtifactKind.LORA_ADAPTER:
            self._verify_adapter_base(manifest)
        encoded = _canonical_json(manifest.to_payload()).decode("utf-8")
        with self._lock:
            try:
                self._connection.execute(
                    "INSERT INTO artifacts (artifact_id, manifest_json, signature) VALUES (?, ?, ?)",
                    (manifest.artifact_id, encoded, signature),
                )
            except sqlite3.IntegrityError as error:
                raise ArtifactRegistryError("Artifact ID is already registered") from error
            return manifest

    def get(self, artifact_id: str) -> ModelArtifactManifest:
        manifest, _ = self.get_signed(artifact_id)
        return manifest

    def get_signed(self, artifact_id: str) -> tuple[ModelArtifactManifest, bytes]:
        _required_text(artifact_id, "artifact_id")
        with self._lock:
            row = self._connection.execute(
                "SELECT manifest_json, signature FROM artifacts WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchone()
        if row is None:
            raise ArtifactNotFoundError(f"Unknown artifact ID: {artifact_id}")
        manifest = self._decode_manifest(row[0], row[1])
        return manifest, bytes(row[1])

    def get_by_digest(self, artifact_sha256: str) -> ModelArtifactManifest:
        _digest(artifact_sha256, "artifact_sha256")
        with self._lock:
            rows = self._connection.execute("SELECT manifest_json, signature FROM artifacts").fetchall()
        for encoded, signature in rows:
            manifest = self._decode_manifest(encoded, signature)
            if manifest.artifact_sha256 == artifact_sha256:
                return manifest
        raise ArtifactNotFoundError("No approved artifact matches the requested digest")

    def approved_for(self, capability_id: str, *, backend_id: str | None = None) -> tuple[ModelArtifactManifest, ...]:
        _required_text(capability_id, "capability_id")
        with self._lock:
            rows = self._connection.execute("SELECT manifest_json, signature FROM artifacts").fetchall()
        manifests = [self._decode_manifest(encoded, signature) for encoded, signature in rows]
        return tuple(
            sorted(
                (
                    manifest
                    for manifest in manifests
                    if capability_id in manifest.capability_ids
                    and (backend_id is None or manifest.backend_id == backend_id)
                ),
                key=lambda manifest: (manifest.artifact_size_bytes, manifest.artifact_id),
            )
        )

    def register_release(
        self,
        manifest: CollectiveReleaseManifest,
        signature: bytes,
    ) -> CollectiveReleaseManifest:
        if not isinstance(manifest, CollectiveReleaseManifest):
            raise TypeError("manifest must be a CollectiveReleaseManifest")
        if not self._verifier.verify_message(collective_release_message(manifest), signature):
            raise ArtifactApprovalError("Collective release lacks a valid operator signature")
        core = self.get(manifest.core_artifact_id)
        if core.role is not ModelRole.WORLD_MODEL_CORE or core.kind is not ArtifactKind.FULL_MODEL:
            raise ArtifactCompatibilityError("Collective release must bind a full-model Stacey Core")
        if core.training_lineage != "STACEY_SCRATCH":
            raise ArtifactCompatibilityError("Collective release Core is not Stacey scratch-trained")
        for binding in manifest.capability_bindings:
            artifact = self.get(binding.artifact_id)
            if binding.capability_id not in artifact.capability_ids:
                raise ArtifactCompatibilityError("Capability binding is not declared by its artifact")
            if artifact.kind is ArtifactKind.LORA_ADAPTER:
                if artifact.base_artifact_sha256 != core.artifact_sha256:
                    raise ArtifactCompatibilityError("Release adapter is bound to a different Core artifact")
            elif artifact.role is not ModelRole.SPECIALIST:
                raise ArtifactCompatibilityError("Learned capability binding must point to a specialist artifact")
        encoded = _canonical_json(manifest.to_payload()).decode("utf-8")
        with self._lock:
            try:
                self._connection.execute(
                    "INSERT INTO collective_releases (release_id, manifest_json, signature) VALUES (?, ?, ?)",
                    (manifest.release_id, encoded, signature),
                )
            except sqlite3.IntegrityError as error:
                raise ArtifactRegistryError("Collective release ID is already registered") from error
        return manifest

    def get_release(self, release_id: str) -> CollectiveReleaseManifest:
        manifest, _ = self.get_release_signed(release_id)
        return manifest

    def get_release_signed(self, release_id: str) -> tuple[CollectiveReleaseManifest, bytes]:
        _required_text(release_id, "release_id")
        with self._lock:
            row = self._connection.execute(
                "SELECT manifest_json, signature FROM collective_releases WHERE release_id = ?",
                (release_id,),
            ).fetchone()
        if row is None:
            raise ArtifactNotFoundError(f"Unknown collective release ID: {release_id}")
        try:
            payload = json.loads(row[0])
        except (TypeError, json.JSONDecodeError) as error:
            raise ArtifactIntegrityError("Stored collective release is malformed") from error
        manifest = CollectiveReleaseManifest.from_payload(payload)
        if not self._verifier.verify_message(collective_release_message(manifest), row[1]):
            raise ArtifactIntegrityError("Stored collective release signature is invalid")
        return manifest, bytes(row[1])

    def active_release(self) -> CollectiveReleaseManifest | None:
        with self._lock:
            rows = self._connection.execute(
                """SELECT sequence, release_id, prior_release_id, action,
                          occurred_at_ns, signature, previous_hash, event_hash
                   FROM release_activations ORDER BY sequence"""
            ).fetchall()
        previous_hash: str | None = None
        active_release_id: str | None = None
        for expected_sequence, row in enumerate(rows, start=1):
            sequence, release_id, prior_release_id, action, occurred_at_ns, signature, claimed_previous, event_hash = row
            if sequence != expected_sequence or claimed_previous != previous_hash or prior_release_id != active_release_id:
                raise ArtifactIntegrityError("Release activation history is discontinuous")
            message = release_activation_message(release_id, prior_release_id, action)
            if not self._verifier.verify_message(message, signature):
                raise ArtifactIntegrityError("Release activation signature is invalid")
            body = {
                "sequence": sequence,
                "release_id": release_id,
                "prior_release_id": prior_release_id,
                "action": action,
                "occurred_at_ns": occurred_at_ns,
                "previous_hash": claimed_previous,
            }
            actual_hash = hashlib.sha256(_canonical_json(body)).hexdigest()
            if actual_hash != event_hash:
                raise ArtifactIntegrityError("Release activation hash verification failed")
            self.get_release(release_id)
            previous_hash = event_hash
            active_release_id = release_id
        return None if active_release_id is None else self.get_release(active_release_id)

    def activate_release(
        self,
        release_id: str,
        signature: bytes,
        *,
        rollback: bool = False,
    ) -> CollectiveReleaseManifest:
        manifest = self.get_release(release_id)
        active = self.active_release()
        prior_release_id = None if active is None else active.release_id
        if not isinstance(rollback, bool):
            raise ValueError("rollback must be a boolean")
        if rollback:
            if active is None or active.rollback_release_id != release_id:
                raise ArtifactCompatibilityError("Requested release is not the active release's rollback target")
            action = "ROLLBACK"
        else:
            if manifest.rollback_release_id != prior_release_id:
                raise ArtifactCompatibilityError("New release must bind the currently active rollback target")
            action = "ACTIVATE"
        message = release_activation_message(release_id, prior_release_id, action)
        if not self._verifier.verify_message(message, signature):
            raise ArtifactApprovalError("Release activation lacks a valid operator signature")
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                latest = self._connection.execute(
                    "SELECT release_id FROM release_activations ORDER BY sequence DESC LIMIT 1"
                ).fetchone()
                current_id = None if latest is None else latest[0]
                if current_id != prior_release_id:
                    raise ArtifactCompatibilityError("Active release changed during activation")
                sequence_row = self._connection.execute(
                    "SELECT sequence, event_hash FROM release_activations ORDER BY sequence DESC LIMIT 1"
                ).fetchone()
                sequence = 1 if sequence_row is None else sequence_row[0] + 1
                previous_hash = None if sequence_row is None else sequence_row[1]
                occurred_at_ns = time_ns()
                body = {
                    "sequence": sequence,
                    "release_id": release_id,
                    "prior_release_id": prior_release_id,
                    "action": action,
                    "occurred_at_ns": occurred_at_ns,
                    "previous_hash": previous_hash,
                }
                event_hash = hashlib.sha256(_canonical_json(body)).hexdigest()
                self._connection.execute(
                    """INSERT INTO release_activations
                       (sequence, release_id, prior_release_id, action, occurred_at_ns,
                        signature, previous_hash, event_hash)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (sequence, release_id, prior_release_id, action, occurred_at_ns, signature, previous_hash, event_hash),
                )
                self._connection.execute("COMMIT")
            except Exception:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
        return manifest

    @contextmanager
    def open_verified(self, artifact_id: str) -> Iterator[tuple[ModelArtifactManifest, BinaryIO]]:
        manifest = self.get(artifact_id)
        descriptor = self._open_artifact_descriptor(manifest.relative_path)
        with os.fdopen(descriptor, "rb") as stream:
            digest = hashlib.sha256()
            size = 0
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
            if digest.hexdigest() != manifest.artifact_sha256 or size != manifest.artifact_size_bytes:
                raise ArtifactIntegrityError("Artifact bytes no longer match the signed manifest")
            stream.seek(0)
            yield manifest, stream

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    def _decode_manifest(self, encoded: str, signature: bytes) -> ModelArtifactManifest:
        try:
            payload = json.loads(encoded)
        except (TypeError, json.JSONDecodeError) as error:
            raise ArtifactIntegrityError("Stored artifact manifest is malformed") from error
        manifest = ModelArtifactManifest.from_payload(payload)
        if not self._verifier.verify(manifest, signature):
            raise ArtifactIntegrityError("Stored artifact signature is invalid")
        return manifest

    def _verify_adapter_base(self, adapter: ModelArtifactManifest) -> None:
        base = self.get_by_digest(adapter.base_artifact_sha256 or "")
        if (
            base.role is not ModelRole.WORLD_MODEL_CORE
            or base.kind is not ArtifactKind.FULL_MODEL
            or base.architecture_id != adapter.base_architecture_id
            or base.config_sha256 != adapter.base_config_sha256
            or base.tokenizer_sha256 != adapter.base_tokenizer_sha256
        ):
            raise ArtifactCompatibilityError("LoRA adapter does not match its exact approved Core base")

    def _verify_artifact_file(self, manifest: ModelArtifactManifest) -> None:
        descriptor = self._open_artifact_descriptor(manifest.relative_path)
        try:
            file_stat = os.fstat(descriptor)
            if not stat.S_ISREG(file_stat.st_mode):
                raise ArtifactIntegrityError("Model artifact must be a regular file")
            digest = hashlib.sha256()
            size = 0
            while chunk := os.read(descriptor, 1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
            if digest.hexdigest() != manifest.artifact_sha256 or size != manifest.artifact_size_bytes:
                raise ArtifactIntegrityError("Artifact bytes do not match the signed manifest")
        finally:
            os.close(descriptor)

    def _open_artifact_descriptor(self, relative_path: str) -> int:
        parts = PurePosixPath(relative_path).parts
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        directory_fd = os.open(self._root, flags | getattr(os, "O_CLOEXEC", 0))
        try:
            for part in parts[:-1]:
                next_fd = os.open(part, flags | getattr(os, "O_CLOEXEC", 0), dir_fd=directory_fd)
                os.close(directory_fd)
                directory_fd = next_fd
            descriptor = os.open(
                parts[-1],
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
                dir_fd=directory_fd,
            )
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                os.close(descriptor)
                raise ArtifactIntegrityError("Model artifact must be a regular file")
            return descriptor
        except OSError as error:
            raise ArtifactIntegrityError("Artifact path is missing or traverses a symlink") from error
        finally:
            os.close(directory_fd)


class ModelRuntimeBackend(Protocol):
    backend_id: str
    max_concurrent_leases: int

    def load(
        self,
        manifest: ModelArtifactManifest,
        artifact: BinaryIO,
        *,
        base_handle: object | None,
    ) -> object: ...

    def infer(self, handle: object, request: object) -> object: ...

    def unload(self, handle: object) -> bool: ...

    def confirm_released(self, handle: object) -> bool: ...


@dataclass(slots=True)
class _ResidentArtifact:
    manifest: ModelArtifactManifest
    backend: ModelRuntimeBackend
    handle: object | None
    resource_lease: ResourceLease
    state: ResidentState
    active_leases: int
    last_used_ns: int


class ModelLease:
    def __init__(
        self,
        manager: ModelLifecycleManager,
        artifact_id: str,
        artifact_sha256: str,
        handle: object,
        *,
        dependencies: tuple[ModelLease, ...] = (),
    ) -> None:
        self._manager = manager
        self.artifact_id = artifact_id
        self.artifact_sha256 = artifact_sha256
        self._handle = handle
        self._dependencies = dependencies
        self._released = False

    def infer(self, request: object) -> object:
        if self._released:
            raise ModelLifecycleError("Model lease has been released")
        return self._manager._infer(self.artifact_id, self._handle, request)

    def release(self) -> None:
        if not self._released:
            try:
                self._manager._release(self.artifact_id)
            finally:
                for dependency in reversed(self._dependencies):
                    dependency.release()
                self._released = True

    def __enter__(self) -> ModelLease:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()


class ModelLifecycleManager:
    def __init__(
        self,
        *,
        registry: ModelArtifactRegistry,
        backends: Mapping[str, ModelRuntimeBackend],
        lease_manager: ResourceLeaseManager,
        clock: Callable[[], int] = time_ns,
    ) -> None:
        if not callable(clock) or not backends:
            raise ValueError("A clock and at least one runtime backend are required")
        for key, backend in backends.items():
            if key != getattr(backend, "backend_id", None):
                raise ValueError("Backend mapping keys must match each backend's backend_id")
            concurrency_limit = getattr(backend, "max_concurrent_leases", None)
            if isinstance(concurrency_limit, bool) or not isinstance(concurrency_limit, int) or concurrency_limit <= 0:
                raise ValueError("Every runtime backend must declare a positive max_concurrent_leases")
        self._registry = registry
        self._backends = dict(backends)
        self._lease_manager = lease_manager
        self._clock = clock
        self._lock = RLock()
        self._resident: dict[str, _ResidentArtifact] = {}

    def acquire(
        self,
        capability_id: str,
        *,
        base_artifact_sha256: str | None = None,
    ) -> ModelLease:
        candidates = self._registry.approved_for(capability_id)
        if base_artifact_sha256 is not None:
            _digest(base_artifact_sha256, "base_artifact_sha256")
            candidates = tuple(
                manifest
                for manifest in candidates
                if manifest.kind is ArtifactKind.FULL_MODEL
                or manifest.base_artifact_sha256 == base_artifact_sha256
            )
        errors: list[Exception] = []
        for manifest in candidates:
            try:
                return self._acquire_approved_manifest(manifest)
            except (ResourceAdmissionDenied, ModelLoadError, ModelQuarantinedError) as error:
                errors.append(error)
        if not candidates:
            raise ArtifactNotFoundError("No approved compatible artifact is registered for this capability")
        raise ModelLifecycleError("No compatible artifact could be admitted") from (errors[-1] if errors else None)

    def acquire_from_active_release(
        self,
        capability_id: str,
        *,
        contract_version: str,
        contract_sha256: str,
        resource_domain: str,
        estimated_required_bytes: int,
        release_id: str | None = None,
    ) -> ModelLease:
        _required_text(capability_id, "capability_id")
        _required_text(contract_version, "contract_version")
        _digest(contract_sha256, "contract_sha256")
        _required_text(resource_domain, "resource_domain")
        if release_id is not None:
            _required_text(release_id, "release_id")
        if (
            isinstance(estimated_required_bytes, bool)
            or not isinstance(estimated_required_bytes, int)
            or estimated_required_bytes < 0
        ):
            raise ValueError("estimated_required_bytes must be a non-negative integer")
        release = self._registry.active_release() if release_id is None else self._registry.get_release(release_id)
        if release is None:
            raise ArtifactNotFoundError("No signed collective release is active")
        binding = next(
            (item for item in release.capability_bindings if item.capability_id == capability_id),
            None,
        )
        if binding is None:
            raise ArtifactNotFoundError("Capability is not present in the active collective release")
        if binding.contract_version != contract_version or binding.contract_sha256 != contract_sha256:
            raise ArtifactCompatibilityError("Local capability contract differs from the signed active release")
        artifact = self._registry.get(binding.artifact_id)
        if capability_id not in artifact.capability_ids:
            raise ArtifactCompatibilityError("Active release capability binding no longer matches its artifact")
        if resource_domain != artifact.resource_domain or estimated_required_bytes > artifact.reserved_bytes:
            raise ArtifactCompatibilityError("Task resource estimate exceeds the active artifact resource profile")
        return self._acquire_approved_manifest(artifact)

    def active_release(self) -> CollectiveReleaseManifest | None:
        return self._registry.active_release()

    def core_artifact_sha256_for_release(self, release_id: str) -> str:
        release = self._registry.get_release(release_id)
        core = self._registry.get(release.core_artifact_id)
        if core.role is not ModelRole.WORLD_MODEL_CORE or core.training_lineage != "STACEY_SCRATCH":
            raise ArtifactCompatibilityError("Pinned release Core is not a Stacey scratch-trained artifact")
        return core.artifact_sha256

    def acquire_core_from_release(self, release_id: str) -> ModelLease:
        release = self._registry.get_release(release_id)
        core = self._registry.get(release.core_artifact_id)
        if (
            core.kind is not ArtifactKind.FULL_MODEL
            or core.role is not ModelRole.WORLD_MODEL_CORE
            or core.training_lineage != "STACEY_SCRATCH"
        ):
            raise ArtifactCompatibilityError("Pinned release Core is not a full scratch-trained Core artifact")
        return self._acquire_approved_manifest(core)

    def _acquire_approved_manifest(self, manifest: ModelArtifactManifest) -> ModelLease:
        if manifest.kind is ArtifactKind.LORA_ADAPTER:
            base = self._registry.get_by_digest(manifest.base_artifact_sha256 or "")
            if base.kind is not ArtifactKind.FULL_MODEL or base.role is not ModelRole.WORLD_MODEL_CORE:
                raise ArtifactCompatibilityError("LoRA base is not an approved full Stacey Core")
            base_lease = self._acquire_manifest(base)
            try:
                adapter_lease = self._acquire_manifest(
                    manifest,
                    base_handle=base_lease._handle,
                )
            except Exception:
                base_lease.release()
                raise
            return ModelLease(
                self,
                manifest.artifact_id,
                manifest.artifact_sha256,
                adapter_lease._handle,
                dependencies=(base_lease,),
            )
        return self._acquire_manifest(manifest)

    def evict(self, artifact_id: str) -> bool:
        with self._lock:
            resident = self._resident.get(artifact_id)
            if resident is None:
                return True
            if resident.active_leases:
                raise ModelLifecycleError("Cannot evict an artifact with active leases")
            if resident.manifest.role is ModelRole.WORLD_MODEL_CORE:
                raise ModelLifecycleError("Persistent Core artifacts require explicit shutdown policy")
            return self._unload_resident(artifact_id, resident)

    def reconcile_quarantine(self, artifact_id: str) -> bool:
        with self._lock:
            resident = self._resident.get(artifact_id)
            if resident is None:
                return True
            if resident.state is not ResidentState.QUARANTINED:
                return False
            try:
                released = resident.backend.confirm_released(resident.handle)
            except Exception:
                return False
            if released is not True:
                return False
            resident.resource_lease.release()
            del self._resident[artifact_id]
            return True

    @property
    def quarantined_artifact_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(
                artifact_id
                for artifact_id, resident in self._resident.items()
                if resident.state is ResidentState.QUARANTINED
            ))

    @property
    def resident_artifact_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._resident))

    def _acquire_manifest(
        self,
        manifest: ModelArtifactManifest,
        *,
        base_handle: object | None = None,
    ) -> ModelLease:
        with self._lock:
            resident = self._resident.get(manifest.artifact_id)
            if resident is not None:
                if resident.state is ResidentState.QUARANTINED:
                    raise ModelQuarantinedError("Artifact remains quarantined pending confirmed release")
                if resident.handle is None:
                    raise ModelQuarantinedError("Resident artifact has no verified runtime handle")
                if resident.active_leases >= resident.backend.max_concurrent_leases:
                    raise ModelLifecycleError("Backend concurrent lease limit is reached")
                if manifest.kind is ArtifactKind.LORA_ADAPTER and base_handle is None:
                    raise ArtifactCompatibilityError("LoRA adapter activation requires its leased exact base")
                resident.active_leases += 1
                resident.last_used_ns = self._clock()
                return ModelLease(self, manifest.artifact_id, manifest.artifact_sha256, resident.handle)

            backend = self._backends.get(manifest.backend_id)
            if backend is None:
                raise ModelLoadError("No runtime backend is registered for the approved artifact")
            resource_lease = self._reserve(manifest)
            try:
                with self._registry.open_verified(manifest.artifact_id) as (_, artifact):
                    handle = backend.load(manifest, artifact, base_handle=base_handle)
                if handle is None:
                    raise ModelLoadError("Runtime backend returned an empty handle")
            except Exception as error:
                if getattr(error, "resources_released", False) is True:
                    resource_lease.release()
                    raise ModelLoadError("Runtime backend failed with confirmed resource release") from error
                self._resident[manifest.artifact_id] = _ResidentArtifact(
                    manifest, backend, getattr(error, "handle", None), resource_lease,
                    ResidentState.QUARANTINED, 0, self._clock(),
                )
                raise ModelQuarantinedError("Load failure left resource release uncertain") from error
            resident = _ResidentArtifact(
                manifest,
                backend,
                handle,
                resource_lease,
                ResidentState.RESIDENT,
                1,
                self._clock(),
            )
            self._resident[manifest.artifact_id] = resident
            return ModelLease(self, manifest.artifact_id, manifest.artifact_sha256, handle)

    def _reserve(self, manifest: ModelArtifactManifest) -> ResourceLease:
        profile = ResourceProfile(
            manifest.resource_profile_name,
            manifest.resource_domain,
            manifest.reserved_bytes,
        )
        while True:
            try:
                return self._lease_manager.acquire(profile)
            except ResourceAdmissionDenied:
                if not self._evict_one_idle():
                    raise

    def _evict_one_idle(self) -> bool:
        candidates = sorted(
            (
                (artifact_id, resident)
                for artifact_id, resident in self._resident.items()
                if resident.state is ResidentState.RESIDENT
                and resident.active_leases == 0
                and resident.manifest.role is ModelRole.SPECIALIST
            ),
            key=lambda item: (item[1].last_used_ns, item[0]),
        )
        for artifact_id, resident in candidates:
            if self._unload_resident(artifact_id, resident):
                return True
        return False

    def _unload_resident(self, artifact_id: str, resident: _ResidentArtifact) -> bool:
        if resident.active_leases:
            return False
        try:
            released = resident.backend.unload(resident.handle)
            if released is True:
                released = resident.backend.confirm_released(resident.handle)
        except Exception:
            released = False
        if released is not True:
            resident.state = ResidentState.QUARANTINED
            return False
        resident.resource_lease.release()
        self._resident.pop(artifact_id, None)
        return True

    def _infer(self, artifact_id: str, handle: object, request: object) -> object:
        with self._lock:
            resident = self._resident.get(artifact_id)
            if resident is None or resident.state is not ResidentState.RESIDENT or resident.handle is not handle:
                raise ModelLifecycleError("Artifact handle is no longer active")
            backend = resident.backend
        return backend.infer(handle, request)

    def _release(self, artifact_id: str) -> None:
        with self._lock:
            resident = self._resident.get(artifact_id)
            if resident is None or resident.active_leases <= 0:
                raise ModelLifecycleError("Model lease accounting is inconsistent")
            resident.active_leases -= 1
            resident.last_used_ns = self._clock()