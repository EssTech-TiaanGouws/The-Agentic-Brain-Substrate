from __future__ import annotations

import gc
import hashlib
import json
from dataclasses import dataclass
from typing import BinaryIO

import torch

from models.stacey.core.checkpoint import load_model_checkpoint_stream
from models.stacey.core.config import (
    STACEY_CORE_ARCHITECTURE_ID,
    STACEY_CORE_TOKENIZER_ID,
    StaceyCoreConfig,
)
from models.stacey.core.inference import StaceyCoreInferenceAdapter
from models.stacey.core.inputs import UnifiedContextIngress
from models.stacey.core.network import StaceyCore
from models.stacey.core.tokens import (
    BOS_TOKEN_ID,
    BYTE_TOKEN_OFFSET,
    BYTE_VOCABULARY_SIZE,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
)

from .model_catalog import ModelRole
from .model_lifecycle import ArtifactKind, ModelArtifactManifest


class StaceyCoreBackendError(RuntimeError):
    def __init__(self, message: str, *, resources_released: bool) -> None:
        super().__init__(message)
        self.resources_released = resources_released


@dataclass(slots=True)
class _StaceyCoreHandle:
    adapter: StaceyCoreInferenceAdapter | None
    device: torch.device
    baseline_allocated_bytes: int


def stacey_core_config_sha256(config: StaceyCoreConfig) -> str:
    if not isinstance(config, StaceyCoreConfig):
        raise TypeError("config must be a StaceyCoreConfig")
    encoded = json.dumps(
        config.to_dict(),
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stacey_core_tokenizer_sha256() -> str:
    protocol = {
        "tokenizer_id": STACEY_CORE_TOKENIZER_ID,
        "vocabulary_size": BYTE_VOCABULARY_SIZE,
        "byte_token_offset": BYTE_TOKEN_OFFSET,
        "special_tokens": {
            "pad": PAD_TOKEN_ID,
            "bos": BOS_TOKEN_ID,
            "eos": EOS_TOKEN_ID,
        },
        "encoding": "UTF-8",
    }
    encoded = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class StaceyCoreRuntimeBackend:
    backend_id = "stacey.core.pytorch.v0"
    max_concurrent_leases = 1

    def __init__(self, *, device: str | torch.device = "cpu", minimum_confidence: float = 0.0) -> None:
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("Stacey Core backend supports CPU and CUDA devices only")
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA is unavailable for the configured Stacey Core backend")
            device_index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            if not 0 <= device_index < torch.cuda.device_count():
                raise ValueError("Configured CUDA device does not exist")
            self.device = torch.device("cuda", device_index)
        if (
            isinstance(minimum_confidence, bool)
            or not isinstance(minimum_confidence, (int, float))
            or not 0.0 <= float(minimum_confidence) <= 1.0
        ):
            raise ValueError("minimum_confidence must be in [0, 1]")
        self.minimum_confidence = float(minimum_confidence)

    def load(
        self,
        manifest: ModelArtifactManifest,
        artifact: BinaryIO,
        *,
        base_handle: object | None,
    ) -> object:
        if (
            not isinstance(manifest, ModelArtifactManifest)
            or manifest.kind is not ArtifactKind.FULL_MODEL
            or manifest.role is not ModelRole.WORLD_MODEL_CORE
            or manifest.backend_id != self.backend_id
            or manifest.architecture_id != STACEY_CORE_ARCHITECTURE_ID
            or base_handle is not None
        ):
            raise StaceyCoreBackendError("Artifact is not compatible with the Stacey Core backend", resources_released=True)

        baseline = self._allocated_bytes()
        model: StaceyCore | None = None
        try:
            model = load_model_checkpoint_stream(
                artifact,
                expected_sha256=manifest.artifact_sha256,
                map_location="cpu",
            )
            if (
                stacey_core_config_sha256(model.config) != manifest.config_sha256
                or stacey_core_tokenizer_sha256() != manifest.tokenizer_sha256
            ):
                raise ValueError("Checkpoint config or tokenizer digest differs from the signed manifest")
            model.to(self.device)
            model.eval()
            adapter = StaceyCoreInferenceAdapter(
                model,
                candidate_id=manifest.artifact_id,
                artifact_sha256=manifest.artifact_sha256,
                minimum_confidence=self.minimum_confidence,
            )
            return _StaceyCoreHandle(adapter, self.device, baseline)
        except Exception as error:
            try:
                if model is not None:
                    model.to("cpu")
                model = None
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                released = self._allocated_bytes() <= baseline
            except Exception:
                released = False
            raise StaceyCoreBackendError(
                "Signed Stacey Core checkpoint could not be loaded",
                resources_released=released,
            ) from error

    def infer(self, handle: object, request: object) -> object:
        if not isinstance(handle, _StaceyCoreHandle) or handle.adapter is None:
            raise StaceyCoreBackendError("Stacey Core handle is unloaded", resources_released=True)
        if not isinstance(request, UnifiedContextIngress):
            raise TypeError("Stacey Core inference requires a UnifiedContextIngress")
        return handle.adapter.decide(request)

    def unload(self, handle: object) -> bool:
        if not isinstance(handle, _StaceyCoreHandle):
            return False
        adapter = handle.adapter
        if adapter is None:
            return True
        try:
            adapter.model.to("cpu")
            handle.adapter = None
            del adapter
            gc.collect()
            if handle.device.type == "cuda":
                torch.cuda.empty_cache()
            return True
        except Exception:
            return False

    def confirm_released(self, handle: object) -> bool:
        if not isinstance(handle, _StaceyCoreHandle) or handle.adapter is not None:
            return False
        try:
            return self._allocated_bytes(handle.device) <= handle.baseline_allocated_bytes
        except Exception:
            return False

    def _allocated_bytes(self, device: torch.device | None = None) -> int:
        selected = self.device if device is None else device
        if selected.type != "cuda":
            return 0
        return int(torch.cuda.memory_allocated(selected))