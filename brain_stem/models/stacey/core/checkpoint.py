from __future__ import annotations

import hashlib
import os
import re
import stat
import tempfile
from pathlib import Path

import torch

from .config import StaceyCoreConfig
from .network import StaceyCore


class StaceyCheckpointError(RuntimeError):
    pass


def _validate_file_mode(file_mode: int) -> None:
    if isinstance(file_mode, bool) or not isinstance(file_mode, int):
        raise StaceyCheckpointError("file_mode must be an explicitly injected integer")
    allowed = stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO
    if file_mode < 0 or file_mode & ~allowed:
        raise StaceyCheckpointError("file_mode contains unsupported permission bits")


def save_model_checkpoint(
    model: StaceyCore,
    destination: str | os.PathLike[str],
    *,
    file_mode: int,
) -> str:
    if not isinstance(model, StaceyCore):
        raise StaceyCheckpointError("model must be a StaceyCore")
    _validate_file_mode(file_mode)
    path = Path(destination)
    if path.is_symlink() or not path.name or not path.parent.is_dir():
        raise StaceyCheckpointError("checkpoint target must be a non-symlink file in an existing directory")

    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".pending", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        os.chmod(temporary_path, file_mode)
        torch.save(
            {
                "format_id": "stacey.core.checkpoint.v0",
                "config": model.config.to_dict(),
                "state_dict": model.state_dict(),
            },
            temporary_path,
        )
        with temporary_path.open("rb") as checkpoint_file:
            os.fsync(checkpoint_file.fileno())
        os.replace(temporary_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model_checkpoint(
    source: str | os.PathLike[str],
    *,
    expected_sha256: str,
    map_location: str | torch.device,
) -> StaceyCore:
    if not isinstance(expected_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise StaceyCheckpointError("expected_sha256 must be a lowercase SHA-256 digest")
    path = Path(source)
    if path.is_symlink() or not path.is_file():
        raise StaceyCheckpointError("checkpoint source must be a regular non-symlink file")
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise StaceyCheckpointError("checkpoint digest does not match the expected artifact digest")
    try:
        bundle = torch.load(path, map_location=map_location, weights_only=True)
    except Exception as error:
        raise StaceyCheckpointError("checkpoint could not be safely loaded") from error
    if not isinstance(bundle, dict) or set(bundle) != {"format_id", "config", "state_dict"}:
        raise StaceyCheckpointError("checkpoint bundle has an invalid shape")
    if bundle["format_id"] != "stacey.core.checkpoint.v0":
        raise StaceyCheckpointError("unsupported Stacey checkpoint format")
    config = StaceyCoreConfig.from_dict(bundle["config"])
    model = StaceyCore(config)
    try:
        model.load_state_dict(bundle["state_dict"], strict=True)
    except Exception as error:
        raise StaceyCheckpointError("checkpoint weights do not match the declared architecture") from error
    return model