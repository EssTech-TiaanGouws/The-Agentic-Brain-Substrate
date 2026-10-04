from __future__ import annotations

import hashlib
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any

import torch


class ResumeCheckpointError(RuntimeError):
    pass


def save_resume_checkpoint(
    payload: dict[str, object],
    destination: str | os.PathLike[str],
    *,
    file_mode: int,
) -> str:
    if not isinstance(payload, dict) or payload.get("format_id") != "stacey.training.resume.v1":
        raise ResumeCheckpointError("payload must be a versioned Stacey training resume state")
    if isinstance(file_mode, bool) or not isinstance(file_mode, int):
        raise ResumeCheckpointError("file_mode must be an explicitly injected integer")
    allowed = stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO
    if file_mode < 0 or file_mode & ~allowed:
        raise ResumeCheckpointError("file_mode contains unsupported permission bits")
    path = Path(destination)
    if path.is_symlink() or not path.name or not path.parent.is_dir():
        raise ResumeCheckpointError("resume checkpoint target must be a non-symlink file in an existing directory")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".pending", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        os.chmod(temporary_path, file_mode)
        torch.save(payload, temporary_path)
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


def load_resume_checkpoint(
    source: str | os.PathLike[str],
    *,
    expected_sha256: str,
    map_location: str | torch.device,
) -> dict[str, Any]:
    if not isinstance(expected_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ResumeCheckpointError("expected_sha256 must be a lowercase SHA-256 digest")
    path = Path(source)
    if path.is_symlink() or not path.is_file():
        raise ResumeCheckpointError("resume checkpoint must be a regular non-symlink file")
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        while chunk := checkpoint_file.read(1024 * 1024):
            digest.update(chunk)
        checkpoint_file.seek(0)
        if digest.hexdigest() != expected_sha256:
            raise ResumeCheckpointError("resume checkpoint digest does not match the supplied reference")
        try:
            payload = torch.load(checkpoint_file, map_location=map_location, weights_only=True)
        except Exception as error:
            raise ResumeCheckpointError("resume checkpoint could not be safely loaded") from error
    if not isinstance(payload, dict) or payload.get("format_id") != "stacey.training.resume.v1":
        raise ResumeCheckpointError("resume checkpoint has an invalid format")
    return payload