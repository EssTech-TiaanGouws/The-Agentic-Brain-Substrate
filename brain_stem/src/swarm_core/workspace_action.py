from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from typing import Protocol

from .collective_scheduler import CapabilityOutput, CollectiveExecutionReport, StepState
from .identity import SignedScopeGrant
from .lease_manager import ResourceProfile
from .transaction_coordinator import TransactionResult
from .workspace_writer import WorkspacePathRejected, WorkspaceWriteCommand, WorkspaceWriteError


class WorkspaceActionProposalError(ValueError):
    pass


class WorkspaceWriter(Protocol):
    def execute(
        self,
        command: WorkspaceWriteCommand,
        authorization: SignedScopeGrant,
    ) -> TransactionResult: ...


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise WorkspaceActionProposalError("workspace proposal contains a duplicate JSON key")
        result[key] = value
    return result


class ReviewedWorkspaceProposalExecutor:
    """Adapts one fully reviewed typed proposal into an independently authorized write."""

    def __init__(
        self,
        *,
        writer: WorkspaceWriter,
        resource_profile: ResourceProfile,
        proposal_capability_id: str = "workspace.write.proposal",
        allowed_format_keys: tuple[str, ...] = ("text/plain",),
        maximum_payload_bytes: int = 64 * 1024,
        maximum_content_bytes: int = 32 * 1024,
    ) -> None:
        if not callable(getattr(writer, "execute", None)):
            raise ValueError("writer must implement execute()")
        if not isinstance(resource_profile, ResourceProfile):
            raise ValueError("resource_profile must be a ResourceProfile")
        if not isinstance(proposal_capability_id, str) or not proposal_capability_id.strip():
            raise ValueError("proposal_capability_id must be a non-empty string")
        if not isinstance(allowed_format_keys, tuple) or not allowed_format_keys or any(
            not isinstance(key, str) or not key.strip() for key in allowed_format_keys
        ):
            raise ValueError("allowed_format_keys must be a non-empty tuple of strings")
        if len(set(allowed_format_keys)) != len(allowed_format_keys):
            raise ValueError("allowed_format_keys must not contain duplicates")
        for name, value in (
            ("maximum_payload_bytes", maximum_payload_bytes),
            ("maximum_content_bytes", maximum_content_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if maximum_content_bytes > maximum_payload_bytes:
            raise ValueError("maximum_content_bytes cannot exceed maximum_payload_bytes")
        self._writer = writer
        self._resource_profile = resource_profile
        self._proposal_capability_id = proposal_capability_id
        self._allowed_format_keys = frozenset(allowed_format_keys)
        self._maximum_payload_bytes = maximum_payload_bytes
        self._maximum_content_bytes = maximum_content_bytes

    def execute(
        self,
        report: CollectiveExecutionReport,
        authorization: SignedScopeGrant,
    ) -> TransactionResult:
        if not isinstance(report, CollectiveExecutionReport):
            raise TypeError("report must be a CollectiveExecutionReport")
        if not isinstance(authorization, SignedScopeGrant):
            raise WorkspaceActionProposalError("A separate signed operator authorization is required")
        if report.clarification_required:
            raise WorkspaceActionProposalError("Clarification workflows cannot dispatch workspace writes")
        if not report.release_id.strip():
            raise WorkspaceActionProposalError("workspace write report is not bound to a release")

        proposal_outputs = tuple(
            output for output in report.outputs if output.capability_id == self._proposal_capability_id
        )
        if len(proposal_outputs) != 1:
            raise WorkspaceActionProposalError("workflow must contain exactly one workspace proposal")
        proposal = proposal_outputs[0]
        if proposal.state is not StepState.SUCCEEDED or not isinstance(proposal.payload, bytes):
            raise WorkspaceActionProposalError("workspace proposal did not pass execution and response validation")
        if not self._has_review_evidence(proposal):
            raise WorkspaceActionProposalError("workspace proposal lacks Block 6/8 review evidence")
        if (
            not isinstance(proposal.artifact_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", proposal.artifact_sha256) is None
        ):
            raise WorkspaceActionProposalError("workspace proposal lacks a signed artifact identity")
        if len(proposal.payload) > self._maximum_payload_bytes:
            raise WorkspaceActionProposalError("workspace proposal exceeds the configured payload limit")

        payload = self._parse_payload(proposal.payload)
        format_key = payload["format_key"]
        if not isinstance(format_key, str) or format_key not in self._allowed_format_keys:
            raise WorkspaceActionProposalError("workspace proposal format is not allowlisted")
        content_encoded = payload["content_base64"]
        if not isinstance(content_encoded, str):
            raise WorkspaceActionProposalError("content_base64 must be a string")
        try:
            content = base64.b64decode(content_encoded, validate=True)
        except (binascii.Error, ValueError) as error:
            raise WorkspaceActionProposalError("content_base64 is malformed") from error
        if not content or len(content) > self._maximum_content_bytes:
            raise WorkspaceActionProposalError("proposed content is empty or exceeds its configured limit")

        expected_digest = payload["expected_current_sha256"]
        if expected_digest is not None and (
            not isinstance(expected_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None
        ):
            raise WorkspaceActionProposalError("expected_current_sha256 must be null or a lowercase SHA-256 digest")
        relative_path = payload["relative_path"]
        if not isinstance(relative_path, str) or not isinstance(format_key, str):
            raise WorkspaceActionProposalError("relative_path and format_key must be strings")

        idempotency_material = json.dumps(
            {
                "transaction_id": report.transaction_id,
                "correlation_id": report.correlation_id,
                "relative_path": relative_path,
                "format_key": format_key,
                "content_sha256": hashlib.sha256(content).hexdigest(),
                "expected_current_sha256": expected_digest,
                "proposal_artifact_sha256": proposal.artifact_sha256,
                "proposal_release_id": report.release_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        try:
            command = WorkspaceWriteCommand(
                transaction_id=report.transaction_id,
                correlation_id=report.correlation_id,
                idempotency_key=f"workspace-proposal:{hashlib.sha256(idempotency_material).hexdigest()}",
                scope=report.scope,
                relative_path=relative_path,
                format_key=format_key,
                content=content,
                expected_current_sha256=expected_digest,
                resource_profile=self._resource_profile,
                proposal_release_id=report.release_id,
                proposal_artifact_sha256=proposal.artifact_sha256,
            )
        except (ValueError, WorkspaceWriteError) as error:
            raise WorkspaceActionProposalError("workspace proposal command is invalid") from error
        return self._writer.execute(command, authorization)

    @staticmethod
    def _has_review_evidence(proposal: CapabilityOutput) -> bool:
        return all(
            isinstance(reference, str) and bool(reference.strip())
            for reference in (proposal.critic_reference, proposal.auditor_reference)
        )

    @staticmethod
    def _parse_payload(payload: bytes) -> dict[str, object]:
        try:
            decoded = payload.decode("utf-8", errors="strict")
            value = json.loads(
                decoded,
                object_pairs_hook=_strict_object,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    WorkspaceActionProposalError(f"invalid JSON constant: {token}")
                ),
            )
        except WorkspaceActionProposalError:
            raise
        except (UnicodeError, json.JSONDecodeError, ValueError) as error:
            raise WorkspaceActionProposalError("workspace proposal is not strict UTF-8 JSON") from error
        if not isinstance(value, dict) or set(value) != {
            "relative_path",
            "format_key",
            "content_base64",
            "expected_current_sha256",
        }:
            raise WorkspaceActionProposalError("workspace proposal has missing or unknown fields")
        return value