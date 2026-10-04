from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from .provenance_graph import ProvenanceGraphLinker, ProvenanceGraphRecord
from .transaction_coordinator import TransactionResult
from .workspace_writer import WorkspaceWriteCommand, normalize_workspace_relative_path
from .world_state import WorldStateLedger, WorldStateRecord, WorldStateAssertion


class WorkspaceProjectionError(RuntimeError):
    def __init__(self, message: str, *, assertion_id: str | None = None) -> None:
        super().__init__(message)
        self.assertion_id = assertion_id


@dataclass(frozen=True, slots=True)
class WorkspaceProjectionResult:
    assertion: WorldStateRecord
    provenance: ProvenanceGraphRecord


class CommittedWorkspaceProjector:
    """Projects only durable, successful, artifact-bound write receipts into FB-004."""

    def __init__(
        self,
        *,
        world_state_ledger: WorldStateLedger,
        provenance_graph: ProvenanceGraphLinker,
    ) -> None:
        if not isinstance(world_state_ledger, WorldStateLedger):
            raise ValueError("world_state_ledger must be a WorldStateLedger")
        if not isinstance(provenance_graph, ProvenanceGraphLinker):
            raise ValueError("provenance_graph must be a ProvenanceGraphLinker")
        self._world_state_ledger = world_state_ledger
        self._provenance_graph = provenance_graph

    def project(
        self,
        command: WorkspaceWriteCommand,
        result: TransactionResult,
    ) -> WorkspaceProjectionResult:
        if not isinstance(command, WorkspaceWriteCommand) or not isinstance(result, TransactionResult):
            raise TypeError("project requires a WorkspaceWriteCommand and TransactionResult")
        if not result.succeeded or result.transaction_id != command.transaction_id:
            raise WorkspaceProjectionError("only the matching successful transaction receipt can be projected")
        if command.proposal_release_id is None or command.proposal_artifact_sha256 is None:
            raise WorkspaceProjectionError("workspace projection requires a release-bound artifact identity")
        match = re.fullmatch(r"sha256:([0-9a-f]{64})", result.verification_ref)
        if match is None:
            raise WorkspaceProjectionError("transaction receipt does not contain a verified content digest")
        content_sha256 = match.group(1)
        relative_path = "/".join(normalize_workspace_relative_path(command.relative_path))
        assertion_material = json.dumps(
            {
                "transaction_id": command.transaction_id,
                "scope": {
                    "tenant_id": command.scope.tenant_id,
                    "user_id": command.scope.user_id,
                    "project_id": command.scope.project_id,
                    "workspace_id": command.scope.workspace_id,
                },
                "relative_path": relative_path,
                "content_sha256": content_sha256,
                "release_id": command.proposal_release_id,
                "artifact_sha256": command.proposal_artifact_sha256,
            },
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        assertion_id = f"workspace-write:{hashlib.sha256(assertion_material).hexdigest()}"
        assertion = WorldStateAssertion(
            assertion_id=assertion_id,
            fact_key=f"workspace-file:{relative_path}",
            scope=command.scope,
            subject=f"workspace:{command.scope.workspace_id}",
            predicate="file-content-sha256",
            value={
                "relative_path": relative_path,
                "content_sha256": content_sha256,
                "release_id": command.proposal_release_id,
                "artifact_sha256": command.proposal_artifact_sha256,
            },
            source_ref=f"workspace-file:{relative_path}",
            source_sha256=content_sha256,
            transaction_id=command.transaction_id,
        )
        try:
            record = self._world_state_ledger.append(assertion)
            provenance = self._provenance_graph.link_assertion(
                assertion_id,
                command.scope,
                command.proposal_artifact_sha256,
            )
        except Exception as error:
            raise WorkspaceProjectionError(
                "committed write lacks complete FB-004 projection; retry with the same receipt",
                assertion_id=assertion_id,
            ) from error
        if provenance.ledger_event_hash != record.event_hash:
            raise WorkspaceProjectionError(
                "provenance link points to a different ledger event",
                assertion_id=assertion_id,
            )
        return WorkspaceProjectionResult(record, provenance)