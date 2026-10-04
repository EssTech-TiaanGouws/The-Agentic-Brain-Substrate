from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from models.stacey.core.inputs import UnifiedContextIngress
from models.stacey.core.outputs import CoreDecisionEnvelope, StaceyOutputError, parse_decision_jsonl
from src.swarm_core.model_catalog import TrainingDataAuthorization, TrainingSource
from src.swarm_core.training_governance import (
    GovernanceApprovalError,
    GovernanceApprovalVerifier,
    SignedGovernanceApproval,
)
from src.swarm_core.training_readiness import TrainingDatasetManifest


class StaceyCorpusError(ValueError):
    pass


_TRAINING_SPLITS = ("training", "validation")


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StaceyCorpusError(f"{field} must be a non-empty string")
    return value


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise StaceyCorpusError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json(text: str, description: str) -> object:
    try:
        return json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                StaceyCorpusError(f"invalid JSON numeric constant in {description}: {token}")
            ),
        )
    except StaceyCorpusError:
        raise
    except (json.JSONDecodeError, UnicodeError) as error:
        raise StaceyCorpusError(f"{description} is not valid UTF-8 JSON") from error


def dataset_content_sha256(
    split_contents: dict[str, bytes],
    *,
    protected_holdout_sha256: str,
) -> str:
    if not isinstance(split_contents, dict) or set(split_contents) != set(_TRAINING_SPLITS):
        raise StaceyCorpusError("split_contents must contain training and validation bytes only")
    if not isinstance(protected_holdout_sha256, str) or re.fullmatch(
        r"[0-9a-f]{64}", protected_holdout_sha256
    ) is None:
        raise StaceyCorpusError("protected_holdout_sha256 must be a lowercase SHA-256 digest")
    digest = hashlib.sha256()
    for split_name in (*_TRAINING_SPLITS, "protected_holdout"):
        content = (
            split_contents[split_name]
            if split_name in split_contents
            else protected_holdout_sha256.encode("ascii")
        )
        if not isinstance(content, bytes):
            raise StaceyCorpusError(f"{split_name} content must be bytes")
        split_bytes = split_name.encode("ascii")
        digest.update(len(split_bytes).to_bytes(4, "big"))
        digest.update(split_bytes)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class StaceyTrainingExample:
    example_id: str
    source_group_id: str
    source_id: str
    authorization_reference: str
    reviewer_reference: str
    ingress: UnifiedContextIngress
    target_decision_jsonl: str
    decision: CoreDecisionEnvelope


@dataclass(frozen=True, slots=True)
class ApprovedStaceyDataset:
    manifest: TrainingDatasetManifest
    manifest_sha256: str
    approval: SignedGovernanceApproval
    training_examples: tuple[StaceyTrainingExample, ...]
    validation_examples: tuple[StaceyTrainingExample, ...]
    holdout_record_ids: tuple[str, ...]
    holdout_source_group_ids: tuple[str, ...]
    holdout_sha256: str
    holdout_evaluation_reference: str

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, TrainingDatasetManifest):
            raise StaceyCorpusError("manifest must be a TrainingDatasetManifest")
        if not isinstance(self.approval, SignedGovernanceApproval):
            raise StaceyCorpusError("dataset must retain its signed approval evidence")
        if not isinstance(self.manifest_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", self.manifest_sha256) is None:
            raise StaceyCorpusError("manifest_sha256 must be a lowercase SHA-256 digest")
        for field_name in ("training_examples", "validation_examples"):
            examples = getattr(self, field_name)
            if not isinstance(examples, tuple) or not examples or any(
                not isinstance(example, StaceyTrainingExample) for example in examples
            ):
                raise StaceyCorpusError(f"{field_name} must contain StaceyTrainingExample values")
        for field_name in ("holdout_record_ids", "holdout_source_group_ids"):
            values = getattr(self, field_name)
            if not isinstance(values, tuple) or not values or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise StaceyCorpusError(f"{field_name} must be a non-empty tuple of strings")
        if not isinstance(self.holdout_sha256, str) or re.fullmatch(
            r"[0-9a-f]{64}", self.holdout_sha256
        ) is None:
            raise StaceyCorpusError("holdout_sha256 must be a lowercase SHA-256 digest")
        _required_text(self.holdout_evaluation_reference, "holdout_evaluation_reference")


def _read_manifest(path: Path) -> tuple[dict[str, object], bytes]:
    if path.is_symlink() or not path.is_file():
        raise StaceyCorpusError("dataset manifest must be a regular non-symlink file")
    manifest_bytes = path.read_bytes()
    manifest_payload = _parse_json(manifest_bytes.decode("utf-8", errors="strict"), "dataset manifest")
    if not isinstance(manifest_payload, dict):
        raise StaceyCorpusError("dataset manifest must be a JSON object")
    return manifest_payload, manifest_bytes


def _source_records(payload: object) -> tuple[TrainingSource, ...]:
    if not isinstance(payload, list) or not payload:
        raise StaceyCorpusError("sources must be a non-empty array")
    sources: list[TrainingSource] = []
    for item in payload:
        if not isinstance(item, dict) or set(item) != {
            "source_id",
            "authorization",
            "authorization_reference",
        }:
            raise StaceyCorpusError("dataset source has missing or unknown fields")
        try:
            authorization = TrainingDataAuthorization(item["authorization"])
            sources.append(
                TrainingSource(
                    source_id=item["source_id"],
                    authorization=authorization,
                    authorization_reference=item["authorization_reference"],
                )
            )
        except (TypeError, ValueError) as error:
            raise StaceyCorpusError("dataset source has invalid authorization metadata") from error
    if len({source.source_id for source in sources}) != len(sources):
        raise StaceyCorpusError("dataset source IDs must be unique")
    return tuple(sources)


def _split_file(root: Path, split_entry: object, split_name: str) -> tuple[bytes, tuple[str, ...]]:
    if not isinstance(split_entry, dict) or set(split_entry) != {"file", "sha256", "record_ids"}:
        raise StaceyCorpusError(f"{split_name} split entry has missing or unknown fields")
    filename = _required_text(split_entry["file"], f"{split_name}.file")
    if Path(filename).name != filename or filename in {".", ".."}:
        raise StaceyCorpusError(f"{split_name} file must be a plain filename within the dataset directory")
    path = root / filename
    if path.is_symlink() or not path.is_file() or path.resolve().parent != root.resolve():
        raise StaceyCorpusError(f"{split_name} file must be a regular file inside the dataset directory")
    content = path.read_bytes()
    digest = split_entry["sha256"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise StaceyCorpusError(f"{split_name}.sha256 must be a lowercase SHA-256 digest")
    if hashlib.sha256(content).hexdigest() != digest:
        raise StaceyCorpusError(f"{split_name} file digest does not match the manifest")
    record_ids = split_entry["record_ids"]
    if not isinstance(record_ids, list) or not record_ids:
        raise StaceyCorpusError(f"{split_name}.record_ids must be a non-empty array")
    checked_ids = tuple(_required_text(record_id, f"{split_name}.record_ids") for record_id in record_ids)
    if len(set(checked_ids)) != len(checked_ids):
        raise StaceyCorpusError(f"{split_name} record IDs must be unique")
    return content, checked_ids


def _parse_examples(
    content: bytes,
    *,
    split_name: str,
    expected_record_ids: tuple[str, ...],
    sources_by_id: dict[str, TrainingSource],
) -> tuple[StaceyTrainingExample, ...]:
    try:
        text = content.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise StaceyCorpusError(f"{split_name} split is not valid UTF-8") from error
    lines = text.splitlines()
    if len(lines) != len(expected_record_ids) or any(not line.strip() for line in lines):
        raise StaceyCorpusError(f"{split_name} line count must match its manifest record IDs")

    examples: list[StaceyTrainingExample] = []
    for line in lines:
        record = _parse_json(line, f"{split_name} example")
        expected_fields = {
            "example_id",
            "source_group_id",
            "source_id",
            "authorization_reference",
            "reviewer_reference",
            "human_reviewed",
            "external_model_generated",
            "ingress",
            "target_decision_jsonl",
        }
        if not isinstance(record, dict) or set(record) != expected_fields:
            raise StaceyCorpusError("training example has missing or unknown fields")
        if record["human_reviewed"] is not True:
            raise StaceyCorpusError("strict scratch training only accepts human-reviewed examples")
        if record["external_model_generated"] is not False:
            raise StaceyCorpusError("strict scratch training rejects external-model-generated examples")

        example_id = _required_text(record["example_id"], "example_id")
        source_id = _required_text(record["source_id"], "source_id")
        authorization_reference = _required_text(record["authorization_reference"], "authorization_reference")
        source = sources_by_id.get(source_id)
        if source is None or source.authorization_reference != authorization_reference:
            raise StaceyCorpusError("example provenance does not match an authorized manifest source")
        try:
            ingress = UnifiedContextIngress.from_payload(record["ingress"])
        except ValueError as error:
            raise StaceyCorpusError(f"example {example_id} has an invalid ingress") from error
        target = record["target_decision_jsonl"]
        if not isinstance(target, str):
            raise StaceyCorpusError("target_decision_jsonl must be a string")
        try:
            decision = parse_decision_jsonl(target, ingress)
        except StaceyOutputError as error:
            raise StaceyCorpusError(f"example {example_id} has an invalid target decision") from error
        examples.append(
            StaceyTrainingExample(
                example_id=example_id,
                source_group_id=_required_text(record["source_group_id"], "source_group_id"),
                source_id=source_id,
                authorization_reference=authorization_reference,
                reviewer_reference=_required_text(record["reviewer_reference"], "reviewer_reference"),
                ingress=ingress,
                target_decision_jsonl=target,
                decision=decision,
            )
        )
    if tuple(example.example_id for example in examples) != expected_record_ids:
        raise StaceyCorpusError(f"{split_name} record order/IDs differ from the manifest")
    return tuple(examples)


def _semantic_example_sha256(example: StaceyTrainingExample) -> str:
    ingress = example.ingress.to_payload()
    ingress["transaction_id"] = "<transaction>"
    ingress["correlation_id"] = "<correlation>"
    ingress["ingress_timestamp_ns"] = 0
    matrix = ingress["hardware_capability_matrix"]
    assert isinstance(matrix, dict)
    for measurement in matrix["resources"]:
        assert isinstance(measurement, dict)
        measurement["observed_at_ns"] = 0

    target = _parse_json(example.target_decision_jsonl, "target decision")
    if not isinstance(target, dict):
        raise StaceyCorpusError("target decision must be a JSON object")
    target["transaction_id"] = "<transaction>"
    target["correlation_id"] = "<correlation>"
    semantic_payload = {
        "ingress": ingress,
        "target": target,
    }
    canonical = json.dumps(
        semantic_payload,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def load_approved_stacey_dataset(
    directory: str | Path,
    *,
    approval_verifier: GovernanceApprovalVerifier,
    approval: SignedGovernanceApproval,
) -> ApprovedStaceyDataset:
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise StaceyCorpusError("dataset directory must be an existing non-symlink directory")
    payload, manifest_bytes = _read_manifest(root / "manifest.json")
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    try:
        approval_verifier.verify(
            approval,
            action="DATASET_APPROVE",
            subject_sha256=manifest_sha256,
        )
    except (GovernanceApprovalError, AttributeError) as error:
        raise StaceyCorpusError("dataset lacks valid signed reviewer approval for these manifest bytes") from error
    expected_fields = {
        "schema_version",
        "dataset_id",
        "version",
        "provenance_reference",
        "license_review_reference",
        "content_sha256",
        "sources",
        "splits",
        "holdout",
    }
    if set(payload) != expected_fields:
        raise StaceyCorpusError("dataset manifest has missing or unknown fields")
    if payload["schema_version"] != "stacey.training.dataset.v1":
        raise StaceyCorpusError("unsupported Stacey training dataset schema version")

    sources = _source_records(payload["sources"])
    split_entries = payload["splits"]
    if not isinstance(split_entries, dict) or set(split_entries) != set(_TRAINING_SPLITS):
        raise StaceyCorpusError("splits must contain only training and validation files")
    split_contents: dict[str, bytes] = {}
    split_ids: dict[str, tuple[str, ...]] = {}
    for split_name in _TRAINING_SPLITS:
        split_contents[split_name], split_ids[split_name] = _split_file(
            root,
            split_entries[split_name],
            split_name,
        )
    holdout = payload["holdout"]
    if not isinstance(holdout, dict) or set(holdout) != {
        "sha256",
        "record_ids",
        "source_group_ids",
        "evaluation_reference",
    }:
        raise StaceyCorpusError("holdout metadata has missing or unknown fields")
    holdout_sha256 = holdout["sha256"]
    if not isinstance(holdout_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", holdout_sha256) is None:
        raise StaceyCorpusError("holdout.sha256 must be a lowercase SHA-256 digest")
    if not isinstance(holdout["record_ids"], list) or not holdout["record_ids"]:
        raise StaceyCorpusError("holdout.record_ids must be a non-empty array")
    if not isinstance(holdout["source_group_ids"], list) or not holdout["source_group_ids"]:
        raise StaceyCorpusError("holdout.source_group_ids must be a non-empty array")
    holdout_record_ids = tuple(
        _required_text(record_id, "holdout.record_ids") for record_id in holdout["record_ids"]
    )
    holdout_group_ids = tuple(
        _required_text(group_id, "holdout.source_group_ids") for group_id in holdout["source_group_ids"]
    )
    if len(set(holdout_record_ids)) != len(holdout_record_ids):
        raise StaceyCorpusError("holdout record IDs must be unique")
    if len(set(holdout_group_ids)) != len(holdout_group_ids):
        raise StaceyCorpusError("holdout source-group IDs must be unique")
    evaluation_reference = _required_text(holdout["evaluation_reference"], "holdout.evaluation_reference")
    content_sha256 = payload["content_sha256"]
    if not isinstance(content_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", content_sha256) is None:
        raise StaceyCorpusError("content_sha256 must be a lowercase SHA-256 digest")
    if dataset_content_sha256(
        split_contents,
        protected_holdout_sha256=holdout_sha256,
    ) != content_sha256:
        raise StaceyCorpusError("combined dataset content digest does not match the manifest")

    training_validation_ids = set(split_ids["training"]) | set(split_ids["validation"])
    if training_validation_ids & set(holdout_record_ids):
        raise StaceyCorpusError("protected holdout record IDs overlap training or validation")

    manifest = TrainingDatasetManifest(
        dataset_id=_required_text(payload["dataset_id"], "dataset_id"),
        version=_required_text(payload["version"], "version"),
        content_sha256=content_sha256,
        provenance_reference=_required_text(payload["provenance_reference"], "provenance_reference"),
        license_review_reference=_required_text(payload["license_review_reference"], "license_review_reference"),
        training_record_ids=split_ids["training"],
        validation_record_ids=split_ids["validation"],
        test_record_ids=holdout_record_ids,
        sources=sources,
    )
    sources_by_id = {source.source_id: source for source in sources}
    examples = {
        split_name: _parse_examples(
            split_contents[split_name],
            split_name=split_name,
            expected_record_ids=split_ids[split_name],
            sources_by_id=sources_by_id,
        )
        for split_name in _TRAINING_SPLITS
    }
    training_group_ids = {example.source_group_id for example in examples["training"]}
    validation_group_ids = {example.source_group_id for example in examples["validation"]}
    if training_group_ids & validation_group_ids:
        raise StaceyCorpusError("source groups must not cross training, validation, and holdout splits")
    if (training_group_ids | validation_group_ids) & set(holdout_group_ids):
        raise StaceyCorpusError("protected holdout source groups overlap training or validation")
    semantic_fingerprints: dict[str, tuple[str, str]] = {}
    for split_name in _TRAINING_SPLITS:
        for example in examples[split_name]:
            fingerprint = _semantic_example_sha256(example)
            prior = semantic_fingerprints.get(fingerprint)
            if prior is not None:
                raise StaceyCorpusError(
                    "semantically duplicate examples cross or repeat dataset splits "
                    f"({prior[0]}:{prior[1]} and {split_name}:{example.example_id})"
                )
            semantic_fingerprints[fingerprint] = (split_name, example.example_id)

    return ApprovedStaceyDataset(
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        approval=approval,
        training_examples=examples["training"],
        validation_examples=examples["validation"],
        holdout_record_ids=holdout_record_ids,
        holdout_source_group_ids=holdout_group_ids,
        holdout_sha256=holdout_sha256,
        holdout_evaluation_reference=evaluation_reference,
    )