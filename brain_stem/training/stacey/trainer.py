from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch import Tensor

from models.stacey.core import StaceyCoreConfig, initialize_stacey_core, save_model_checkpoint
from models.stacey.core.tokens import PAD_TOKEN_ID
from src.swarm_core.hardware_profile import HardwareProfile
from src.swarm_core.training_readiness import (
    CoreTrainingReadinessChecker,
    CoreTrainingRunPlan,
    TrainingMethod,
)
from src.swarm_core.training_governance import (
    GovernanceApprovalError,
    GovernanceApprovalJournal,
    GovernanceApprovalVerifier,
    SignedGovernanceApproval,
)
from .resume_checkpoint import ResumeCheckpointError, load_resume_checkpoint, save_resume_checkpoint

from .corpus import ApprovedStaceyDataset, StaceyTrainingExample
from .objectives import StaceyTrainingError, teacher_forcing_loss


class StaceyTrainingRunError(RuntimeError):
    pass


class StaceyTrainingBlockedError(StaceyTrainingRunError):
    def __init__(self, blockers: tuple[str, ...]) -> None:
        self.blockers = blockers
        super().__init__("training readiness blocked: " + "; ".join(blockers))


@dataclass(frozen=True, slots=True)
class StaceyTrainerConfig:
    device_id: str
    learning_rate: float
    weight_decay: float
    micro_batch_size: int
    gradient_accumulation_steps: int
    gradient_clip_norm: float
    validation_interval_steps: int
    early_stopping_patience: int
    maximum_hardware_profile_age_seconds: int

    def __post_init__(self) -> None:
        if not isinstance(self.device_id, str) or not self.device_id.strip():
            raise ValueError("device_id must be a non-empty string")
        for field_name in ("learning_rate", "gradient_clip_norm"):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{field_name} must be a finite positive number")
        if (
            isinstance(self.weight_decay, bool)
            or not isinstance(self.weight_decay, (int, float))
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
        ):
            raise ValueError("weight_decay must be a finite non-negative number")
        for field_name in (
            "micro_batch_size",
            "gradient_accumulation_steps",
            "validation_interval_steps",
            "early_stopping_patience",
            "maximum_hardware_profile_age_seconds",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.device_id != "cpu" and re.fullmatch(r"cuda:[0-9]+", self.device_id) is None:
            raise ValueError("device_id must be 'cpu' or a CUDA device such as 'cuda:0'")


def training_config_sha256(
    model_config: StaceyCoreConfig,
    trainer_config: StaceyTrainerConfig,
) -> str:
    if not isinstance(model_config, StaceyCoreConfig) or not isinstance(trainer_config, StaceyTrainerConfig):
        raise TypeError("training_config_sha256 requires model and trainer configs")
    payload = {
        "model_config": model_config.to_dict(),
        "trainer_config": asdict(trainer_config),
        "optimizer": "torch.optim.AdamW",
        "precision": "fp32",
        "dataset_mode": "human_reviewed_task_graph_sft_v1",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class StaceyTrainingRunReport:
    candidate_id: str
    readiness_plan_sha256: str
    dataset_sha256: str
    optimizer_steps: int
    best_validation_loss: float
    final_training_loss: float
    checkpoint_path: str
    checkpoint_sha256: str
    stop_reason: str
    resume_state_path: str | None = None
    resume_state_sha256: str | None = None
    resumed_from_step: int = 0


def _batch_tensors(
    examples: tuple[StaceyTrainingExample, ...],
    *,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    ingresses = tuple(example.ingress for example in examples)
    source_ids = model.encode_ingresses(ingresses, device=device)  # type: ignore[attr-defined]
    encoded_targets = tuple(model.codec.encode(example.target_decision_jsonl) for example in examples)  # type: ignore[attr-defined]
    target_length = max(len(tokens) for tokens in encoded_targets)
    if target_length > model.config.maximum_output_tokens:  # type: ignore[attr-defined]
        raise StaceyTrainingRunError("training target exceeds configured maximum_output_tokens")
    target_ids = torch.full(
        (len(encoded_targets), target_length),
        PAD_TOKEN_ID,
        dtype=torch.long,
        device=device,
    )
    for row_index, tokens in enumerate(encoded_targets):
        target_ids[row_index, : len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)
    return source_ids, target_ids


def _batch_stream(
    examples: tuple[StaceyTrainingExample, ...],
    *,
    batch_size: int,
    rng: random.Random,
) -> Iterator[tuple[StaceyTrainingExample, ...]]:
    while True:
        indices = list(range(len(examples)))
        rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            yield tuple(examples[index] for index in indices[start : start + batch_size])


class _ResumableBatchStream:
    def __init__(
        self,
        examples: tuple[StaceyTrainingExample, ...],
        *,
        batch_size: int,
        seed: int,
        state: object | None = None,
    ) -> None:
        if not examples:
            raise StaceyTrainingRunError("training split is empty")
        self._examples = examples
        self._batch_size = batch_size
        self._rng = random.Random(seed)
        self._indices: list[int] = []
        self._offset = 0
        self._epoch = 0
        if state is None:
            return
        if not isinstance(state, dict) or set(state) != {"indices", "offset", "epoch", "rng_state"}:
            raise StaceyTrainingRunError("resume data cursor has an invalid shape")
        indices = state["indices"]
        offset = state["offset"]
        epoch = state["epoch"]
        rng_state = state["rng_state"]
        if (
            not isinstance(indices, list)
            or len(indices) != len(examples)
            or any(isinstance(index, bool) or not isinstance(index, int) for index in indices)
            or sorted(indices) != list(range(len(examples)))
            or isinstance(offset, bool)
            or not isinstance(offset, int)
            or not 0 <= offset <= len(indices)
            or isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or epoch < 0
        ):
            raise StaceyTrainingRunError("resume data cursor values are invalid")
        try:
            self._rng.setstate(rng_state)
        except (TypeError, ValueError) as error:
            raise StaceyTrainingRunError("resume data cursor RNG state is invalid") from error
        self._indices = list(indices)
        self._offset = offset
        self._epoch = epoch

    def next_batch(self) -> tuple[StaceyTrainingExample, ...]:
        if not self._indices or self._offset >= len(self._indices):
            self._indices = list(range(len(self._examples)))
            self._rng.shuffle(self._indices)
            self._offset = 0
            self._epoch += 1
        selected = self._indices[self._offset : self._offset + self._batch_size]
        self._offset += len(selected)
        return tuple(self._examples[index] for index in selected)

    def state(self) -> dict[str, object]:
        return {
            "indices": list(self._indices),
            "offset": self._offset,
            "epoch": self._epoch,
            "rng_state": self._rng.getstate(),
        }


def _evaluate_loss(
    model: torch.nn.Module,
    examples: tuple[StaceyTrainingExample, ...],
    *,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    weighted_loss = 0.0
    example_count = 0
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            source_ids, target_ids = _batch_tensors(batch, model=model, device=device)
            logits = model(source_ids, target_ids[:, :-1])
            loss = teacher_forcing_loss(logits, target_ids[:, 1:])
            if not bool(torch.isfinite(loss)):
                raise StaceyTrainingRunError("validation produced non-finite loss")
            weighted_loss += float(loss) * len(batch)
            example_count += len(batch)
    if not example_count:
        raise StaceyTrainingRunError("validation split is empty")
    return weighted_loss / example_count


def train_stacey_from_scratch(
    *,
    plan: CoreTrainingRunPlan,
    dataset: ApprovedStaceyDataset,
    hardware_profile: HardwareProfile,
    model_config: StaceyCoreConfig,
    trainer_config: StaceyTrainerConfig,
    approval_verifier: GovernanceApprovalVerifier,
    approval_journal: GovernanceApprovalJournal,
    run_approval: SignedGovernanceApproval,
    checkpoint_path: str | Path,
    checkpoint_file_mode: int,
    resume_state_path: str | Path | None = None,
    resume_from_sha256: str | None = None,
) -> StaceyTrainingRunReport:
    """Run only the bounded task-graph SFT phase from random initialization."""
    if not isinstance(plan, CoreTrainingRunPlan):
        raise TypeError("plan must be a CoreTrainingRunPlan")
    if not isinstance(dataset, ApprovedStaceyDataset):
        raise TypeError("dataset must be an ApprovedStaceyDataset loaded by the strict corpus loader")
    if not isinstance(hardware_profile, HardwareProfile):
        raise TypeError("hardware_profile must be a measured HardwareProfile")
    if not isinstance(model_config, StaceyCoreConfig) or not isinstance(trainer_config, StaceyTrainerConfig):
        raise TypeError("model_config and trainer_config must be validated configs")
    if plan.method is not TrainingMethod.FROM_SCRATCH:
        raise StaceyTrainingRunError("strict Stacey training runner accepts FROM_SCRATCH plans only")
    if plan.dataset != dataset.manifest:
        raise StaceyTrainingRunError("training plan dataset manifest does not match the validated corpus")
    if plan.architecture_reference != model_config.architecture_id:
        raise StaceyTrainingRunError("run plan architecture does not match model configuration")
    if plan.tokenizer_reference != model_config.tokenizer_id:
        raise StaceyTrainingRunError("run plan tokenizer does not match the Stacey-owned tokenizer config")
    if plan.random_seed < 0:
        raise StaceyTrainingRunError("run plan random seed is invalid")
    if plan.context_length_tokens < max(
        model_config.maximum_input_tokens,
        model_config.maximum_output_tokens,
    ):
        raise StaceyTrainingRunError("run context limit is below configured input/output bounds")
    if plan.training_config_sha256 != training_config_sha256(model_config, trainer_config):
        raise StaceyTrainingRunError("training configuration digest does not match the approved run plan")
    if not callable(getattr(approval_verifier, "verify", None)):
        raise StaceyTrainingBlockedError(("trusted governance approval verifier is unavailable",))
    if not callable(getattr(approval_journal, "consume", None)):
        raise StaceyTrainingBlockedError(("durable training approval journal is unavailable",))
    is_resume = resume_from_sha256 is not None
    if is_resume and not callable(getattr(approval_journal, "confirm_consumed", None)):
        raise StaceyTrainingBlockedError(("durable approval journal cannot confirm a prior run for resume",))
    if resume_state_path is None and resume_from_sha256 is not None:
        raise StaceyTrainingRunError("resume_from_sha256 requires resume_state_path")
    if resume_from_sha256 is not None and re.fullmatch(r"[0-9a-f]{64}", resume_from_sha256) is None:
        raise StaceyTrainingRunError("resume_from_sha256 must be a lowercase SHA-256 digest")
    if not isinstance(run_approval, SignedGovernanceApproval):
        raise StaceyTrainingBlockedError(("signed training-run approval is unavailable",))
    try:
        dataset_approval_reference = approval_verifier.verify(
            dataset.approval,
            action="DATASET_APPROVE",
            subject_sha256=dataset.manifest_sha256,
        )
        approval_subject = CoreTrainingReadinessChecker.plan_digest(plan)
        if is_resume:
            run_approval_reference = approval_journal.confirm_consumed(
                run_approval,
                action="TRAINING_RUN_APPROVE",
                subject_sha256=approval_subject,
            )
        else:
            run_approval_reference = approval_journal.consume(
                run_approval,
                action="TRAINING_RUN_APPROVE",
                subject_sha256=approval_subject,
            )
    except GovernanceApprovalError as error:
        raise StaceyTrainingBlockedError((str(error),)) from error
    if dataset_approval_reference != dataset.approval.approval_id:
        raise StaceyTrainingBlockedError(("dataset approval reference does not match verified evidence",))
    if run_approval_reference != plan.run_approval_reference:
        raise StaceyTrainingBlockedError(("training approval reference does not match the approved run plan",))
    if plan.execution_profile.accelerator_profile_id != trainer_config.device_id:
        raise StaceyTrainingRunError("execution profile device does not match trainer device")
    if plan.execution_profile.precision_profile_id != "fp32":
        raise StaceyTrainingRunError("this scratch runner currently supports the explicit fp32 profile only")

    now_ns = time.time_ns()
    profile_age_ns = now_ns - hardware_profile.observed_at_ns
    maximum_age_ns = trainer_config.maximum_hardware_profile_age_seconds * 1_000_000_000
    if profile_age_ns < 0 or profile_age_ns > maximum_age_ns:
        raise StaceyTrainingRunError("hardware profile is stale or has a future timestamp")

    device = torch.device(trainer_config.device_id)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise StaceyTrainingRunError("requested CUDA device is not visible to PyTorch")
    if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
        raise StaceyTrainingRunError("requested CUDA device index is not visible to PyTorch")

    destination = Path(checkpoint_path)
    if (destination.exists() or destination.is_symlink()) and not is_resume:
        raise StaceyTrainingRunError("checkpoint destination must be a new, non-symlink path")
    if not destination.parent.is_dir():
        raise StaceyTrainingRunError("checkpoint parent directory must already exist")
    resume_destination = None if resume_state_path is None else Path(resume_state_path)
    if resume_destination is not None:
        if resume_destination.is_symlink() or not resume_destination.name or not resume_destination.parent.is_dir():
            raise StaceyTrainingRunError("resume-state path must be a non-symlink file in an existing directory")
        if resume_destination.resolve() == destination.resolve():
            raise StaceyTrainingRunError("resume-state and candidate checkpoint paths must be distinct")
        if is_resume and not resume_destination.is_file():
            raise StaceyTrainingRunError("resume-state checkpoint does not exist")
        if not is_resume and resume_destination.exists():
            raise StaceyTrainingRunError("resume-state path already exists; supply its digest to continue")
    artifact_store_reference = str(destination.parent.resolve())
    environment = hardware_profile.to_training_environment_evidence(
        device_id=trainer_config.device_id,
        execution_profile_id=plan.execution_profile.profile_id,
        accessible_dataset_ids=(dataset.manifest.dataset_id,),
        approved_run_references=(run_approval_reference,),
        artifact_store_reference=artifact_store_reference,
    )
    readiness = CoreTrainingReadinessChecker().assess(plan, environment)
    if not readiness.ready_to_start:
        raise StaceyTrainingBlockedError(readiness.blockers)

    plan_digest = readiness.plan_sha256
    resume_state: dict[str, object] | None = None
    if is_resume:
        try:
            resume_state = load_resume_checkpoint(
                resume_destination,
                expected_sha256=resume_from_sha256,
                map_location="cpu",
            )
        except ResumeCheckpointError as error:
            raise StaceyTrainingRunError("resume checkpoint could not be verified") from error
        expected_state_fields = {
            "format_id",
            "candidate_id",
            "plan_sha256",
            "dataset_sha256",
            "training_config_sha256",
            "checkpoint_path",
            "optimizer_steps",
            "model_state_dict",
            "optimizer_state_dict",
            "batch_stream_state",
            "best_validation_loss",
            "final_training_loss",
            "unimproved_validations",
            "best_checkpoint_sha256",
            "torch_cpu_rng_state",
            "torch_cuda_rng_state",
        }
        if set(resume_state) != expected_state_fields:
            raise StaceyTrainingRunError("resume checkpoint has an unexpected field set")
        if (
            resume_state["candidate_id"] != plan.candidate_id
            or resume_state["plan_sha256"] != plan_digest
            or resume_state["dataset_sha256"] != dataset.manifest.content_sha256
            or resume_state["training_config_sha256"] != plan.training_config_sha256
            or resume_state["checkpoint_path"] != str(destination.resolve())
        ):
            raise StaceyTrainingRunError("resume checkpoint is bound to a different run, dataset, config, or path")
        resumed_from_step = resume_state["optimizer_steps"]
        if (
            isinstance(resumed_from_step, bool)
            or not isinstance(resumed_from_step, int)
            or not 0 < resumed_from_step < plan.execution_profile.maximum_steps
        ):
            raise StaceyTrainingRunError("resume checkpoint step is outside the remaining run budget")
        for field_name in ("unimproved_validations",):
            value = resume_state[field_name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise StaceyTrainingRunError(f"resume checkpoint {field_name} is invalid")
        for field_name in ("best_validation_loss", "final_training_loss"):
            value = resume_state[field_name]
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise StaceyTrainingRunError(f"resume checkpoint {field_name} is invalid")
        best_checkpoint_sha256 = resume_state["best_checkpoint_sha256"]
        if best_checkpoint_sha256 is not None:
            if (
                not isinstance(best_checkpoint_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", best_checkpoint_sha256) is None
                or not destination.is_file()
                or hashlib.sha256(destination.read_bytes()).hexdigest() != best_checkpoint_sha256
            ):
                raise StaceyTrainingRunError("best validation checkpoint is missing or has changed")
        elif destination.exists():
            raise StaceyTrainingRunError("unselected model checkpoint exists without a bound validation digest")
    else:
        resumed_from_step = 0
        best_checkpoint_sha256 = None

    torch.manual_seed(plan.random_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(plan.random_seed)
    model = initialize_stacey_core(model_config, random_seed=plan.random_seed).to(device)
    if model.parameter_count != plan.parameter_count:
        raise StaceyTrainingRunError("run plan parameter count does not match initialized random model")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=trainer_config.learning_rate,
        weight_decay=trainer_config.weight_decay,
    )
    if resume_state is not None:
        try:
            model.load_state_dict(resume_state["model_state_dict"], strict=True)
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])
            cpu_rng_state = resume_state["torch_cpu_rng_state"]
            if not isinstance(cpu_rng_state, torch.Tensor) or cpu_rng_state.dtype is not torch.uint8:
                raise ValueError("invalid CPU RNG state")
            torch.set_rng_state(cpu_rng_state)
            cuda_rng_state = resume_state["torch_cuda_rng_state"]
            if device.type == "cuda":
                if not isinstance(cuda_rng_state, torch.Tensor) or cuda_rng_state.dtype is not torch.uint8:
                    raise ValueError("invalid CUDA RNG state")
                torch.cuda.set_rng_state(cuda_rng_state, device)
            elif cuda_rng_state is not None:
                raise ValueError("resume checkpoint was captured on a different device type")
        except Exception as error:
            raise StaceyTrainingRunError("resume model, optimizer, or RNG state is incompatible") from error
        stream_state = resume_state["batch_stream_state"]
    else:
        stream_state = None
    train_stream = _ResumableBatchStream(
        dataset.training_examples,
        batch_size=trainer_config.micro_batch_size,
        seed=plan.random_seed,
        state=stream_state,
    )
    best_loss_state = None if resume_state is None else resume_state["best_validation_loss"]
    best_validation_loss = math.inf if best_loss_state is None else float(best_loss_state)
    final_loss_state = None if resume_state is None else resume_state["final_training_loss"]
    final_training_loss = math.inf if final_loss_state is None else float(final_loss_state)
    optimizer_steps = resumed_from_step
    unimproved_validations = 0 if resume_state is None else resume_state["unimproved_validations"]
    resume_state_sha256: str | None = resume_from_sha256
    stop_reason = "maximum_steps"
    started_at = time.monotonic()

    for step in range(resumed_from_step + 1, plan.execution_profile.maximum_steps + 1):
        if time.monotonic() - started_at >= plan.execution_profile.maximum_wall_time_seconds:
            stop_reason = "maximum_wall_time"
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(trainer_config.gradient_accumulation_steps):
            batch = train_stream.next_batch()
            source_ids, target_ids = _batch_tensors(batch, model=model, device=device)
            logits = model(source_ids, target_ids[:, :-1])
            loss = teacher_forcing_loss(logits, target_ids[:, 1:])
            if not bool(torch.isfinite(loss)):
                raise StaceyTrainingRunError("training produced non-finite loss")
            (loss / trainer_config.gradient_accumulation_steps).backward()
            accumulated_loss += float(loss.detach()) / trainer_config.gradient_accumulation_steps

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            trainer_config.gradient_clip_norm,
        )
        if not bool(torch.isfinite(gradient_norm)):
            optimizer.zero_grad(set_to_none=True)
            raise StaceyTrainingRunError("training produced non-finite gradient norm")
        optimizer.step()
        optimizer_steps = step
        final_training_loss = accumulated_loss

        if step % plan.execution_profile.checkpoint_interval_steps == 0:
            snapshot_path = destination.with_name(
                f"{destination.stem}.step-{step:08d}{destination.suffix}"
            )
            if snapshot_path.exists() or snapshot_path.is_symlink():
                if not is_resume:
                    raise StaceyTrainingRunError("periodic checkpoint destination already exists")
            else:
                save_model_checkpoint(model, snapshot_path, file_mode=checkpoint_file_mode)

        should_validate = (
            step % trainer_config.validation_interval_steps == 0
            or step == plan.execution_profile.maximum_steps
        )
        should_stop = False
        if should_validate:
            validation_loss = _evaluate_loss(
                model,
                dataset.validation_examples,
                batch_size=trainer_config.micro_batch_size,
                device=device,
            )
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                unimproved_validations = 0
                best_checkpoint_sha256 = save_model_checkpoint(
                    model,
                    destination,
                    file_mode=checkpoint_file_mode,
                )
            else:
                unimproved_validations += 1
                if unimproved_validations >= trainer_config.early_stopping_patience:
                    stop_reason = "validation_early_stopping"
                    should_stop = True

        if resume_destination is not None and (
            step % plan.execution_profile.checkpoint_interval_steps == 0 or should_validate
        ):
            state_payload: dict[str, object] = {
                "format_id": "stacey.training.resume.v1",
                "candidate_id": plan.candidate_id,
                "plan_sha256": plan_digest,
                "dataset_sha256": dataset.manifest.content_sha256,
                "training_config_sha256": plan.training_config_sha256,
                "checkpoint_path": str(destination.resolve()),
                "optimizer_steps": optimizer_steps,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "batch_stream_state": train_stream.state(),
                "best_validation_loss": None if math.isinf(best_validation_loss) else best_validation_loss,
                "final_training_loss": None if math.isinf(final_training_loss) else final_training_loss,
                "unimproved_validations": unimproved_validations,
                "best_checkpoint_sha256": best_checkpoint_sha256,
                "torch_cpu_rng_state": torch.get_rng_state(),
                "torch_cuda_rng_state": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            }
            try:
                resume_state_sha256 = save_resume_checkpoint(
                    state_payload,
                    resume_destination,
                    file_mode=checkpoint_file_mode,
                )
            except ResumeCheckpointError as error:
                raise StaceyTrainingRunError("resume checkpoint could not be persisted") from error
        if should_stop:
            break

    if optimizer_steps == 0:
        raise StaceyTrainingRunError("training stopped before the first optimizer update")
    if not destination.is_file():
        validation_loss = _evaluate_loss(
            model,
            dataset.validation_examples,
            batch_size=trainer_config.micro_batch_size,
            device=device,
        )
        best_validation_loss = validation_loss
        best_checkpoint_sha256 = save_model_checkpoint(
            model,
            destination,
            file_mode=checkpoint_file_mode,
        )
        if resume_destination is not None:
            state_payload = {
                "format_id": "stacey.training.resume.v1",
                "candidate_id": plan.candidate_id,
                "plan_sha256": plan_digest,
                "dataset_sha256": dataset.manifest.content_sha256,
                "training_config_sha256": plan.training_config_sha256,
                "checkpoint_path": str(destination.resolve()),
                "optimizer_steps": optimizer_steps,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "batch_stream_state": train_stream.state(),
                "best_validation_loss": best_validation_loss,
                "final_training_loss": final_training_loss,
                "unimproved_validations": unimproved_validations,
                "best_checkpoint_sha256": best_checkpoint_sha256,
                "torch_cpu_rng_state": torch.get_rng_state(),
                "torch_cuda_rng_state": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            }
            try:
                resume_state_sha256 = save_resume_checkpoint(
                    state_payload,
                    resume_destination,
                    file_mode=checkpoint_file_mode,
                )
            except ResumeCheckpointError as error:
                raise StaceyTrainingRunError("fallback resume checkpoint could not be persisted") from error
    checkpoint_sha256 = hashlib.sha256(destination.read_bytes()).hexdigest()
    return StaceyTrainingRunReport(
        candidate_id=plan.candidate_id,
        readiness_plan_sha256=readiness.plan_sha256,
        dataset_sha256=dataset.manifest.content_sha256,
        optimizer_steps=optimizer_steps,
        best_validation_loss=best_validation_loss,
        final_training_loss=final_training_loss,
        checkpoint_path=str(destination),
        checkpoint_sha256=checkpoint_sha256,
        stop_reason=stop_reason,
        resume_state_path=None if resume_destination is None else str(resume_destination),
        resume_state_sha256=resume_state_sha256,
        resumed_from_step=resumed_from_step,
    )