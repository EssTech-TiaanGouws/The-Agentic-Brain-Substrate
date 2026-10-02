from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from models.stacey.core.inputs import (
    ActiveResourceLease,
    HardwareCapabilityMatrix,
    IntentKind,
    IntentVector,
    LedgerAssertion,
    ResourceMeasurement,
    SpecialistAvailability,
    SpecialistSlot,
    UnifiedContextIngress,
)
from models.stacey.core.outputs import (
    CoreDecisionEnvelope,
    EdgeCondition,
    parse_decision_jsonl,
)
from substrate.contracts import ScopeVector

from .model_catalog import EvaluationEvidence


class StaceyBenchmarkError(ValueError):
    pass


class StaceyBenchmarkSplit(str, Enum):
    DEVELOPMENT = "DEVELOPMENT"
    HOLDOUT_FIXTURE = "HOLDOUT_FIXTURE"


_METRICS = (
    "case_pass_rate",
    "scope_integrity_rate",
    "task_graph_validity_rate",
    "clarification_accuracy_rate",
    "capability_coverage_rate",
    "required_dependency_rate",
    "consequence_prediction_rate",
    "resource_estimate_accuracy_rate",
)


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StaceyBenchmarkError(f"{field} must be a non-empty string")
    return value


def _text_tuple(values: object, field: str, *, allow_empty: bool = True) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise StaceyBenchmarkError(f"{field} must be a tuple")
    if not allow_empty and not values:
        raise StaceyBenchmarkError(f"{field} must not be empty")
    for value in values:
        _text(value, field)
    if len(set(values)) != len(values):
        raise StaceyBenchmarkError(f"{field} must not contain duplicates")
    return values


@dataclass(frozen=True, slots=True)
class ResourceEstimateExpectation:
    capability_id: str
    resource_domain: str
    minimum_bytes: int
    maximum_bytes: int

    def __post_init__(self) -> None:
        _text(self.capability_id, "capability_id")
        _text(self.resource_domain, "resource_domain")
        for name in ("minimum_bytes", "maximum_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise StaceyBenchmarkError(f"{name} must be a non-negative integer")
        if self.minimum_bytes > self.maximum_bytes:
            raise StaceyBenchmarkError("minimum_bytes must not exceed maximum_bytes")


@dataclass(frozen=True, slots=True)
class StaceyCoreBenchmarkCase:
    case_id: str
    ingress: UnifiedContextIngress
    expected_clarification: bool
    required_capabilities: tuple[str, ...]
    forbidden_capabilities: tuple[str, ...]
    required_dependencies: tuple[tuple[str, str], ...]
    consequence_concepts: tuple[str, ...]
    resource_expectations: tuple[ResourceEstimateExpectation, ...]

    def __post_init__(self) -> None:
        _text(self.case_id, "case_id")
        if not isinstance(self.ingress, UnifiedContextIngress):
            raise StaceyBenchmarkError("ingress must be a UnifiedContextIngress")
        if not isinstance(self.expected_clarification, bool):
            raise StaceyBenchmarkError("expected_clarification must be a boolean")
        _text_tuple(self.required_capabilities, "required_capabilities")
        _text_tuple(self.forbidden_capabilities, "forbidden_capabilities")
        _text_tuple(self.consequence_concepts, "consequence_concepts", allow_empty=False)
        if set(self.required_capabilities) & set(self.forbidden_capabilities):
            raise StaceyBenchmarkError("a capability cannot be both required and forbidden")
        if not isinstance(self.required_dependencies, tuple):
            raise StaceyBenchmarkError("required_dependencies must be a tuple")
        for dependency in self.required_dependencies:
            if not isinstance(dependency, tuple) or len(dependency) != 2:
                raise StaceyBenchmarkError("dependencies must be capability pairs")
            _text(dependency[0], "dependency source capability")
            _text(dependency[1], "dependency target capability")
        if not isinstance(self.resource_expectations, tuple) or any(
            not isinstance(expectation, ResourceEstimateExpectation)
            for expectation in self.resource_expectations
        ):
            raise StaceyBenchmarkError("resource_expectations contains invalid entries")
        expectation_keys = tuple(
            (item.capability_id, item.resource_domain) for item in self.resource_expectations
        )
        if len(set(expectation_keys)) != len(expectation_keys):
            raise StaceyBenchmarkError("resource expectations must be unique per capability and domain")
        if not {item.capability_id for item in self.resource_expectations} <= set(self.required_capabilities):
            raise StaceyBenchmarkError("resource expectations must refer to required capabilities")
        if self.expected_clarification and (self.required_capabilities or self.resource_expectations):
            raise StaceyBenchmarkError("clarification cases cannot require specialist execution")


@dataclass(frozen=True, slots=True)
class StaceyCoreBenchmarkSuite:
    suite_id: str
    split: StaceyBenchmarkSplit
    cases: tuple[StaceyCoreBenchmarkCase, ...]

    def __post_init__(self) -> None:
        _text(self.suite_id, "suite_id")
        if not isinstance(self.split, StaceyBenchmarkSplit):
            raise StaceyBenchmarkError("split must be a StaceyBenchmarkSplit")
        if not isinstance(self.cases, tuple) or not self.cases:
            raise StaceyBenchmarkError("suite must contain at least one case")
        if any(not isinstance(case, StaceyCoreBenchmarkCase) for case in self.cases):
            raise StaceyBenchmarkError("suite contains an invalid case")
        case_ids = tuple(case.case_id for case in self.cases)
        if len(set(case_ids)) != len(case_ids):
            raise StaceyBenchmarkError("case IDs must be unique")


@dataclass(frozen=True, slots=True)
class StaceyCoreBenchmarkPolicy:
    minimum_case_pass_rate: float
    minimum_metric_rates: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        _rate(self.minimum_case_pass_rate, "minimum_case_pass_rate")
        if not isinstance(self.minimum_metric_rates, tuple):
            raise StaceyBenchmarkError("minimum_metric_rates must be a tuple")
        seen: set[str] = set()
        for entry in self.minimum_metric_rates:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise StaceyBenchmarkError("metric thresholds must be (name, rate) pairs")
            name, rate = entry
            _text(name, "metric name")
            if name not in _METRICS or name in seen:
                raise StaceyBenchmarkError(f"unknown or duplicate metric: {name}")
            _rate(rate, name)
            seen.add(name)


def _rate(value: object, field: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise StaceyBenchmarkError(f"{field} must be a finite value in [0, 1]")


class StaceyDecisionCandidate(Protocol):
    candidate_id: str
    artifact_sha256: str

    def decide(self, ingress: UnifiedContextIngress) -> CoreDecisionEnvelope: ...


@dataclass(frozen=True, slots=True)
class StaceyCoreCaseResult:
    case_id: str
    checks: tuple[tuple[str, bool], ...]
    error_type: str | None = None

    @property
    def passed(self) -> bool:
        return self.error_type is None and all(value for _, value in self.checks)


@dataclass(frozen=True, slots=True)
class StaceyCoreBenchmarkReport:
    candidate_id: str
    suite_id: str
    case_results: tuple[StaceyCoreCaseResult, ...]
    metric_rates: tuple[tuple[str, float], ...]
    accepted: bool
    evidence: EvaluationEvidence


class StaceyCoreBenchmarkRunner:
    """Scores validated Core envelopes against explicit task-graph expectations."""

    def run(
        self,
        *,
        candidate: StaceyDecisionCandidate,
        suite: StaceyCoreBenchmarkSuite,
        policy: StaceyCoreBenchmarkPolicy,
        run_reference: str,
        baseline_reference: str,
    ) -> StaceyCoreBenchmarkReport:
        candidate_id = _text(getattr(candidate, "candidate_id", None), "candidate_id")
        artifact_digest = getattr(candidate, "artifact_sha256", None)
        if not isinstance(artifact_digest, str) or re.fullmatch(r"[0-9a-f]{64}", artifact_digest) is None:
            raise StaceyBenchmarkError("candidate must expose a lowercase SHA-256 artifact digest")
        if not isinstance(suite, StaceyCoreBenchmarkSuite):
            raise StaceyBenchmarkError("suite must be a StaceyCoreBenchmarkSuite")
        if not isinstance(policy, StaceyCoreBenchmarkPolicy):
            raise StaceyBenchmarkError("policy must be a StaceyCoreBenchmarkPolicy")
        _text(run_reference, "run_reference")
        _text(baseline_reference, "baseline_reference")

        case_results = tuple(self._evaluate_case(candidate, case) for case in suite.cases)
        metric_rates = self._calculate_rates(case_results)
        rates = dict(metric_rates)
        accepted = (
            rates["case_pass_rate"] >= policy.minimum_case_pass_rate
            and all(rates[name] >= threshold for name, threshold in policy.minimum_metric_rates)
        )
        evidence = EvaluationEvidence(
            suite_id=suite.suite_id,
            run_reference=run_reference,
            artifact_sha256=artifact_digest,
            baseline_reference=baseline_reference,
            passed=accepted,
            metrics=metric_rates,
        )
        return StaceyCoreBenchmarkReport(
            candidate_id=candidate_id,
            suite_id=suite.suite_id,
            case_results=case_results,
            metric_rates=metric_rates,
            accepted=accepted,
            evidence=evidence,
        )

    @staticmethod
    def _evaluate_case(
        candidate: StaceyDecisionCandidate,
        case: StaceyCoreBenchmarkCase,
    ) -> StaceyCoreCaseResult:
        try:
            proposed = candidate.decide(case.ingress)
            if not isinstance(proposed, CoreDecisionEnvelope):
                raise StaceyBenchmarkError("candidate returned an invalid decision type")
            decision = parse_decision_jsonl(proposed.to_jsonl(), case.ingress)
        except Exception as error:
            return StaceyCoreCaseResult(
                case_id=case.case_id,
                checks=tuple((name, False) for name in _METRICS if name != "case_pass_rate"),
                error_type=type(error).__name__,
            )

        graph = decision.task_dependency_graph
        nodes_by_id = {node.step_id: node for node in graph.nodes}
        capabilities = {node.capability_id for node in graph.nodes}
        edges = {
            (edge.source_step_id, edge.target_step_id, edge.condition)
            for edge in graph.edges
        }
        dependencies_valid = all(
            any(
                nodes_by_id[source_step].capability_id == upstream
                and nodes_by_id[target_step].capability_id == downstream
                and condition is EdgeCondition.ON_SUCCESS
                for source_step, target_step, condition in edges
            )
            for upstream, downstream in case.required_dependencies
        )
        estimate_by_step = {estimate.step_id: estimate for estimate in graph.resource_estimates}
        node_by_capability = {node.capability_id: node for node in graph.nodes}
        resources_valid = all(
            expectation.capability_id in node_by_capability
            and (estimate := estimate_by_step[node_by_capability[expectation.capability_id].step_id]).resource_domain
            == expectation.resource_domain
            and expectation.minimum_bytes <= estimate.estimated_required_bytes <= expectation.maximum_bytes
            for expectation in case.resource_expectations
        )
        normalized_summary = " ".join(decision.predicted_consequences_summary.casefold().split())
        consequences_valid = all(
            " ".join(concept.casefold().split()) in normalized_summary
            for concept in case.consequence_concepts
        )
        checks = (
            ("scope_integrity_rate", decision.scope_vector == case.ingress.scope_vector),
            ("task_graph_validity_rate", True),
            (
                "clarification_accuracy_rate",
                decision.clarification.required == case.expected_clarification,
            ),
            (
                "capability_coverage_rate",
                set(case.required_capabilities) <= capabilities
                and not (set(case.forbidden_capabilities) & capabilities),
            ),
            ("required_dependency_rate", dependencies_valid),
            ("consequence_prediction_rate", consequences_valid),
            ("resource_estimate_accuracy_rate", resources_valid),
        )
        return StaceyCoreCaseResult(case.case_id, checks)

    @staticmethod
    def _calculate_rates(
        results: tuple[StaceyCoreCaseResult, ...],
    ) -> tuple[tuple[str, float], ...]:
        denominator = len(results)
        if denominator == 0:
            raise StaceyBenchmarkError("cannot score an empty result set")
        rates = {
            "case_pass_rate": sum(result.passed for result in results) / denominator,
        }
        for metric in _METRICS:
            if metric == "case_pass_rate":
                continue
            rates[metric] = sum(dict(result.checks).get(metric, False) for result in results) / denominator
        return tuple(sorted(rates.items()))


def build_stacey_task_graph_fixture_suite() -> StaceyCoreBenchmarkSuite:
    """Small reproducible rubric fixture; not a held-out semantic benchmark."""
    scope = ScopeVector("tenant-benchmark", "user-benchmark", "project-benchmark", "workspace-benchmark")
    observed_at_ns = 1_800_000_000_000_000_000
    facts = (
        LedgerAssertion(
            assertion_id="reviewed-record-state",
            content_summary="The selected record set is present and verified.",
            provenance_sha256="a" * 64,
            last_observed_state="VERIFIED",
        ),
    )
    normal_slots = (
        SpecialistSlot(
            block_id="BLOCK_10_STRUCTURAL_INGRESS",
            capability_id="data.extract",
            status=SpecialistAvailability.AVAILABLE,
            resource_domain="accelerator-memory",
            estimated_required_bytes=120_000_000,
            artifact_sha256="b" * 64,
            input_modalities=("structured-data",),
        ),
        SpecialistSlot(
            block_id="BLOCK_12_LINGUISTIC_COPY",
            capability_id="text.summarize",
            status=SpecialistAvailability.AVAILABLE,
            resource_domain="accelerator-memory",
            estimated_required_bytes=80_000_000,
            artifact_sha256="c" * 64,
            input_modalities=("text",),
        ),
    )

    def ingress(
        case_id: str,
        intent: str,
        slots: tuple[SpecialistSlot, ...],
        assertions: tuple[LedgerAssertion, ...] = facts,
    ) -> UnifiedContextIngress:
        return UnifiedContextIngress(
            protocol_version="1.0",
            transaction_id=f"tx-{case_id}",
            correlation_id=f"turn-{case_id}",
            ingress_timestamp_ns=observed_at_ns,
            intent_vector=IntentVector(IntentKind.USER_LANGUAGE, intent),
            scope_vector=scope,
            canonical_state_assertions=assertions,
            hardware_capability_matrix=HardwareCapabilityMatrix(
                resources=(ResourceMeasurement("accelerator-memory", 2_000_000_000, observed_at_ns),),
                active_leases=(ActiveResourceLease("lease-existing", "accelerator-memory", 100_000_000),),
                available_specialist_slots=slots,
            ),
        )

    normal_ingress = ingress(
        "extract-summarize",
        "Extract the verified records, then summarize the findings.",
        normal_slots,
    )
    ambiguous_ingress = ingress(
        "ambiguous-share",
        "Share it with them.",
        normal_slots,
        assertions=(
            LedgerAssertion(
                assertion_id="missing-recipient-context",
                content_summary="No recipient or shareable content was identified.",
                provenance_sha256="d" * 64,
                last_observed_state="MISSING_CONTEXT",
            ),
        ),
    )
    failed_slot = SpecialistSlot(
        block_id="BLOCK_10_STRUCTURAL_INGRESS",
        capability_id="data.extract",
        status=SpecialistAvailability.QUARANTINED,
        resource_domain="accelerator-memory",
        estimated_required_bytes=120_000_000,
        artifact_sha256="b" * 64,
        input_modalities=("structured-data",),
    )
    alternative_slot = SpecialistSlot(
        block_id="BLOCK_9_ALGORITHMIC_CODER",
        capability_id="data.extract.alternative",
        status=SpecialistAvailability.AVAILABLE,
        resource_domain="accelerator-memory",
        estimated_required_bytes=150_000_000,
        artifact_sha256="e" * 64,
        input_modalities=("structured-data",),
    )
    revision_ingress = ingress(
        "quarantined-extractor-revision",
        "Extract the verified records and summarize the findings.",
        (failed_slot, normal_slots[1], alternative_slot),
        assertions=(
            *facts,
            LedgerAssertion(
                assertion_id="extractor-failure",
                content_summary="data.extract failed validation and is quarantined.",
                provenance_sha256="f" * 64,
                last_observed_state="SPECIALIST_FAILED",
            ),
        ),
    )

    return StaceyCoreBenchmarkSuite(
        suite_id="stacey-task-graph-context-revision-fixture-v1",
        split=StaceyBenchmarkSplit.DEVELOPMENT,
        cases=(
            StaceyCoreBenchmarkCase(
                case_id="extract-before-summarize",
                ingress=normal_ingress,
                expected_clarification=False,
                required_capabilities=("data.extract", "text.summarize"),
                forbidden_capabilities=(),
                required_dependencies=(("data.extract", "text.summarize"),),
                consequence_concepts=("extract records", "summarize findings"),
                resource_expectations=(
                    ResourceEstimateExpectation("data.extract", "accelerator-memory", 100_000_000, 140_000_000),
                    ResourceEstimateExpectation("text.summarize", "accelerator-memory", 60_000_000, 100_000_000),
                ),
            ),
            StaceyCoreBenchmarkCase(
                case_id="ambiguous-share-asks-first",
                ingress=ambiguous_ingress,
                expected_clarification=True,
                required_capabilities=(),
                forbidden_capabilities=("data.extract", "text.summarize"),
                required_dependencies=(),
                consequence_concepts=("recipient", "content"),
                resource_expectations=(),
            ),
            StaceyCoreBenchmarkCase(
                case_id="revise-after-extractor-failure",
                ingress=revision_ingress,
                expected_clarification=False,
                required_capabilities=("data.extract.alternative", "text.summarize"),
                forbidden_capabilities=("data.extract",),
                required_dependencies=(("data.extract.alternative", "text.summarize"),),
                consequence_concepts=("failed extractor", "alternative"),
                resource_expectations=(
                    ResourceEstimateExpectation(
                        "data.extract.alternative", "accelerator-memory", 130_000_000, 170_000_000
                    ),
                    ResourceEstimateExpectation("text.summarize", "accelerator-memory", 60_000_000, 100_000_000),
                ),
            ),
        ),
    )


def build_stacey_task_graph_holdout_fixture_suite() -> StaceyCoreBenchmarkSuite:
    """Visible rubric holdout fixtures; not protected or human-reviewed evaluation data."""
    scope = ScopeVector(
        "tenant-benchmark-holdout",
        "user-benchmark-holdout",
        "project-benchmark-holdout",
        "workspace-benchmark-holdout",
    )
    observed_at_ns = 1_800_000_000_000_000_001

    def make_ingress(
        case_id: str,
        intent: str,
        slots: tuple[SpecialistSlot, ...],
        assertion: LedgerAssertion,
    ) -> UnifiedContextIngress:
        return UnifiedContextIngress(
            protocol_version="1.0",
            transaction_id=f"tx-holdout-{case_id}",
            correlation_id=f"turn-holdout-{case_id}",
            ingress_timestamp_ns=observed_at_ns,
            intent_vector=IntentVector(IntentKind.USER_LANGUAGE, intent),
            scope_vector=scope,
            canonical_state_assertions=(assertion,),
            hardware_capability_matrix=HardwareCapabilityMatrix(
                resources=(ResourceMeasurement("accelerator-memory", 2_000_000_000, observed_at_ns),),
                active_leases=(ActiveResourceLease("lease-holdout", "accelerator-memory", 90_000_000),),
                available_specialist_slots=slots,
            ),
        )

    inventory_slots = (
        SpecialistSlot(
            block_id="BLOCK_10_STRUCTURAL_INGRESS",
            capability_id="data.extract",
            status=SpecialistAvailability.AVAILABLE,
            resource_domain="accelerator-memory",
            estimated_required_bytes=110_000_000,
            artifact_sha256="1" * 64,
            input_modalities=("structured-data",),
        ),
        SpecialistSlot(
            block_id="BLOCK_9_ALGORITHMIC_CODER",
            capability_id="stats.aggregate",
            status=SpecialistAvailability.AVAILABLE,
            resource_domain="accelerator-memory",
            estimated_required_bytes=50_000_000,
            artifact_sha256="2" * 64,
            input_modalities=("structured-data",),
        ),
        SpecialistSlot(
            block_id="BLOCK_12_LINGUISTIC_COPY",
            capability_id="text.summarize",
            status=SpecialistAvailability.AVAILABLE,
            resource_domain="accelerator-memory",
            estimated_required_bytes=80_000_000,
            artifact_sha256="3" * 64,
            input_modalities=("text",),
        ),
    )
    inventory_ingress = make_ingress(
        "inventory-three-stage",
        "Count verified inventory by category and summarize totals.",
        inventory_slots,
        LedgerAssertion(
            "inventory-is-verified",
            "The inventory records are verified and grouped by item.",
            "4" * 64,
            "VERIFIED",
        ),
    )

    archive_ingress = make_ingress(
        "archive-ambiguity",
        "Archive the latest one in the usual place.",
        inventory_slots,
        LedgerAssertion(
            "archive-target-unknown",
            "The intended item version and destination are not identified.",
            "5" * 64,
            "MISSING_CONTEXT",
        ),
    )

    failed_summarizer = SpecialistSlot(
        block_id="BLOCK_12_LINGUISTIC_COPY",
        capability_id="text.summarize",
        status=SpecialistAvailability.QUARANTINED,
        resource_domain="accelerator-memory",
        estimated_required_bytes=80_000_000,
        artifact_sha256="3" * 64,
        input_modalities=("text",),
    )
    alternate_summarizer = SpecialistSlot(
        block_id="BLOCK_12_LINGUISTIC_COPY",
        capability_id="text.summarize.alternative",
        status=SpecialistAvailability.AVAILABLE,
        resource_domain="accelerator-memory",
        estimated_required_bytes=100_000_000,
        artifact_sha256="6" * 64,
        input_modalities=("text",),
    )
    summary_revision_ingress = make_ingress(
        "summary-specialist-revision",
        "Summarize the verified findings.",
        (failed_summarizer, alternate_summarizer),
        LedgerAssertion(
            "summary-specialist-failed",
            "text.summarize failed review and is quarantined; a reviewed alternative is available.",
            "7" * 64,
            "SPECIALIST_FAILED",
        ),
    )

    return StaceyCoreBenchmarkSuite(
        suite_id="stacey-task-graph-context-revision-holdout-fixture-v1",
        split=StaceyBenchmarkSplit.HOLDOUT_FIXTURE,
        cases=(
            StaceyCoreBenchmarkCase(
                case_id="inventory-three-stage-dependency",
                ingress=inventory_ingress,
                expected_clarification=False,
                required_capabilities=("data.extract", "stats.aggregate", "text.summarize"),
                forbidden_capabilities=(),
                required_dependencies=(
                    ("data.extract", "stats.aggregate"),
                    ("stats.aggregate", "text.summarize"),
                ),
                consequence_concepts=(
                    "extract inventory categories",
                    "calculate totals",
                    "summarize inventory",
                ),
                resource_expectations=(
                    ResourceEstimateExpectation("data.extract", "accelerator-memory", 90_000_000, 130_000_000),
                    ResourceEstimateExpectation("stats.aggregate", "accelerator-memory", 40_000_000, 70_000_000),
                    ResourceEstimateExpectation("text.summarize", "accelerator-memory", 60_000_000, 100_000_000),
                ),
            ),
            StaceyCoreBenchmarkCase(
                case_id="archive-ambiguity-asks-first",
                ingress=archive_ingress,
                expected_clarification=True,
                required_capabilities=(),
                forbidden_capabilities=("data.extract", "stats.aggregate", "text.summarize"),
                required_dependencies=(),
                consequence_concepts=("which item", "destination"),
                resource_expectations=(),
            ),
            StaceyCoreBenchmarkCase(
                case_id="revise-after-summary-specialist-failure",
                ingress=summary_revision_ingress,
                expected_clarification=False,
                required_capabilities=("text.summarize.alternative",),
                forbidden_capabilities=("text.summarize",),
                required_dependencies=(),
                consequence_concepts=("previous summary specialist failed", "alternate summarizer"),
                resource_expectations=(
                    ResourceEstimateExpectation(
                        "text.summarize.alternative", "accelerator-memory", 80_000_000, 120_000_000
                    ),
                ),
            ),
        ),
    )