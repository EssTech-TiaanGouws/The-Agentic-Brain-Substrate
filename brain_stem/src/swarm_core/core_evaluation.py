from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Protocol

from substrate.contracts import ScopeVector

from .model_catalog import EvaluationEvidence


class CoreEvaluationError(ValueError):
    pass


_METRIC_NAMES = frozenset(
    {
        "case_pass_rate",
        "scope_integrity_rate",
        "intent_interpretation_rate",
        "clarification_accuracy_rate",
        "capability_coverage_rate",
        "task_graph_validity_rate",
    }
)


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CoreEvaluationError(f"{field_name} must be a non-empty string")
    return value


def _validate_unique_texts(values: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise CoreEvaluationError(f"{field_name} must be a tuple")
    for value in values:
        _required_text(value, field_name)
    if len(set(values)) != len(values):
        raise CoreEvaluationError(f"{field_name} must not contain duplicates")
    return values


@dataclass(frozen=True, slots=True)
class CoreRequest:
    request_id: str
    raw_intent: str
    trusted_scope: ScopeVector

    def __post_init__(self) -> None:
        _required_text(self.request_id, "request_id")
        _required_text(self.raw_intent, "raw_intent")
        if not isinstance(self.trusted_scope, ScopeVector) or not self.trusted_scope.is_complete():
            raise CoreEvaluationError("request requires a complete trusted four-field scope")


@dataclass(frozen=True, slots=True)
class CorePlanStep:
    step_id: str
    capability_id: str
    depends_on: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _required_text(self.step_id, "step_id")
        _required_text(self.capability_id, "capability_id")
        _validate_unique_texts(self.depends_on, "depends_on")


@dataclass(frozen=True, slots=True)
class CoreDecision:
    interpreted_intent: str
    scope: ScopeVector
    requires_clarification: bool
    plan: tuple[CorePlanStep, ...]


class WorldModelCoreCandidate(Protocol):
    candidate_id: str
    artifact_sha256: str

    def decide(self, request: CoreRequest) -> CoreDecision: ...


@dataclass(frozen=True, slots=True)
class CoreBenchmarkCase:
    case_id: str
    request: CoreRequest
    expected_clarification: bool
    required_capabilities: tuple[str, ...] = ()
    forbidden_capabilities: tuple[str, ...] = ()
    required_dependencies: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _required_text(self.case_id, "case_id")
        if not isinstance(self.request, CoreRequest):
            raise CoreEvaluationError("case request must be a CoreRequest")
        if not isinstance(self.expected_clarification, bool):
            raise CoreEvaluationError("expected_clarification must be a boolean")
        _validate_unique_texts(self.required_capabilities, "required_capabilities")
        _validate_unique_texts(self.forbidden_capabilities, "forbidden_capabilities")
        if set(self.required_capabilities) & set(self.forbidden_capabilities):
            raise CoreEvaluationError("a capability cannot be both required and forbidden")
        if not isinstance(self.required_dependencies, tuple):
            raise CoreEvaluationError("required_dependencies must be a tuple")
        for edge in self.required_dependencies:
            if not isinstance(edge, tuple) or len(edge) != 2:
                raise CoreEvaluationError("dependency requirements must be (upstream_step, downstream_step) pairs")
            _required_text(edge[0], "upstream_step")
            _required_text(edge[1], "downstream_step")


@dataclass(frozen=True, slots=True)
class CoreBenchmarkSuite:
    suite_id: str
    cases: tuple[CoreBenchmarkCase, ...]

    def __post_init__(self) -> None:
        _required_text(self.suite_id, "suite_id")
        if not isinstance(self.cases, tuple) or not self.cases:
            raise CoreEvaluationError("benchmark suite must contain at least one case")
        if any(not isinstance(case, CoreBenchmarkCase) for case in self.cases):
            raise CoreEvaluationError("benchmark suite cases must be CoreBenchmarkCase values")
        case_ids = tuple(case.case_id for case in self.cases)
        if len(set(case_ids)) != len(case_ids):
            raise CoreEvaluationError("benchmark case IDs must be unique")


@dataclass(frozen=True, slots=True)
class CoreAcceptancePolicy:
    minimum_case_pass_rate: float
    minimum_metric_rates: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        self._validate_rate(self.minimum_case_pass_rate, "minimum_case_pass_rate")
        if not isinstance(self.minimum_metric_rates, tuple):
            raise CoreEvaluationError("minimum_metric_rates must be a tuple")
        seen: set[str] = set()
        for entry in self.minimum_metric_rates:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise CoreEvaluationError("metric thresholds must be (metric_name, rate) pairs")
            name, value = entry
            _required_text(name, "metric_name")
            if name not in _METRIC_NAMES:
                raise CoreEvaluationError(f"unknown Core evaluation metric: {name}")
            if name in seen:
                raise CoreEvaluationError("metric thresholds must not contain duplicate names")
            self._validate_rate(value, name)
            seen.add(name)

    @staticmethod
    def _validate_rate(value: object, field_name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CoreEvaluationError(f"{field_name} must be numeric")
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise CoreEvaluationError(f"{field_name} must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class CoreCaseResult:
    case_id: str
    checks: tuple[tuple[str, bool], ...]
    error_type: str | None = None

    @property
    def passed(self) -> bool:
        return self.error_type is None and all(value for _, value in self.checks)


@dataclass(frozen=True, slots=True)
class CoreBenchmarkReport:
    candidate_id: str
    suite_id: str
    case_results: tuple[CoreCaseResult, ...]
    metric_rates: tuple[tuple[str, float], ...]
    accepted: bool
    evidence: EvaluationEvidence


class CoreBenchmarkRunner:
    """Runs structured Core tasks; thresholds and candidate execution are injected."""

    def run(
        self,
        *,
        candidate: WorldModelCoreCandidate,
        suite: CoreBenchmarkSuite,
        policy: CoreAcceptancePolicy,
        run_reference: str,
        baseline_reference: str,
    ) -> CoreBenchmarkReport:
        _required_text(getattr(candidate, "candidate_id", None), "candidate_id")
        artifact_digest = getattr(candidate, "artifact_sha256", None)
        if not isinstance(artifact_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", artifact_digest):
            raise CoreEvaluationError("candidate must expose a lowercase SHA-256 artifact digest")
        if not isinstance(suite, CoreBenchmarkSuite):
            raise CoreEvaluationError("suite must be a CoreBenchmarkSuite")
        if not isinstance(policy, CoreAcceptancePolicy):
            raise CoreEvaluationError("policy must be a CoreAcceptancePolicy")
        _required_text(run_reference, "run_reference")
        _required_text(baseline_reference, "baseline_reference")

        results = tuple(self._evaluate_case(candidate, case) for case in suite.cases)
        metric_rates = self._calculate_rates(results)
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
        return CoreBenchmarkReport(
            candidate_id=candidate.candidate_id,
            suite_id=suite.suite_id,
            case_results=results,
            metric_rates=metric_rates,
            accepted=accepted,
            evidence=evidence,
        )

    @staticmethod
    def _evaluate_case(candidate: WorldModelCoreCandidate, case: CoreBenchmarkCase) -> CoreCaseResult:
        try:
            decision = candidate.decide(case.request)
        except Exception as error:
            return CoreCaseResult(
                case_id=case.case_id,
                checks=tuple((name, False) for name in _METRIC_NAMES if name != "case_pass_rate"),
                error_type=type(error).__name__,
            )

        if not isinstance(decision, CoreDecision):
            return CoreCaseResult(
                case_id=case.case_id,
                checks=tuple((name, False) for name in _METRIC_NAMES if name != "case_pass_rate"),
                error_type="InvalidCoreDecision",
            )

        plan_valid, steps_by_id = CoreBenchmarkRunner._validate_plan(decision, case)
        capabilities = (
            {step.capability_id for step in decision.plan if isinstance(step, CorePlanStep)}
            if isinstance(decision.plan, tuple)
            else set()
        )
        interpreted_intent_present = (
            isinstance(decision.interpreted_intent, str)
            and bool(decision.interpreted_intent.strip())
        )
        checks = (
            ("scope_integrity_rate", decision.scope == case.request.trusted_scope),
            ("intent_interpretation_rate", interpreted_intent_present),
            (
                "clarification_accuracy_rate",
                isinstance(decision.requires_clarification, bool)
                and decision.requires_clarification == case.expected_clarification,
            ),
            (
                "capability_coverage_rate",
                set(case.required_capabilities) <= capabilities
                and not (set(case.forbidden_capabilities) & capabilities),
            ),
            ("task_graph_validity_rate", plan_valid),
        )
        return CoreCaseResult(case_id=case.case_id, checks=checks)

    @staticmethod
    def _validate_plan(
        decision: CoreDecision,
        case: CoreBenchmarkCase,
    ) -> tuple[bool, dict[str, CorePlanStep]]:
        if not isinstance(decision.requires_clarification, bool):
            return False, {}
        if not isinstance(decision.plan, tuple):
            return False, {}
        if decision.requires_clarification:
            return not decision.plan, {}
        if not decision.plan or any(not isinstance(step, CorePlanStep) for step in decision.plan):
            return False, {}

        steps_by_id = {step.step_id: step for step in decision.plan}
        if len(steps_by_id) != len(decision.plan):
            return False, {}
        for step in decision.plan:
            if any(dependency not in steps_by_id for dependency in step.depends_on):
                return False, {}

        visited: set[str] = set()
        visiting: set[str] = set()

        def has_cycle(step_id: str) -> bool:
            if step_id in visiting:
                return True
            if step_id in visited:
                return False
            visiting.add(step_id)
            if any(has_cycle(dependency) for dependency in steps_by_id[step_id].depends_on):
                return True
            visiting.remove(step_id)
            visited.add(step_id)
            return False

        if any(has_cycle(step_id) for step_id in steps_by_id):
            return False, {}
        if any(
            upstream not in steps_by_id
            or downstream not in steps_by_id
            or upstream not in steps_by_id[downstream].depends_on
            for upstream, downstream in case.required_dependencies
        ):
            return False, {}
        return True, steps_by_id

    @staticmethod
    def _calculate_rates(results: tuple[CoreCaseResult, ...]) -> tuple[tuple[str, float], ...]:
        denominator = len(results)
        if denominator == 0:
            raise CoreEvaluationError("cannot calculate metrics for an empty result set")
        rates = {
            metric_name: sum(
                1
                for result in results
                if result.passed
                if metric_name == "case_pass_rate"
            ) / denominator
            for metric_name in ("case_pass_rate",)
        }
        metric_names = tuple(name for name in next((result.checks for result in results), ()))
        for metric_name, _ in metric_names:
            rates[metric_name] = sum(
                1
                for result in results
                if dict(result.checks).get(metric_name, False)
            ) / denominator
        return tuple(sorted(rates.items()))