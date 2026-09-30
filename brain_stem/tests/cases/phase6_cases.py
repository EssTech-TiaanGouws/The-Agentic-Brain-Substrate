from __future__ import annotations

import unittest

from substrate.contracts import ScopeVector
from swarm_core.core_evaluation import (
    CoreAcceptancePolicy,
    CoreBenchmarkCase,
    CoreBenchmarkRunner,
    CoreBenchmarkSuite,
    CoreDecision,
    CorePlanStep,
    CoreRequest,
)


SCOPE = ScopeVector("tenant-eval", "user-eval", "project-eval", "workspace-eval")
ARTIFACT_SHA256 = "c" * 64


class FixtureCore:
    candidate_id = "fixture-world-model-core"
    artifact_sha256 = ARTIFACT_SHA256

    def __init__(self, decisions: dict[str, CoreDecision | Exception]) -> None:
        self.decisions = decisions

    def decide(self, request: CoreRequest) -> CoreDecision:
        result = self.decisions[request.request_id]
        if isinstance(result, Exception):
            raise result
        return result


def make_request(request_id: str, raw_intent: str) -> CoreRequest:
    return CoreRequest(request_id, raw_intent, SCOPE)


def make_suite() -> CoreBenchmarkSuite:
    return CoreBenchmarkSuite(
        suite_id="core-planning-fixture-v1",
        cases=(
            CoreBenchmarkCase(
                case_id="simple-document-route",
                request=make_request("simple", "Summarize the selected document"),
                expected_clarification=False,
                required_capabilities=("document.inspect",),
                forbidden_capabilities=("external.send",),
            ),
            CoreBenchmarkCase(
                case_id="ambiguous-request-asks",
                request=make_request("ambiguous", "Send it to them"),
                expected_clarification=True,
            ),
            CoreBenchmarkCase(
                case_id="dependency-plan-orders-extraction",
                request=make_request("multi-step", "Extract and summarize the records"),
                expected_clarification=False,
                required_capabilities=("data.extract", "text.summarize"),
                required_dependencies=(("extract", "summarize"),),
            ),
        ),
    )


def make_good_core() -> FixtureCore:
    return FixtureCore(
        {
            "simple": CoreDecision(
                interpreted_intent="summarize selected document",
                scope=SCOPE,
                requires_clarification=False,
                plan=(CorePlanStep("summarize", "document.inspect"),),
            ),
            "ambiguous": CoreDecision(
                interpreted_intent="identify recipient and content before sending",
                scope=SCOPE,
                requires_clarification=True,
                plan=(),
            ),
            "multi-step": CoreDecision(
                interpreted_intent="extract records before summarization",
                scope=SCOPE,
                requires_clarification=False,
                plan=(
                    CorePlanStep("extract", "data.extract"),
                    CorePlanStep("summarize", "text.summarize", depends_on=("extract",)),
                ),
            ),
        }
    )


def strict_policy() -> CoreAcceptancePolicy:
    return CoreAcceptancePolicy(
        minimum_case_pass_rate=1.0,
        minimum_metric_rates=(
            ("scope_integrity_rate", 1.0),
            ("intent_interpretation_rate", 1.0),
            ("clarification_accuracy_rate", 1.0),
            ("capability_coverage_rate", 1.0),
            ("task_graph_validity_rate", 1.0),
        ),
    )


class CoreEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.suite = make_suite()
        self.runner = CoreBenchmarkRunner()

    def run_core(self, candidate: FixtureCore, policy: CoreAcceptancePolicy | None = None):
        return self.runner.run(
            candidate=candidate,
            suite=self.suite,
            policy=policy or strict_policy(),
            run_reference="fixture-run-001",
            baseline_reference="human-reviewed-baseline-001",
        )

    def test_core_passes_representative_routing_clarification_and_dependency_suite(self) -> None:
        report = self.run_core(make_good_core())

        self.assertTrue(report.accepted)
        self.assertTrue(report.evidence.passed)
        self.assertEqual(dict(report.metric_rates)["case_pass_rate"], 1.0)
        self.assertEqual(report.evidence.artifact_sha256, ARTIFACT_SHA256)

    def test_scope_mismatch_lowers_scope_and_case_scores(self) -> None:
        decisions = make_good_core().decisions.copy()
        wrong_scope = ScopeVector("tenant-other", "user-eval", "project-eval", "workspace-eval")
        decisions["simple"] = CoreDecision(
            interpreted_intent="summarize selected document",
            scope=wrong_scope,
            requires_clarification=False,
            plan=(CorePlanStep("summarize", "document.inspect"),),
        )

        report = self.run_core(FixtureCore(decisions))

        self.assertFalse(report.accepted)
        self.assertEqual(dict(report.metric_rates)["scope_integrity_rate"], 2 / 3)

    def test_cyclic_task_graph_fails_graph_validity(self) -> None:
        decisions = make_good_core().decisions.copy()
        decisions["multi-step"] = CoreDecision(
            interpreted_intent="extract and summarize",
            scope=SCOPE,
            requires_clarification=False,
            plan=(
                CorePlanStep("extract", "data.extract", depends_on=("summarize",)),
                CorePlanStep("summarize", "text.summarize", depends_on=("extract",)),
            ),
        )

        report = self.run_core(FixtureCore(decisions))

        self.assertFalse(report.accepted)
        self.assertLess(dict(report.metric_rates)["task_graph_validity_rate"], 1.0)

    def test_core_exception_is_recorded_as_failed_case_not_success(self) -> None:
        decisions = make_good_core().decisions.copy()
        decisions["ambiguous"] = RuntimeError("candidate inference failed")

        report = self.run_core(FixtureCore(decisions))

        self.assertFalse(report.accepted)
        ambiguous = next(result for result in report.case_results if result.case_id == "ambiguous-request-asks")
        self.assertEqual(ambiguous.error_type, "RuntimeError")
        self.assertFalse(ambiguous.passed)

    def test_malformed_core_output_counts_as_failure_without_crashing_evaluator(self) -> None:
        decisions = make_good_core().decisions.copy()
        decisions["simple"] = CoreDecision(
            interpreted_intent=None,
            scope=SCOPE,
            requires_clarification=1,
            plan=None,
        )

        report = self.run_core(FixtureCore(decisions))

        self.assertFalse(report.accepted)
        self.assertEqual(dict(report.metric_rates)["intent_interpretation_rate"], 2 / 3)
        self.assertEqual(dict(report.metric_rates)["clarification_accuracy_rate"], 2 / 3)
        self.assertEqual(dict(report.metric_rates)["task_graph_validity_rate"], 2 / 3)

    def test_acceptance_thresholds_are_injected_not_hardcoded(self) -> None:
        decisions = make_good_core().decisions.copy()
        decisions["simple"] = CoreDecision(
            interpreted_intent="summarize selected document",
            scope=SCOPE,
            requires_clarification=False,
            plan=(CorePlanStep("summarize", "unregistered.capability"),),
        )
        permissive_policy = CoreAcceptancePolicy(minimum_case_pass_rate=0.0)

        report = self.run_core(FixtureCore(decisions), permissive_policy)

        self.assertTrue(report.accepted)
        self.assertLess(dict(report.metric_rates)["case_pass_rate"], 1.0)

    def test_invalid_thresholds_and_unknown_metrics_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CoreAcceptancePolicy(minimum_case_pass_rate=1.2)
        with self.assertRaises(ValueError):
            CoreAcceptancePolicy(
                minimum_case_pass_rate=0.5,
                minimum_metric_rates=(("model_smartness", 0.5),),
            )

    def test_ambiguous_cases_require_clarification_without_execution_plan(self) -> None:
        decisions = make_good_core().decisions.copy()
        decisions["ambiguous"] = CoreDecision(
            interpreted_intent="send something to someone",
            scope=SCOPE,
            requires_clarification=False,
            plan=(CorePlanStep("send", "external.send"),),
        )

        report = self.run_core(FixtureCore(decisions))

        self.assertFalse(report.accepted)
        self.assertEqual(dict(report.metric_rates)["clarification_accuracy_rate"], 2 / 3)
        self.assertEqual(dict(report.metric_rates)["capability_coverage_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()