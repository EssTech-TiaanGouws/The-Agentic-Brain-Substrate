import unittest
from dataclasses import replace

from substrate import IngressRejected, IntegrityViolation, SubstrateKernel
from substrate.blocks.block_06_adversarial_verifier import AdversarialVerifier
from substrate.blocks.block_08_epistemic_auditor import EpistemicAuditor
from substrate.contracts import Intent, ScopeVector
from substrate.workers import MockWorker


SCOPE = ScopeVector(
    tenant_id="tenant-a",
    user_id="user-a",
    project_id="project-a",
    workspace_id="workspace-a",
)


def make_intent(**changes: object) -> Intent:
    values: dict[str, object] = {
        "transaction_id": "tx-1",
        "correlation_id": "turn-1",
        "action": "ROUTE_TASK",
        "goal": "Explain the routing flow",
        "scope": SCOPE,
    }
    values.update(changes)
    return Intent(**values)  # type: ignore[arg-type]


class RecordingVerifier(AdversarialVerifier):
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def verify(self, intent, plan, result) -> None:
        self.events.append("block-6")
        super().verify(intent, plan, result)


class RecordingAuditor(EpistemicAuditor):
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def audit(self, intent, plan, result) -> None:
        self.events.append("block-8")
        super().audit(intent, plan, result)


class TamperedWorker(MockWorker):
    def execute(self, intent, plan):
        result = super().execute(intent, plan)
        return replace(result, accepted_goal="unrelated goal")


class KernelTests(unittest.TestCase):
    def test_routes_with_cpu_mock_and_four_field_scope(self) -> None:
        report = SubstrateKernel().route(make_intent(), SCOPE)

        self.assertEqual(report.profile.category, "analysis")
        self.assertEqual(report.worker_result.status, "MOCKED")
        self.assertEqual(report.worker_result.accepted_goal, "Explain the routing flow")
        self.assertEqual(report.plan.steps[0].capability_key, "mock.general")

    def test_scope_mismatch_rejects_before_worker_execution(self) -> None:
        worker = MockWorker()
        other_scope = replace(SCOPE, project_id="project-b")

        with self.assertRaises(IngressRejected):
            SubstrateKernel(worker=worker).route(make_intent(), other_scope)

        self.assertEqual(worker.calls, 0)

    def test_incomplete_scope_is_rejected(self) -> None:
        incomplete_scope = replace(SCOPE, workspace_id=" ")

        with self.assertRaises(IngressRejected):
            SubstrateKernel().route(make_intent(scope=incomplete_scope), SCOPE)

    def test_only_typed_route_intent_is_accepted(self) -> None:
        worker = MockWorker()

        with self.assertRaises(IngressRejected):
            SubstrateKernel(worker=worker).route(make_intent(action="RUN_SHELL"), SCOPE)

        self.assertEqual(worker.calls, 0)

    def test_malformed_runtime_fields_are_rejected(self) -> None:
        with self.assertRaises(IngressRejected):
            SubstrateKernel().route(make_intent(transaction_id=42), SCOPE)

    def test_critic_runs_before_epistemic_audit(self) -> None:
        events: list[str] = []
        kernel = SubstrateKernel(
            adversarial_verifier=RecordingVerifier(events),
            epistemic_auditor=RecordingAuditor(events),
        )

        kernel.route(make_intent(), SCOPE)

        self.assertEqual(events, ["block-6", "block-8"])

    def test_epistemic_audit_rejects_untraceable_worker_result(self) -> None:
        with self.assertRaises(IntegrityViolation):
            SubstrateKernel(worker=TamperedWorker()).route(make_intent(), SCOPE)


if __name__ == "__main__":
    unittest.main()