import unittest
from dataclasses import replace
from time import time

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from substrate import IngressRejected, IntegrityViolation, SubstrateKernel
from substrate.blocks.block_06_adversarial_verifier import AdversarialVerifier
from substrate.blocks.block_08_epistemic_auditor import EpistemicAuditor
from substrate.contracts import Intent, ScopeVector
from substrate.workers import MockWorker
from src.swarm_core.identity import (
    AuthorizationError,
    Ed25519ScopeAuthorizationVerifier,
    SignedScopeGrant,
    intent_authorization_digest,
    scope_grant_message,
)


SCOPE = ScopeVector(
    tenant_id="tenant-a",
    user_id="user-a",
    project_id="project-a",
    workspace_id="workspace-a",
)
PRIVATE_KEY = Ed25519PrivateKey.generate()
PUBLIC_KEY = PRIVATE_KEY.public_key().public_bytes(
    encoding=serialization.Encoding.Raw,
    format=serialization.PublicFormat.Raw,
)
AUTHORIZATION_VERIFIER = Ed25519ScopeAuthorizationVerifier(
    PUBLIC_KEY,
    audience="brain-stem-test",
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


def make_grant(intent: Intent, scope: ScopeVector | None = None) -> SignedScopeGrant:
    now = int(time())
    unsigned = SignedScopeGrant(
        scope=scope or intent.scope,
        action=intent.action,
        transaction_id=intent.transaction_id,
        correlation_id=intent.correlation_id,
        audience="brain-stem-test",
        issued_at=now,
        expires_at=now + 60,
        grant_id=f"grant-{intent.transaction_id}",
        request_sha256=intent_authorization_digest(intent),
        signature=b"",
    )
    return replace(unsigned, signature=PRIVATE_KEY.sign(scope_grant_message(unsigned)))


def make_kernel(**kwargs: object) -> SubstrateKernel:
    return SubstrateKernel(authorization_verifier=AUTHORIZATION_VERIFIER, **kwargs)


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
        intent = make_intent()
        report = make_kernel().route(intent, make_grant(intent))

        self.assertEqual(report.profile.category, "analysis")
        self.assertEqual(report.worker_result.status, "MOCKED")
        self.assertEqual(report.worker_result.accepted_goal, "Explain the routing flow")
        self.assertEqual(report.plan.steps[0].capability_key, "mock.general")

    def test_scope_mismatch_rejects_before_worker_execution(self) -> None:
        worker = MockWorker()
        other_scope = replace(SCOPE, project_id="project-b")
        intent = make_intent()

        with self.assertRaises(IngressRejected):
            make_kernel(worker=worker).route(intent, make_grant(intent, other_scope))

        self.assertEqual(worker.calls, 0)

    def test_incomplete_scope_is_rejected(self) -> None:
        incomplete_scope = replace(SCOPE, workspace_id=" ")
        intent = make_intent(scope=incomplete_scope)

        with self.assertRaises(IngressRejected):
            make_kernel().route(intent, make_grant(intent, incomplete_scope))

    def test_only_typed_route_intent_is_accepted(self) -> None:
        worker = MockWorker()
        intent = make_intent(action="RUN_SHELL")

        with self.assertRaises(IngressRejected):
            make_kernel(worker=worker).route(intent, make_grant(intent))

        self.assertEqual(worker.calls, 0)

    def test_malformed_runtime_fields_are_rejected(self) -> None:
        with self.assertRaises(IngressRejected):
            make_kernel().route(make_intent(transaction_id=42), make_grant(make_intent(transaction_id=42)))

    def test_critic_runs_before_epistemic_audit(self) -> None:
        events: list[str] = []
        kernel = make_kernel(
            adversarial_verifier=RecordingVerifier(events),
            epistemic_auditor=RecordingAuditor(events),
        )
        intent = make_intent()

        kernel.route(intent, make_grant(intent))

        self.assertEqual(events, ["block-6", "block-8"])

    def test_epistemic_audit_rejects_untraceable_worker_result(self) -> None:
        intent = make_intent()
        with self.assertRaises(IntegrityViolation):
            make_kernel(worker=TamperedWorker()).route(intent, make_grant(intent))

    def test_unsigned_or_expired_scope_grant_is_rejected(self) -> None:
        intent = make_intent()
        grant = make_grant(intent)
        tampered = replace(grant, scope=replace(SCOPE, project_id="project-b"))
        with self.assertRaises(IngressRejected):
            make_kernel().route(intent, tampered)

        expired = replace(grant, issued_at=int(time()) - 120, expires_at=int(time()) - 60)
        expired = replace(expired, signature=PRIVATE_KEY.sign(scope_grant_message(expired)))
        with self.assertRaises(IngressRejected):
            make_kernel().route(intent, expired)

    def test_kernel_requires_an_authorization_verifier(self) -> None:
        with self.assertRaises(ValueError):
            SubstrateKernel()

    def test_grant_cannot_be_replayed_for_another_transaction(self) -> None:
        intent = make_intent()
        other_intent = make_intent(transaction_id="tx-2")
        with self.assertRaises(AuthorizationError):
            AUTHORIZATION_VERIFIER.authorize(other_intent, make_grant(intent))

    def test_grant_cannot_be_reused_after_goal_text_changes(self) -> None:
        intent = make_intent()
        altered_intent = make_intent(goal="Perform an unrelated action")
        with self.assertRaises(AuthorizationError):
            AUTHORIZATION_VERIFIER.authorize(altered_intent, make_grant(intent))


if __name__ == "__main__":
    unittest.main()