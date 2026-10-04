from __future__ import annotations

import unittest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from substrate.contracts import Intent, ScopeVector
from src.swarm_core.identity import Ed25519ScopeAuthorizationVerifier
from src.swarm_core.operator_authority import (
    ConfirmedScopeGrantIssuer,
    OperatorAuthorityError,
    OperatorConfirmation,
)


SCOPE = ScopeVector("tenant-a", "user-a", "project-a", "workspace-a")


class FakeConfirmationProvider:
    def __init__(self, *, approved: bool = True, alter_digest: bool = False, fail: bool = False) -> None:
        self.approved = approved
        self.alter_digest = alter_digest
        self.fail = fail
        self.calls = []

    def confirm(self, intent, request_sha256):
        self.calls.append((intent, request_sha256))
        if self.fail:
            raise RuntimeError("confirmation UI unavailable")
        digest = "0" * 64 if self.alter_digest else request_sha256
        return OperatorConfirmation(self.approved, "confirmation-1", digest)


class FakeProtectedSigner:
    signer_reference = "os-key:operator-1"

    def __init__(self, *, fail: bool = False, invalid_signature: bool = False) -> None:
        self.private_key = Ed25519PrivateKey.generate()
        self.fail = fail
        self.invalid_signature = invalid_signature
        self.calls = 0

    @property
    def public_key(self) -> bytes:
        return self.private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    def sign(self, message: bytes) -> bytes:
        self.calls += 1
        if self.fail:
            raise RuntimeError("keystore unavailable")
        if self.invalid_signature:
            return b"invalid"
        return self.private_key.sign(message)


class ConfirmedScopeGrantIssuerTests(unittest.TestCase):
    def make_intent(self, *, action: str = "INSPECT_SYSTEM", goal: str = "Inspect viewport safely.") -> Intent:
        return Intent(
            transaction_id="tx-authority",
            correlation_id="turn-authority",
            action=action,
            goal=goal,
            scope=SCOPE,
        )

    def make_issuer(self, confirmation, signer):
        return ConfirmedScopeGrantIssuer(
            confirmation_provider=confirmation,
            signing_provider=signer,
            audience="authority-test",
            permitted_actions=("INSPECT_SYSTEM", "WRITE_WORKSPACE_FILE"),
            clock=lambda: 100,
            lifetime_seconds=60,
        )

    def test_issued_grant_verifies_for_exact_intent_and_scope(self) -> None:
        confirmation = FakeConfirmationProvider()
        signer = FakeProtectedSigner()
        grant = self.make_issuer(confirmation, signer).issue(self.make_intent())
        verifier = Ed25519ScopeAuthorizationVerifier(
            signer.public_key,
            audience="authority-test",
            clock=lambda: 100,
        )

        self.assertEqual(verifier.authorize(self.make_intent(), grant), SCOPE)
        with self.assertRaises(PermissionError):
            verifier.authorize(self.make_intent(goal="Inspect a different target."), grant)

    def test_denial_digest_mismatch_and_confirmation_failure_never_sign(self) -> None:
        for confirmation in (
            FakeConfirmationProvider(approved=False),
            FakeConfirmationProvider(alter_digest=True),
            FakeConfirmationProvider(fail=True),
        ):
            signer = FakeProtectedSigner()
            with self.subTest(confirmation=confirmation):
                with self.assertRaises(OperatorAuthorityError):
                    self.make_issuer(confirmation, signer).issue(self.make_intent())
                self.assertEqual(signer.calls, 0)

    def test_nonallowlisted_action_and_signer_failure_are_denied(self) -> None:
        confirmation = FakeConfirmationProvider()
        signer = FakeProtectedSigner()
        with self.assertRaisesRegex(OperatorAuthorityError, "not allowlisted"):
            self.make_issuer(confirmation, signer).issue(self.make_intent(action="RUN_SHELL"))
        self.assertEqual(signer.calls, 0)

        failing_signer = FakeProtectedSigner(fail=True)
        with self.assertRaisesRegex(OperatorAuthorityError, "signing provider failed"):
            self.make_issuer(FakeConfirmationProvider(), failing_signer).issue(self.make_intent())

        invalid_signer = FakeProtectedSigner(invalid_signature=True)
        with self.assertRaisesRegex(OperatorAuthorityError, "malformed signature"):
            self.make_issuer(FakeConfirmationProvider(), invalid_signer).issue(self.make_intent())


if __name__ == "__main__":
    unittest.main()