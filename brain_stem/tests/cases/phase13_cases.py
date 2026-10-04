from __future__ import annotations

import unittest
from dataclasses import replace
import tempfile
from pathlib import Path
from time import time

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from src.swarm_core.training_governance import (
    Ed25519GovernanceApprovalVerifier,
    GovernanceApprovalError,
    GovernanceApprovalJournal,
    GovernanceApprovalReplayError,
    SignedGovernanceApproval,
    governance_approval_message,
)


class SignedGovernanceApprovalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.private_key = Ed25519PrivateKey.generate()
        public_key = self.private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        self.verifier = Ed25519GovernanceApprovalVerifier(
            {("reviewer-1", "DATA_REVIEWER"): public_key},
            allowed_roles_by_action={"DATASET_APPROVE": ("DATA_REVIEWER",)},
            audience="deployment-a",
        )

    def make_approval(self, *, subject: str = "a" * 64, issued_at: int | None = None) -> SignedGovernanceApproval:
        now = int(time()) if issued_at is None else issued_at
        unsigned = SignedGovernanceApproval(
            approval_id="approval-1",
            reviewer_id="reviewer-1",
            reviewer_role="DATA_REVIEWER",
            action="DATASET_APPROVE",
            subject_sha256=subject,
            audience="deployment-a",
            issued_at=now,
            expires_at=now + 60,
            signature=b"",
        )
        return replace(unsigned, signature=self.private_key.sign(governance_approval_message(unsigned)))

    def test_valid_approval_is_bound_to_exact_action_subject_audience_and_role(self) -> None:
        approval = self.make_approval()
        self.assertEqual(
            self.verifier.verify(
                approval,
                action="DATASET_APPROVE",
                subject_sha256="a" * 64,
            ),
            "approval-1",
        )

        for action, digest in (
            ("TRAINING_RUN_APPROVE", "a" * 64),
            ("DATASET_APPROVE", "b" * 64),
        ):
            with self.assertRaises(GovernanceApprovalError):
                self.verifier.verify(approval, action=action, subject_sha256=digest)

    def test_expired_and_future_approvals_are_rejected(self) -> None:
        now = int(time())
        expired_unsigned = replace(self.make_approval(), issued_at=now - 120, expires_at=now - 60, signature=b"")
        expired = replace(expired_unsigned, signature=self.private_key.sign(governance_approval_message(expired_unsigned)))
        with self.assertRaises(GovernanceApprovalError):
            self.verifier.verify(expired, action="DATASET_APPROVE", subject_sha256="a" * 64)

        future_unsigned = replace(self.make_approval(), issued_at=now + 60, expires_at=now + 120, signature=b"")
        future = replace(future_unsigned, signature=self.private_key.sign(governance_approval_message(future_unsigned)))
        with self.assertRaises(GovernanceApprovalError):
            self.verifier.verify(future, action="DATASET_APPROVE", subject_sha256="a" * 64)

    def test_signature_tampering_unknown_reviewer_and_wrong_role_are_rejected(self) -> None:
        approval = self.make_approval()
        tampered = replace(approval, subject_sha256="b" * 64)
        with self.assertRaises(GovernanceApprovalError):
            self.verifier.verify(tampered, action="DATASET_APPROVE", subject_sha256="b" * 64)

        unknown = replace(approval, reviewer_id="unknown-reviewer")
        with self.assertRaises(GovernanceApprovalError):
            self.verifier.verify(unknown, action="DATASET_APPROVE", subject_sha256="a" * 64)

        wrong_role = replace(approval, reviewer_role="RUN_AUTHORIZER")
        with self.assertRaises(GovernanceApprovalError):
            self.verifier.verify(wrong_role, action="DATASET_APPROVE", subject_sha256="a" * 64)

    def test_consumed_run_approval_cannot_replay_after_journal_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "approvals.sqlite3"
            approval = self.make_approval()
            with GovernanceApprovalJournal(path, verifier=self.verifier) as journal:
                self.assertEqual(
                    journal.consume(
                        approval,
                        action="DATASET_APPROVE",
                        subject_sha256="a" * 64,
                    ),
                    approval.approval_id,
                )

            with GovernanceApprovalJournal(path, verifier=self.verifier) as reopened:
                with self.assertRaises(GovernanceApprovalReplayError):
                    reopened.consume(
                        approval,
                        action="DATASET_APPROVE",
                        subject_sha256="a" * 64,
                    )
                self.assertEqual(
                    reopened.confirm_consumed(
                        approval,
                        action="DATASET_APPROVE",
                        subject_sha256="a" * 64,
                    ),
                    approval.approval_id,
                )
                with self.assertRaises(GovernanceApprovalError):
                    reopened.confirm_consumed(
                        approval,
                        action="DATASET_APPROVE",
                        subject_sha256="b" * 64,
                    )


if __name__ == "__main__":
    unittest.main()
