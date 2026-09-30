from __future__ import annotations

import unittest

from src.swarm_core.model_catalog import (
    CandidateStatus,
    CandidateTransitionError,
    EvaluationEvidence,
    ModelCandidate,
    ModelCandidateCatalog,
    ModelCatalogError,
    ModelRole,
    TrainingDataAuthorization,
    TrainingSource,
)


ARTIFACT_DIGEST = "a" * 64


def make_candidate(
    candidate_id: str = "core-candidate-a",
    *,
    role: ModelRole = ModelRole.WORLD_MODEL_CORE,
    capabilities: tuple[str, ...] = (
        "core.intent_understanding",
        "core.reasoning_planning",
        "core.specialist_selection",
    ),
) -> ModelCandidate:
    return ModelCandidate(
        candidate_id=candidate_id,
        version="0.1.0",
        role=role,
        capability_ids=capabilities,
        artifact_sha256=ARTIFACT_DIGEST,
        artifact_size_bytes=420_000_000,
        parameter_count=700_000_000,
        training_sources=(
            TrainingSource(
                source_id="curated-reasoning-corpus-v1",
                authorization=TrainingDataAuthorization.APPROVED_DATASET,
                authorization_reference="dataset-approval-ref",
            ),
        ),
    )


def make_evidence(
    *,
    passed: bool = True,
    artifact_sha256: str = ARTIFACT_DIGEST,
) -> EvaluationEvidence:
    return EvaluationEvidence(
        suite_id="core-reasoning-suite-v1",
        run_reference="evaluation-run-ref",
        artifact_sha256=artifact_sha256,
        baseline_reference="baseline-run-ref",
        passed=passed,
        metrics=(
            ("plan_validity", 0.92),
            ("scope_preservation", 1.0),
        ),
    )


class ModelCandidateCatalogTests(unittest.TestCase):
    def test_empty_catalog_does_not_discover_installed_models(self) -> None:
        catalog = ModelCandidateCatalog()

        self.assertEqual(catalog.approved_for("core.reasoning_planning"), ())

    def test_candidate_requires_an_explicit_authorized_training_source(self) -> None:
        candidate = make_candidate()
        catalog = ModelCandidateCatalog()

        with self.assertRaises(ModelCatalogError):
            unauthorized = TrainingSource(
                source_id="unapproved-source",
                authorization="IMPLICIT_LOCAL_DATA",
                authorization_reference="missing-consent",
            )
            catalog.propose(
                ModelCandidate(
                    candidate_id="unauthorized-candidate",
                    version=candidate.version,
                    role=candidate.role,
                    capability_ids=candidate.capability_ids,
                    artifact_sha256=candidate.artifact_sha256,
                    artifact_size_bytes=candidate.artifact_size_bytes,
                    parameter_count=candidate.parameter_count,
                    training_sources=(unauthorized,),
                )
            )

    def test_proposed_candidate_is_not_routable_before_eval_and_approval(self) -> None:
        catalog = ModelCandidateCatalog()
        proposed = catalog.propose(make_candidate())

        self.assertEqual(proposed.status, CandidateStatus.PROPOSED)
        self.assertEqual(catalog.approved_for("core.reasoning_planning"), ())

        evaluated = catalog.record_evaluation(proposed.candidate_id, make_evidence())
        self.assertEqual(evaluated.status, CandidateStatus.EVALUATED)
        self.assertEqual(catalog.approved_for("core.reasoning_planning"), ())

        approved = catalog.approve(proposed.candidate_id, "review-approval-ref")
        self.assertEqual(approved.status, CandidateStatus.APPROVED)
        self.assertEqual(catalog.approved_for("core.reasoning_planning"), (approved,))

    def test_failed_candidate_evaluation_cannot_be_approved(self) -> None:
        catalog = ModelCandidateCatalog()
        candidate = catalog.propose(make_candidate())
        rejected = catalog.record_evaluation(candidate.candidate_id, make_evidence(passed=False))

        self.assertEqual(rejected.status, CandidateStatus.REJECTED)
        with self.assertRaises(CandidateTransitionError):
            catalog.approve(candidate.candidate_id, "review-approval-ref")
        self.assertEqual(catalog.approved_for("core.reasoning_planning"), ())

    def test_evaluation_must_bind_to_exact_candidate_artifact_digest(self) -> None:
        catalog = ModelCandidateCatalog()
        candidate = catalog.propose(make_candidate())

        with self.assertRaises(ModelCatalogError):
            catalog.record_evaluation(candidate.candidate_id, make_evidence(artifact_sha256="b" * 64))

        self.assertEqual(catalog.get(candidate.candidate_id).status, CandidateStatus.PROPOSED)

    def test_duplicate_candidate_id_is_rejected(self) -> None:
        catalog = ModelCandidateCatalog()
        catalog.propose(make_candidate())

        with self.assertRaises(ModelCatalogError):
            catalog.propose(make_candidate())

    def test_capability_lookup_returns_all_approved_candidates_without_auto_selection(self) -> None:
        catalog = ModelCandidateCatalog()
        approved_candidates = []
        for candidate_id in ("core-candidate-a", "core-candidate-b"):
            candidate = catalog.propose(make_candidate(candidate_id))
            catalog.record_evaluation(candidate_id, make_evidence())
            approved_candidates.append(catalog.approve(candidate_id, f"approval:{candidate_id}"))

        self.assertEqual(
            catalog.approved_for("core.reasoning_planning"),
            tuple(approved_candidates),
        )

    def test_core_and_specialist_roles_share_the_same_capability_contract(self) -> None:
        core = make_candidate()
        specialist = make_candidate(
            "critic-candidate-a",
            role=ModelRole.SPECIALIST,
            capabilities=("evaluation.critique",),
        )

        self.assertEqual(core.role, ModelRole.WORLD_MODEL_CORE)
        self.assertEqual(specialist.role, ModelRole.SPECIALIST)
        self.assertEqual(specialist.capability_ids, ("evaluation.critique",))


if __name__ == "__main__":
    unittest.main()