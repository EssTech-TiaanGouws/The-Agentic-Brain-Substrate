from .blocks.block_02_semantic_classifier.service import SemanticClassifier
from .blocks.block_03_tactical_decomposer.service import TacticalDecomposer
from .blocks.block_06_adversarial_verifier.service import AdversarialVerifier
from .blocks.block_08_epistemic_auditor.service import EpistemicAuditor
from .contracts import ExecutionReport, Intent, ScopeVector
from .errors import IngressRejected
from .workers.mock_worker import MockWorker


class SubstrateKernel:
    """In-process control plane for the CPU-only routing prototype."""

    def __init__(
        self,
        worker: MockWorker | None = None,
        adversarial_verifier: AdversarialVerifier | None = None,
        epistemic_auditor: EpistemicAuditor | None = None,
    ) -> None:
        self._classifier = SemanticClassifier()
        self._decomposer = TacticalDecomposer()
        self._worker = worker or MockWorker()
        self._adversarial_verifier = adversarial_verifier or AdversarialVerifier()
        self._epistemic_auditor = epistemic_auditor or EpistemicAuditor()

    def route(self, intent: Intent, authorized_scope: ScopeVector) -> ExecutionReport:
        self._validate_ingress(intent, authorized_scope)

        profile = self._classifier.classify(intent)
        plan = self._decomposer.decompose(intent, profile)
        worker_result = self._worker.execute(intent, plan)

        self._adversarial_verifier.verify(intent, plan, worker_result)
        self._epistemic_auditor.audit(intent, plan, worker_result)

        return ExecutionReport(
            transaction_id=intent.transaction_id,
            correlation_id=intent.correlation_id,
            scope=intent.scope,
            profile=profile,
            plan=plan,
            worker_result=worker_result,
        )

    @staticmethod
    def _validate_ingress(intent: Intent, authorized_scope: ScopeVector) -> None:
        if not isinstance(intent, Intent):
            raise IngressRejected("Ingress payload must be a typed Intent")
        if not isinstance(intent.scope, ScopeVector) or not intent.scope.is_complete():
            raise IngressRejected("Ingress requires all four non-empty scope fields")
        if not isinstance(authorized_scope, ScopeVector) or not authorized_scope.is_complete():
            raise IngressRejected("Trusted authorization scope is incomplete")
        if intent.scope != authorized_scope:
            raise IngressRejected("Intent scope does not match authorized caller scope")
        identifiers_and_goal = (
            intent.transaction_id,
            intent.correlation_id,
            intent.goal,
        )
        if any(not isinstance(value, str) or not value.strip() for value in identifiers_and_goal):
            raise IngressRejected("Transaction ID, correlation ID, and goal must be non-empty strings")
        if intent.action != "ROUTE_TASK":
            raise IngressRejected("Unsupported declarative intent action")