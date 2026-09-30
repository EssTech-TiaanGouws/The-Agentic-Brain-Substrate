from ...contracts import Intent, TaskPlan, WorkerResult
from ...errors import IntegrityViolation


class EpistemicAuditor:
    def audit(self, intent: Intent, plan: TaskPlan, result: WorkerResult) -> None:
        if plan.transaction_id != intent.transaction_id:
            raise IntegrityViolation("Task plan is not bound to the inbound transaction")
        if result.accepted_goal != intent.goal:
            raise IntegrityViolation("Worker result cannot be traced to the inbound goal")