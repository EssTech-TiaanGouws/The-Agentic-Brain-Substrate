from ...contracts import Intent, TaskPlan, WorkerResult
from ...errors import IntegrityViolation


class AdversarialVerifier:
    _ALLOWED_CAPABILITIES = {"mock.general"}

    def verify(self, intent: Intent, plan: TaskPlan, result: WorkerResult) -> None:
        step_ids = tuple(step.step_id for step in plan.steps)
        if not step_ids or len(step_ids) != len(set(step_ids)):
            raise IntegrityViolation("Plan has no steps or contains duplicate step IDs")
        if any(step.capability_key not in self._ALLOWED_CAPABILITIES for step in plan.steps):
            raise IntegrityViolation("Plan references an unregistered capability")
        if result.status != "MOCKED":
            raise IntegrityViolation("CPU prototype received a non-mock worker result")
        if result.transaction_id != intent.transaction_id:
            raise IntegrityViolation("Worker result transaction ID mismatch")
        if result.correlation_id != intent.correlation_id:
            raise IntegrityViolation("Worker result correlation ID mismatch")
        if set(result.completed_step_ids) != set(step_ids):
            raise IntegrityViolation("Worker result does not match the task plan")