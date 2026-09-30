from ..contracts import Intent, TaskPlan, WorkerResult


class MockWorker:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, intent: Intent, plan: TaskPlan) -> WorkerResult:
        self.calls += 1
        return WorkerResult(
            status="MOCKED",
            transaction_id=intent.transaction_id,
            correlation_id=intent.correlation_id,
            accepted_goal=intent.goal,
            completed_step_ids=tuple(step.step_id for step in plan.steps),
        )