from ...contracts import Intent, IntentProfile, PlanStep, TaskPlan


class TacticalDecomposer:
    def decompose(self, intent: Intent, profile: IntentProfile) -> TaskPlan:
        return TaskPlan(
            transaction_id=intent.transaction_id,
            category=profile.category,
            steps=(PlanStep(step_id="step-1", capability_key="mock.general"),),
        )