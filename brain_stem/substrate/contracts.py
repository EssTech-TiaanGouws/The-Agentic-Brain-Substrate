from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ScopeVector:
    tenant_id: str
    user_id: str
    project_id: str
    workspace_id: str

    def is_complete(self) -> bool:
        return all(
            isinstance(value, str) and bool(value.strip())
            for value in (
                self.tenant_id,
                self.user_id,
                self.project_id,
                self.workspace_id,
            )
        )


@dataclass(frozen=True, slots=True)
class Intent:
    transaction_id: str
    correlation_id: str
    action: str
    goal: str
    scope: ScopeVector


@dataclass(frozen=True, slots=True)
class IntentProfile:
    category: str


@dataclass(frozen=True, slots=True)
class PlanStep:
    step_id: str
    capability_key: str
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TaskPlan:
    transaction_id: str
    category: str
    steps: tuple[PlanStep, ...]


@dataclass(frozen=True, slots=True)
class WorkerResult:
    status: str
    transaction_id: str
    correlation_id: str
    accepted_goal: str
    completed_step_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExecutionReport:
    transaction_id: str
    correlation_id: str
    scope: ScopeVector
    profile: IntentProfile
    plan: TaskPlan
    worker_result: WorkerResult