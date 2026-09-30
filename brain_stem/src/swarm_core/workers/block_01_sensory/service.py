from ...workers.contracts import MockWorkerRequest, MockWorkerResult, WorkerLifecycle


class SensoryOpticalMockWorker:
    block_id = 1
    capability_key = "sensory.inspect"
    lifecycle = WorkerLifecycle.TRANSIENT

    def execute(self, request: MockWorkerRequest) -> MockWorkerResult:
        if not isinstance(request, MockWorkerRequest):
            raise TypeError("request must be a MockWorkerRequest")
        return MockWorkerResult(
            block_id=self.block_id,
            capability_key=self.capability_key,
            lifecycle=self.lifecycle,
            status="MOCKED",
            input_digest=request.content_digest,
        )