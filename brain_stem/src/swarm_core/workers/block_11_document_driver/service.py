from ...format_adapters import DocumentFormatRegistry, DocumentPayload
from ...workers.contracts import MockWorkerRequest, MockWorkerResult, WorkerLifecycle


class DeclarativeDocumentDriverMockWorker:
    block_id = 11
    capability_key = "document.stage"
    lifecycle = WorkerLifecycle.TRANSIENT

    def __init__(self, formats: DocumentFormatRegistry) -> None:
        if not isinstance(formats, DocumentFormatRegistry):
            raise TypeError("formats must be a DocumentFormatRegistry")
        self._formats = formats

    def execute(self, request: MockWorkerRequest) -> MockWorkerResult:
        if not isinstance(request, MockWorkerRequest):
            raise TypeError("request must be a MockWorkerRequest")
        if request.format_key is None:
            raise ValueError("Block 11 requires an explicit document format key")

        prepared = self._formats.prepare(
            DocumentPayload(format_key=request.format_key, content=request.content)
        )
        return MockWorkerResult(
            block_id=self.block_id,
            capability_key=self.capability_key,
            lifecycle=self.lifecycle,
            status="STAGED_IN_MEMORY",
            input_digest=request.content_digest,
            output_digest=prepared.digest,
            output_content=prepared.content,
            verification_ref=prepared.verification_ref,
        )