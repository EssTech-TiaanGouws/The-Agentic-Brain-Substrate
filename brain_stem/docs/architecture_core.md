# Agentic Brain Substrate: Master Architecture

**Document version:** 2026.09.30.Absolute-Core
**Status:** Active master source of truth; Stacey Core v0 random-init neural skeleton and synthetic smoke path implemented; training has not started
**Runtime boundary:** Local, in-process, platform- and model-provider-neutral

This is the only live source of planning, architecture decisions, implementation status, and verification results. `brain_stem/charter.md` preserves the full frozen blueprint inputs, including the detailed taxonomy and cross-network swarm vision; this master records current interpretation, refinements, and measured status. Code, schemas, and tests implement or check the master; no parallel roadmap or test-result report is maintained.

## 1. Purpose and Operating Model

**Stacey is the complete user-facing model:** the learned World Model Core, dynamically selected specialist models, their shared world state, and the deterministic runtime that keeps their operation coherent and safe. Stacey is not just the Core and not any one specialist. The Core is one persistent learned model intended to remain resident on the available accelerator; it interprets context, reasons about world/task state and action consequences, compiles or revises task graphs, selects capability slots, and integrates specialist results. It does not generate conversational copy or directly perform every task.

The Core is distinct from deterministic execution controls and the durable world-state ledger. The Core is neural cognition. The runtime enforces authorization, resource admission, model lifecycle, typed contracts, receipts, and action policy. The Canonical World Model Ledger (FB-004) is provenance-bearing persistent state that the Core can query and update; it is not the Core's weights or hidden activations. Model proposals do not become trusted world state until validated and committed through deterministic controls.

The design is provider-neutral. Model runtimes, hardware backends, identity sources, and future surface connectors are replaceable adapters, not hard-coded core dependencies. Local laptop, mobile edge, and optional private-cloud execution all belong in the design; local hardware is the first evaluation surface. Cross-surface placement depends on capability fit, measured resources, privacy scope, availability, and latency/cost policy. It is not a fixed model-to-device map.

The current repository is not a trained learned swarm: it contains deterministic CPU mocks, resource/receipt primitives, an injected-authority transaction coordinator, in-memory adapters, and Stacey Core v0's random-init encoder-decoder scaffold. It has no trained Core or specialist weights, real dataset pipeline, production training backend, mobile NPU execution, cloud delegation, or real file writer.

The system is intended to expand across reasoning/planning, coding, document and data work, multimodal interaction, communication, and authorized actions. "Everything imaginable" is an extensibility goal, not a guarantee that any finite model set solves every task. Each capability claim must be demonstrated by task-specific evaluation.

## 2. Taxonomy and Implementation Status

There is one persistent learned World Model Core (Block 0) and an initial set of fifteen transient capability slots (Blocks 1–15). Stacey is the collective across the Core and specialist models. The 8 cognitive layers describe the overall flow; the 15 blocks refine capability roles. Neither count dictates model-file count. The starter taxonomy is extensible by append-only, versioned capability IDs. A capability may have multiple candidate artifacts, and a model may serve multiple capabilities only when evaluation justifies it. Current specialist implementations are software mocks; Stacey Core v0 is randomly initialized and untrained. The supplied sizes and hot-swap performance claims remain unverified targets.

### Eight Cognitive Layers

These are functional stages, not a requirement to train exactly eight models:

1. **Sense:** learned perception models turn image, audio, and other modality input into structured observations.
2. **Understand:** the Core interprets raw user intent by default; a separate understanding specialist is optional if evaluation shows it helps.
3. **Reason:** the Core builds hypotheses, handles ambiguity, predicts consequences, and creates/revises dependency-aware plans.
4. **Act:** learned specialists propose typed operations; deterministic services authorize, stage, commit, and receipt them.
5. **Specialist network:** expandable learned capability slots handle coding, extraction, critique, language, and future domains.
6. **Memory:** deterministic scoped stores preserve facts, episodes, artifacts, and provenance; learned retrieval/ranking may assist but cannot silently rewrite canonical state.
7. **Orchestration:** the Core chooses the cognitive path; deterministic scheduling enforces resource budgets, isolation, and model lifecycle.
8. **Execution tools:** deterministic permissioned interfaces act on the local or explicitly authorized external environment.

### Stacey Core v0 Contract

Stacey Core v0 is **one learned model** handling understanding, reasoning, planning, consequence prediction, and specialist selection. These are unified responsibilities inside one Core candidate, not separate learned models on every ingress. Specialist capabilities remain separate slots activated by the Core's plan.

The Core receives one schema-validated Unified Context Ingress Envelope containing exactly four data matrices:

1. **Inbound intent vector:** raw user language or a typed upstream event signal. The Core must not receive raw file dumps or unbounded system logs; those are summarized and sourced into ledger assertions by the host/specialists.
2. **Trusted scope lock:** tenant, user, project, and workspace IDs supplied by the deterministic host. The Core must echo them exactly; it cannot authenticate or change them.
3. **Canonical ledger snapshot:** bounded state assertions, verified facts, current environment state, and provenance references already filtered to the trusted scope.
4. **Hardware and swarm capability matrix:** timestamped resource-domain measurements, active leases, and registered/approved specialist slots with capability IDs, status, artifact digest, modalities, and estimated requirements.

Before any specialist or tool runs, the Core emits one JSONL Declarative Task Graph Envelope. Required contents:

- Transaction and correlation IDs, `assigned_block_id=BLOCK_0_CORE`, and an internal predicted-consequences summary.
- The unchanged trusted four-field scope vector.
- An acyclic task DAG naming registered capability slots, target block IDs, conditional dependencies, and exactly one per-step resource estimate. Estimates are untrusted predictions; the deterministic lease manager checks fresh measurements and may deny the plan.
- A calibrated `confidence_coefficient` in `[0,1]`, where higher means more confident, plus structured clarification reason codes. The host compares it against an injected minimum-confidence policy; below threshold, it refuses specialist dispatch unless the output explicitly requests clarification. Calibration is an empirical property, not guaranteed by emitting a number.

The Core output is not user-facing chat, executable code, shell commands, or a direct mutation. Clarification is an action directive with reason codes; a separate Respond capability may phrase it for the user. Before execution, a deterministic validator checks transaction identity, exact scope echo, registered and available capability/block pairs, DAG acyclicity, resource-estimate coverage, and the clarification threshold. Resource estimates do not grant or reserve memory.

### Stacey Model Package Separation

- `brain_stem/models/stacey/core/` contains only Stacey Core v0's architecture candidate, configuration, byte-token protocol, ingress/decision types, random initializer, and model checkpoint code.
- `brain_stem/training/stacey/` contains training objectives and smoke-run helpers. It is separate from the model package; dataset loaders, a real trainer, and run orchestration are not yet implemented.
- `brain_stem/src/swarm_core/` remains deterministic collective infrastructure: specialist catalog, evaluation, resource leases, ledgers, and coordination. It does not become Stacey's learned Core.

The first architecture hypothesis is a configurable UTF-8 byte-level encoder-decoder Transformer. It uses no pre-existing installed model weights or tokenizer vocabulary. It emits serialized structured decisions autoregressively. This is an experiment candidate, not a final architecture selection; compare alternatives against the same task-graph benchmark. Dimensions, layer counts, context limits, dropout, initialization seed, and resource profile are explicit configuration, not machine-specific constants. A random-init forward pass proves only that tensor plumbing works.

### Surfaces and Learning

- **Laptop:** first evaluation surface for the persistent Core and dynamically selected specialists; no specific device is assumed by the architecture.
- **Mobile:** later edge surface for compact, quantized models. Updates are versioned and require provenance, evaluation, compatibility checks, and approval.
- **Private cloud:** optional extension for tasks exceeding local capability. Send only authorized, minimized context; validate returned results locally.

Use a hybrid, capability-specific model-development strategy: compare teacher-student distillation, task-specific training/fine-tuning, and other methods. Teachers may be training tools but are not runtime dependencies. Do not silently reuse models already installed on a user's system; each candidate's source weights, version, and training lineage must be explicit. Train from approved datasets or explicitly opted-in traces only. The Foundry may propose versioned candidate models; holdout, regression, calibration, safety, and provenance checks plus explicit approval are required before live admission. It may not self-modify or deploy into the active swarm.

User activation authorizes a bounded task/session and declared scope. The agent may autonomously complete steps within that scope without asking for each action. A new task or scope expansion requires renewed activation. Deterministic policy and resource controls remain authoritative; models cannot grant themselves permissions.

Treat the under-400 MB Core as an aspirational target to compare against larger candidates. At nominal 4-bit weight storage, 400 MB holds about 800 million raw parameters before metadata and runtime buffers. Model quality cannot be inferred from size or from removing chat/fact weights. Measure reasoning quality, memory use, load latency, and runtime throughput separately before setting service targets.

| ID | Functional block | Lifecycle | Proposed size | Status |
|---|---|---|---:|---|
| 0 | World Model Core / Switchboard | Persistent learned Core target | <400 MB aspirational | Stacey Core v0 encoder-decoder scaffold exists; random weights, untrained |
| 1 | Sensory Optical | Transient | 150–200 MB | Implemented as an in-memory mock; no vision model |
| 2 | Semantic Classifier | Transient | 80–120 MB | Implemented; deterministic lexical mock |
| 3 | Tactical Decomposer | Transient | 250–300 MB | Implemented; deterministic one-step plan |
| 4 | Temporal Chrono | Transient | 100–140 MB | Deferred |
| 5 | Auditory Telemetry | Transient | 90–130 MB | Deferred |
| 6 | Adversarial Verifier | Transient | 180–220 MB | Implemented; validates plan and worker result |
| 7 | Relational Context | Transient | 140–180 MB | Deferred; context decay policy needs specification and measurement |
| 8 | Epistemic Auditor | Transient | 160–200 MB | Implemented; checks goal and transaction traceability |
| 9 | Algorithmic Coder | Transient | 300–350 MB | Implemented as a mock artifact inspector; no neural coder or compiler |
| 10 | Structural Ingress | Transient | 120–160 MB | Deferred |
| 11 | Declarative API Driver | Transient | 200–250 MB | Implemented as an in-memory staging mock; no filesystem mutations |
| 12 | Linguistic Copy | Transient | 220–260 MB | Deferred; no language generation |
| 13 | Volatility Sensor | Transient | 90 MB | Deferred |
| 14 | Provenance Graph-Linker | Transient | 110 MB | Deferred; no canonical ledger yet |
| 15 | SIMD Architectural Adapter | Transient | 130 MB | Deferred optional hardware adapter |

“Implemented” here means a local Python module in the current prototype, not a neural splinter. The core maintains the functional taxonomy even where implementation is deferred. Do not add block IDs outside 0–15 or combine block responsibilities without updating this document.

## 3. Building Blocks and Control Flow

The `brain_stem/substrate/` package is the Phase 1 composition root and typed contract boundary. `SubstrateKernel` is the light kernel; each implemented block has its own directory under `brain_stem/substrate/blocks/`. The mock worker is separately isolated under `brain_stem/substrate/workers/`. The Phase 2–4 building blocks live under `brain_stem/src/swarm_core/`. Tests exercise them without a server, model download, GPU, or external process.

For `ROUTE_TASK`, the current control flow is:

1. Ingress accepts a typed `Intent` and trusted `authorized_scope`.
2. The kernel requires non-empty `tenant_id`, `user_id`, `project_id`, and `workspace_id` on both scopes and requires exact equality. It also checks the supported action and non-empty transaction ID, correlation ID, and goal. Reject before invoking any worker on failure.
3. Block 2 classifies the intent; Block 3 creates a task plan.
4. The registered in-process CPU mock worker executes the plan.
5. Block 6 validates step uniqueness, the registered mock capability, status, transaction/correlation IDs, and completed steps.
6. Block 8 verifies the plan transaction and returned goal against the original intent.
7. Only after both checks pass does the kernel return an `ExecutionReport` to its in-process caller.

Block 6 always precedes Block 8. There is no network port or output listener. The host invoking the kernel is responsible for obtaining `authorized_scope` from a trusted local identity boundary; comparing two untrusted caller-supplied values is not authorization. The current prototype accepts that trusted scope as an explicit method argument and does not implement identity authentication or a ledger.

The current classifier and decomposer are deliberately small deterministic mocks, not trained models. The current planner emits one `mock.general` step. Unknown capabilities fail closed in Block 6. The worker interface is replaceable without changing the ingress, contract, or validation boundaries.

## 3A. Stacey Core v0 Skeleton

Stacey is the total learned Core-plus-specialist collective. The Core in `brain_stem/models/stacey/core/` is separate from the deterministic `brain_stem/src/swarm_core/` execution and governance substrate and from the separate `brain_stem/training/stacey/` training support. It is one learned model candidate for understanding, reasoning, planning, consequence prediction, and specialist selection; it is not eight separate layer models and is not the complete Stacey collective by itself.

The v0 architecture is explicitly a **configurable byte-level encoder-decoder Transformer experiment**, not a frozen final architecture. `StaceyCoreConfig` injects dimensions, layer counts, context bounds, and dropout. The UTF-8 byte codec has a stable 259-symbol vocabulary including PAD/BOS/EOS, avoiding dependence on an existing installed model or tokenizer. Other architectures/tokenizers remain candidates to compare using the Core benchmark.

The neural model receives one schema-validated `UnifiedContextIngress` with four matrices: raw language or structured event intent; the trusted four-field scope; a bounded canonical-state ledger snapshot with provenance; and measured resource/lease plus approved-specialist capability metadata. The deterministic host supplies and validates trusted scope and data permissions. Raw files, unbounded conversation logs, and arbitrary executable payloads are not model ingress.

For every turn, Stacey Core v0 is intended to emit one JSONL `CoreDecisionEnvelope` with four required primitives: (1) exact scope echo, checked against trusted ingress; (2) a task DAG naming capability slots and conditional edges; (3) per-step resource estimates, advisory until checked by deterministic lease/admission control; and (4) a calibrated `confidence_coefficient` plus structured clarification reason codes. It also carries transaction/correlation IDs and an internal predicted-consequence summary. The Core does not output user-facing conversational copy, executable code, shell commands, or file mutations. Invalid JSON, scope drift, unavailable capability selection, missing resource estimates, cyclic graphs, or a plan that dispatches while clarification is required are rejected before execution.

`initialize_stacey_core` creates reproducible random initial weights only from an explicit seed. The configurable candidate reports parameter count and a raw weight-storage estimate for profile comparisons; neither is a runtime-memory measurement. `forward` returns token logits; greedy decoding is bounded by injected config and produces tokens only. JSON parsing and semantic envelope validation are separate, deterministic operations. A deterministic confidence-policy gate blocks specialist dispatch below a caller-injected threshold unless clarification was requested. Checkpoints use atomic replacement, an explicit permission mode and SHA-256 verification, and PyTorch's weights-only loader. The saved model checkpoint does not include optimizer/data-cursor resume state or constitute an approved candidate.

`brain_stem/training/stacey/` is intentionally separate. Its teacher-forcing objective and one-step smoke runner accept caller-provided tensors; the smoke test uses synthetic fixture input/target, proves finite loss/gradients/parameter update, and consumes no dataset. This is plumbing validation, not training Stacey or measuring intelligence. No dataset, tokenizer training, production trainer, accelerator adapter, or trained weights are included.

**Stacey Core v0 implementation state:** `brain_stem/models/stacey/core/` contains a configurable, randomly initialized byte-level encoder-decoder candidate and typed four-matrix ingress/task-graph output contracts. The current smoke path uses synthetic fixture targets to check tensor shapes, finite forward/loss values, gradients, parameter updates, deterministic seeding, bounded decoding, and SHA-256 checkpoint round trips. It does not establish that random weights understand intent, reason, predict outcomes, or produce valid plans reliably. The `confidence_coefficient` uses higher values for greater confidence; its calibration is untrained and unmeasured. The runtime's caller-supplied threshold is the safeguard until learned calibration is evaluated. Parameter count and raw weight-storage estimates are exposed for profile comparison, but neither measures runtime memory or accelerator residency.

Core parameter count and raw storage-size estimates are available to compare configuration candidates; neither estimates peak runtime memory, accelerator residency, speed, or trained quality. Do not present the sample small configuration or the under-400 MB goal as a chosen production architecture.

## 4. Contracts and Security Rules

`brain_stem/substrate/contracts.py` defines the in-process typed contracts. `brain_stem/schemas/contracts.schema.json` describes the serialized Phase 1 intent envelope. The current kernel accepts Python dataclasses directly; it does not yet decode JSON or invoke a runtime schema validator. Any serialized ingress adapter must validate against the schema and invoke the same scope checks before routing. The scope vector is exactly four fields; a missing, blank, extra, or mismatched field is rejected. Tenant isolation is based on trusted authorization context, not the scope payload alone.

### Rule I: Declarative Intent Only

Runtime actions are typed intent values, not shell arrays, executable strings, or arbitrary commands. The current implementation accepts only `ROUTE_TASK`. Future create/modify/verify operations must be added as explicit typed schemas and routed through FB-038 (Document Operations Authority); no block may write directly to the filesystem. Source code intended for a future file edit may be opaque content data, never an instruction to execute it.

### Rule II: Twin-Receipt Mutation Guard

The Phase 1 routing slice performs no mutation. Phase 3 composes the durable receipt ledger with injected transaction, stage, and mutation-authority interfaces, but does not provide a real filesystem writer. Before any real mutation is enabled, the sequence remains mandatory: validate and authorize; durably persist an FB-027 `PREPARED` receipt; call FB-038; validate the final state; persist an `OUTCOME` receipt. If mutation or finalization fails, compensate from recorded prior state. Report `INDETERMINATE` if compensation cannot be verified. If `PREPARED` persistence fails, do not mutate. Current tests use fakes and do not prove disk mutation recovery.

### Rule III: No Hidden Hardcoding; Model and Provider Neutrality

Do not embed machine-specific budgets, memory cushions, model sizes, filesystem locations, provider endpoints, hardware flags, test fault switches, or mutable policy values in the kernel or block logic. Supply operational values through validated typed configuration or an injected capability/resource adapter; fail clearly when required configuration is absent rather than silently selecting a machine-specific default. Resolve resource budgets from a named runtime profile and current resource snapshot, with the profile stored/configured outside the core.

Stable protocol invariants are not hidden configuration: block IDs, the four scope field names, typed action names, schema version, and receipt state names remain explicit in contracts and code. Test fixtures may contain example values, but fault injection belongs in fake adapters, not untrusted intent payloads. The Phase 1 keyword vocabulary and `mock.general` route are temporary deterministic fixtures, not production classification or an extensible capability registry.

Core interfaces use functional capability keys and resource profiles, never commercial model names, provider URLs, fixed IP addresses, or cloud labels. Optional runtime and hardware adapters own their own configuration. The core does not depend on any model provider.

### Rule IV: Local-First In-Process Runtime

The kernel, blocks, worker, FB-027, and FB-038 boundaries are local in-process modules. Do not add always-running third-party daemons, unmanaged socket listeners, or mandatory external services. A future optional library dependency must not silently create an external runtime service.

### Additional System Laws

- **Taxonomy isolation:** exactly Block 0 plus Blocks 1–15; each block has one documented functional responsibility.
- **Critique coercion:** all future generated or worker-produced output passes Block 6, then Block 8, before the caller receives it.
- **Resource cleanup:** any future transient resource is released deterministically on success, failure, and cancellation, using structured cleanup (`finally`/context managers). Python garbage collection alone is not proof of OS unmapping or device-memory release.
- **Planning order:** Blocks 2 and 3 precede any future Block 12 language generation.
- **Consensus:** ambiguous or conflicting assertions must be resolved by a future validation-cluster consensus flow before committing to FB-004, the canonical ledger. Neither consensus nor FB-004 is implemented.
- **Provenance:** future ledger assertions must be source-linked through Block 14. No immutable hash chain is implemented today.
- **Scope-bound patches:** future mobile updates must bind to all four scope fields. Mobile distillation and autonomous splinter creation are deferred and require a separate security and lifecycle design in this master document before implementation.
- **Hardware tuning:** Block 15 may host optional capability detection and tuning. CUDA/RTX/AVX profiles are not core requirements and cannot be hardcoded into the portable kernel.

## 5. Failure Semantics

Malformed or incomplete intent, missing identifiers, unsupported action, incomplete trusted scope, or any four-field mismatch raises `IngressRejected` before worker execution. A structurally invalid plan or mismatched worker result raises `IntegrityViolation`; no execution report is returned. Worker exceptions propagate to the caller and do not become success. The Phase 1 prototype performs no persistent mutation. The Phase 3 coordinator connects receipts to injected workers and mutation authorities; real target mutation and compensation remain unimplemented.

## 6. Phased Plan and Acceptance Criteria

The phases below are maintained only here. A phase is implementation sequencing, not a gate that blocks independent work. Record outcomes inline in Section 7 and keep moving when an unmet item is safely deferrable.

### Phase 0: Architecture and Contract Baseline

**Status: Complete for the initial prototype.** This master document defines the taxonomy, rules, boundaries, current status, and forward plan. The Phase 1 intent schema is machine-readable. The README is navigation only; no separate roadmap or test-result document exists.

### Phase 1: CPU-Only General Routing

**Status: Implemented and tested.** Keep the router in-process; use deterministic mock workers. Require exact four-field scope checks, planning before work, and ordered Block 6 then Block 8 validation. Run with the Python standard library only.

### Phase 2: Lifecycle and Integrity

**Status: Foundational building blocks implemented; transaction orchestration is not implemented.** The proposal is useful as a failure-path checklist (admission denial, worker failure, outcome-receipt failure, compensation failure, and cleanup), but its sample manager was not copied. An in-memory mutable dictionary is neither durable nor immutable; the backup snapshot and rollback are placeholders; target writes are not actually performed through FB-038; the manifest path and splinter key are unused; and magic request flags/paths are not safe fault-injection interfaces.

#### No-Hardcoding Resource Admission

- Inject a validated `ResourcePolicy` and a live `ResourceSnapshot` from the selected runtime adapter. Represent capacities in one explicit base unit (bytes) and identify the resource domain (for example host memory, mapped-file residency, or device memory). Do not put `6144`, `1536`, or `350` MB defaults in the kernel.
- A 6 GB device limit or 1.5 GB PagedAttention cushion is an optional named profile value only when the selected backend actually exposes that resource and cache. It is not a platform-wide invariant. Missing or stale measurements fail closed for the operation that needs them; they do not fall back to a guessed ceiling.
- Admission reserves the requested budget atomically before work. Use a per-process synchronized lease/allocator so concurrent tasks cannot all pass against the same free capacity. Validate configured and requested values as finite, non-negative, and within representable limits. Releasing the lease in structured cleanup is authoritative; `gc.collect()` is not a memory-release guarantee and must not be used as a substitute for admission accounting.
- Inject fake resource providers and explicit budgets in tests. Assert exact boundary behavior, over-budget denial before worker invocation, concurrent reservation safety, and release on every exit path.

#### Two-Phase Receipts and Mutation Order

Keep FB-027 and FB-038 as local in-process authority interfaces. Their storage path, durability mode, and mutation target authorization come from explicit configuration and trusted scope, not hard-coded paths. JSONL may be a serialization choice, but does not imply a server or listener.

For a mutating intent, the order is:

1. Validate the typed intent, all four trusted scope fields, transaction identity/idempotency, and target authorization.
2. Acquire the resource lease. Pure validation and admission do not modify the target.
3. Persist and durably flush an immutable FB-027 `PREPARED` event containing enough transaction, scope, target, and prior-state/delta references to recover. If it cannot be persisted, deny before any worker side effect or target mutation.
4. Execute through the capability worker and FB-038 staging boundary; never write the target directly from a block. Run Block 6, then Block 8, and verify the staged result before commit.
5. FB-038 commits the authorized change. FB-027 appends a durable `OUTCOME` event only after verifying the committed state.
6. If any step after `PREPARED` fails, FB-038 compensates from recorded prior state/delta and verifies restoration. Append an immutable compensation outcome when possible. If restoration cannot be verified, return `INDETERMINATE`, never success, and block further mutations against the unresolved transaction/target until reconciliation.
7. Release the resource lease in `finally` on success, failure, or cancellation. No telemetry path may translate `INDETERMINATE` into success.

Receipts are append-only state transitions, not mutable status dictionaries: `PREPARED` -> `OUTCOME`, `COMPENSATED`, or `INDETERMINATE`. Enforce transaction uniqueness/idempotency and serialized state transitions within the process. A crash that leaves only `PREPARED` is unresolved, not success; recovery must inspect the durable journal and target state and compensate or reconcile before permitting a retry. If writing the terminal receipt itself fails, preserve the unresolved prepared state and surface an explicit failure; do not fabricate a durable terminal record.

#### Phase 2 Test Matrix

Keep the four proposed scenarios: admission overflow is denied; worker failure is compensated; outcome persistence failure plus failed compensation returns `INDETERMINATE`; resource accounting returns to baseline in all paths. Improve them as follows:

- Inject faults using fake receipt-store, worker, resource-provider, and mutation-authority implementations. Never accept `test_inject_fault` or a magic target path from an intent.
- Add PREPARED persistence failure (assert worker and mutation authority were never called), successful commit and durable OUTCOME, failed compensation with no success result, idempotent duplicate transaction handling, malformed/stale resource snapshots, concurrent admission, cleanup after cancellation/exception, and startup recovery of unresolved PREPARED events.
- Use a temporary isolated store and test targets supplied by fixtures. No machine-specific paths, hardware ceilings, target files, or provider labels live in production code.

The transaction ordering and acceptance tests are required before enabling file mutation; they do not prevent progress on independent CPU-only work. Keep Phase 2 local and in-process, with no daemon or socket service. A resource domain that the selected CPU mock runtime cannot observe is not claimed as protected by its test results.

### Phase 3: Transaction Coordinator

Add a synchronous in-process coordinator that composes the Phase 2 lease manager and receipt ledger with injected stage worker, ordered Block 6/8 verifier, and FB-038 commit/compensation authority. Validate and authorize before admission; reserve a lease; persist `PREPARED`; stage and verify; commit; append `OUTCOME`. A failure before commit invokes compensation. Verified compensation records `COMPENSATED`; failed compensation raises an explicit `INDETERMINATE` error. If commit was attempted but terminal `OUTCOME` persistence fails, do not compensate based on an uncertain receipt write; surface unresolved state and require journal/target reconciliation. No path returns success for unresolved or indeterminate work. Duplicate completed requests are idempotent; unresolved `PREPARED` transactions block replay pending reconciliation. Resource leases release on all exits.

**Status: Implemented and tested as an orchestration boundary.** `brain_stem/src/swarm_core/transaction_coordinator.py` composes injected authorization/commit authority, resource lease, receipt ledger, stage worker, and ordered verifier interfaces. It implements preparation denial, replay/no-reexecution, recovery blocking, compensation, strict indeterminate errors, and unresolved outcome-write handling. It does not implement an OS file writer or claim cross-process locking. Deterministic test doubles cover same-process concurrent transactions and concurrent duplicate requests. Keep the coordinator bounded and synchronous; do not add a background loop or socket service.

### Phase 4: Transient Document Format Adapters

Define a pluggable `IDocumentFormatHandler` interface and explicit registry. Keep Block 1 (sensory/structure inspection), Block 9 (algorithmic/code artifact inspection), and Block 11 (declarative document staging) as separate transient, provider-neutral mock worker packages. The adapters may validate/transform in-memory payloads only; they do not load neural models, compile or execute generated code, or write files. Format handlers and worker capabilities are injected, not selected through machine-specific defaults. Tests cover registry selection, unsupported formats, worker isolation, and concurrent coordinator use across transactions.

**Status: Implemented and tested as in-memory mocks.** `brain_stem/src/swarm_core/format_adapters.py` defines the handler protocol and explicit registry. Separate Block 1, 9, and 11 packages are composed by `brain_stem/src/swarm_core/workers/registry.py`. Block 11 returns staged bytes and verification metadata only; no handler has filesystem authority.

### Phase 5: Optional Hardware Adapter

Only after a measured runtime is selected, evaluate source-compiled hardware tuning through an optional adapter. Measure CPU feature detection and backend correctness on supported hardware. An RTX 4050 or CUDA target is not required for the platform-neutral core; do not claim acceleration from configuration flags alone.

### Neural Model Development Plan

**Status: Random-initialized Core skeleton and training plumbing are smoke-tested; learned-model training/inference is not implemented.** The repository contains a candidate-governance catalog, structured Core evaluator, explicit training-run preflight, Stacey's v0 model topology, and a synthetic loss/gradient/checkpoint path. No learned candidate has been trained or evaluated.

1. **Core benchmark first:** evaluate the learned World Model Core locally for raw-intent interpretation, reasoning/planning, ambiguity handling, scope interpretation, consequence prediction, and specialist selection. The benchmark is hardware-neutral; profile candidate hardware only after task quality is measured.
2. **Core candidate comparison:** compare new, brand-neutral architecture/weight candidates and parameter/quantization profiles. Do not select a model size by assumption. Keep the under-400 MB profile as a target and document quality/resource tradeoffs.
3. **First specialist: Critique/Evaluation:** build an independent learned reviewer for both plans and completed results. It checks evidence, provenance, uncertainty/calibration, and explicit system rules. Model agreement is advisory, never proof of truth.
4. **Expand from observed gaps:** use benchmark failures to select the next capability slot (for example coding, structured-data extraction, sensing, or response generation). Measure each specialist alone and as part of the Core-led system; compare against Core-only and single-model baselines.
5. **Governed learning:** evaluate distillation, capability-specific training/fine-tuning, or other methods per slot. Candidate generation, evaluation, explicit approval, versioned registry admission, and rollback precede live use. Training data requires approval or explicit opt-in.
6. **Surface expansion:** once local behavior is characterized, evaluate mobile and optional private-cloud profiles against the same capability contracts, privacy controls, and quality tests.

Implemented planning support: `brain_stem/src/swarm_core/model_catalog.py` records proposed Core/specialist candidates with capability IDs, artifact hash/size, parameter count, authorized training-source references, evaluation evidence bound to the exact artifact, and explicit approval references. Capability lookup returns every approved candidate; it does not silently select or load one. Eight contract tests exercise provenance, evaluation, rejection, and approval.

`brain_stem/src/swarm_core/core_evaluation.py` defines a hardware-neutral Core candidate interface, structured benchmark cases, injected acceptance thresholds, and artifact-bound `EvaluationEvidence` suitable for the candidate catalog. `CoreExperimentWorkbench` checks that an injected inference adapter matches the registered candidate ID and artifact digest, runs the benchmark, and records a passing or rejected evaluation. It never approves the candidate; approval remains a separate explicit action. The initial contract measures scope integrity, task-graph validity, clarification accuracy, capability coverage, and interpreted-intent presence. Twelve fixture-based tests cover evaluator behavior plus catalog integration, digest binding, failure recording, and approval separation. This is an evaluator harness, not a learned model, real benchmark result, training data, or performance claim.

`brain_stem/src/swarm_core/training_readiness.py` defines a Core run manifest for from-scratch, distillation, or fine-tuning experiments. It binds architecture/tokenizer/config references, candidate identity, parameter/context targets, seed, dataset content hash and provenance, license-review reference, disjoint train/validation/test record IDs, authorized sources, compute profile, checkpoint limits, teacher/base lineage, and explicit run approval. `CoreTrainingReadinessChecker` compares this plan against injected backend capabilities, accessible datasets, approved runs, teacher availability, and measured accelerator memory, returning specific blockers and a reproducible plan digest. Nine fixture tests cover ready and blocked plans, split leakage, lineage, and plan-digest provenance. This is a preflight contract only: it does not verify dataset contents or licenses, connect to a GPU provider, execute training, or validate a produced artifact.

### Stacey Core v0 First Benchmark

The first real-model benchmark is **Task-Graph Generation and Context-Aware Revision**. Cases must cover:

- exact preservation of all four trusted scope fields;
- capability-slot selection and valid acyclic dependency graphs;
- ambiguity/low-confidence cases that request clarification and dispatch no specialist work;
- consequence prediction and resource-estimate fields, scored against reviewed expected outcomes;
- revision after a specialist failure appears in a subsequent ledger snapshot, avoiding the failed route and selecting a valid alternative when one exists.

Thresholds remain caller-supplied and provisional until a human-reviewed dataset and baseline exist. Current fixture cases test contract mechanics, not semantic reasoning. A Core candidate is compared across size/config profiles and against a baseline; the under-400 MB target is accepted only if measured quality and resource costs justify it.

Measure first-token latency, Core decision latency, specialist cold-load time, prompt-processing throughput, generated tokens/second, end-to-end task success, quality, memory, and energy separately. Autoregressive response generation is sequential, so thousands of output tokens/second is not a universal promise. Establish numeric targets from measured baselines and explicit workload profiles.

The initial task suite ultimately covers reasoning/planning, coding/software work, documents/structured data, multimodal interaction, and general authorized actions. The first learned-model milestone is Stacey Core task-graph generation and context-aware revision; the first specialist milestone is Critique/Evaluation.

### Operational Verification

`verify_parity_gates.py` is an on-demand, one-shot local command: it runs the complete test suite with bytecode writing disabled, checks required source/schema/master files and single-source policy, prints a strict true/false clearance value, and returns a matching process status. It must not persist a separate parity/test-results document. **Latest verification:** all 96 canonical tests pass; clearance is false only because preserved root-level `docs/`, `src/`, and `tests/` duplicate the canonical `brain_stem/` tree. No always-on autonomous coding loop or daemon is permitted; implementation remains an explicit bounded coding operation with operator review.

### Deferred Tracks

Real neural inference, generated-code execution, actual document/file mutation, mobile model deployment/distillation, private-cloud delegation, autonomous Foundry operation, and persistent world-model consensus remain future work. Add their design, risk, and acceptance criteria here before implementation.

## 7. Implementation and Verification Record

**2026-09-30: Initial CPU routing slice**

- Implemented the in-process `SubstrateKernel`, typed intent/scope/plan/result contracts, deterministic Blocks 2 and 3, CPU mock worker, and Blocks 6 and 8.
- Added tests for successful routing, exact scope mismatch rejection before worker calls, incomplete scope rejection, malformed runtime field rejection, unsupported action rejection, Block 6-before-Block 8 ordering, and epistemic rejection of tampered worker output.
- Command: `PYTHONDONTWRITEBYTECODE=1 python3 verify_parity_gates.py`
- Result: **7 tests passed.** The schema also passed JSON parsing and Draft 2020-12 validation with a representative intent. No model download, GPU, daemon, network listener, or third-party Python package is required to run the routing tests.
- Not verified or implemented at the Phase 1 checkpoint: real authentication, neural inference, memory limits, runtime hardware admission, persistent receipts, filesystem mutation, rollback, crash recovery, ledger provenance, and sub-millisecond swapping.

**2026-09-30: Phase 2 resource and receipt building blocks**

- Added `brain_stem/src/swarm_core/lease_manager.py` with injected resource policy, profile, live snapshot, freshness validation, atomic thread-safe byte leases, and deterministic context-manager release. No device-specific budget or fallback is embedded in production code.
- Added `brain_stem/src/swarm_core/durable_ledger.py` with append-only JSONL events, durable flush, injected storage path and permission mode, contiguous sequence and SHA-256 chain validation, transaction/idempotency checks, shared in-process per-path serialization, and startup enumeration of unresolved `PREPARED` transactions.
- Added `brain_stem/tests/cases/phase2_cases.py` covering exact admission boundary, over-budget denial, concurrent leases, multiple concurrent ledger instances, append ordering, idempotent/conflicting retries, PREPARED and terminal write failures, partial-tail rollback, unresolved recovery, lease cleanup, invalid/stale resource configuration and snapshots, injected storage-mode validation, and journal corruption.
- Command: `PYTHONDONTWRITEBYTECODE=1 python3 verify_parity_gates.py`
- Result: **24 tests passed** (7 Phase 1, 17 Phase 2 contract tests). No third-party daemon, model, GPU, or test fault flag in production code.
- Phase 2 checkpoint limits: resource snapshots were test-injected; no real hardware provider existed. The ledger is synchronized within one process, not across processes, and its hash chain detects edits but does not prevent privileged filesystem tampering. At that checkpoint, transaction orchestration, FB-038 writes, actual compensation, and crash reconciliation had not yet been implemented; no filesystem mutation was enabled.

**2026-09-30: Phase 3/4 coordinator and adapter slice**

- Added `brain_stem/src/swarm_core/transaction_coordinator.py`, with an atomic ledger prepare-once API and injected authorization, stage worker, Block 6/8 verifier, and commit/compensation authority interfaces. No target file is written by the coordinator.
- Added `brain_stem/src/swarm_core/format_adapters.py` and separate transient mock packages for Blocks 1, 9, and 11. Block 11 prepares content in memory only; the format/capability registries reject missing, duplicate, and unknown entries.
- Added `brain_stem/tests/cases/phase3_cases.py` for PREPARED denial, exact stage/critic/audit/commit ordering, worker/commit failures, compensation/INDETERMINATE handling, unresolved outcome receipt, replay/idempotency, cross-session concurrency, duplicate concurrent execution, and lease cleanup.
- Added `brain_stem/tests/cases/phase4_cases.py` for handler order, duplicate/unknown formats, invalid handler output, mock worker taxonomy, and in-memory-only document staging.
- The single discovered suite entrypoint is `brain_stem/tests/test_swarm_primitives.py`.
- The neural planning pass added `brain_stem/src/swarm_core/model_catalog.py` and `brain_stem/tests/cases/phase5_cases.py`. It governs candidate metadata and approval but does not load, train, or claim to evaluate neural weights.
- Added `brain_stem/src/swarm_core/core_evaluation.py` and `brain_stem/tests/cases/phase6_cases.py` as a structured Core benchmark contract. Its fixture tests validate the evaluator mechanics only; no learned candidate has been scored.
- Added root `verify_parity_gates.py`, a one-shot full-suite and repository-state check. It prints clearance and exit status; it does not persist `master_parity_ledger.json` or run continuously.
- Result before the model catalog: **45 tests passed** (7 Phase 1, 17 Phase 2, 13 Phase 3, 8 Phase 4); parity sentinel reported `ARCHITECTURE_CLEARANCE=true` in the canonical tree.
- Current canonical test total: **96 tests passed** (74 previous scaffold tests plus 22 Stacey Core v0 schema, neural tensor, checkpoint, confidence-gate, and synthetic training-smoke tests). This verifies contracts and CPU tensor plumbing, not learned-model quality or actual training readiness on an accelerator backend.
- Limits: transaction authorities are fakes; no real file writer, target backup, compensation implementation, hardware resource provider, model inference, cross-process journal lock, persistent parity result file, or autonomous daemon exists. A receipt write error after commit is explicitly unresolved and requires reconciliation before retry.

**2026-09-30: Sovereign brain_stem indexing pass**

- Preserved the complete supplied cross-network blueprint in `brain_stem/charter.md`; all three surfaces, Block 0 plus Blocks 1–15, and Laws I–III are present.
- Re-anchored the active architecture at `brain_stem/docs/architecture_core.md`, the intent schema at `brain_stem/schemas/contracts.schema.json`, and implementation/tests under `brain_stem/src/` and `brain_stem/tests/`.
- Renamed the capability package to `brain_stem/src/swarm_core/`; consolidated discovery at `brain_stem/tests/test_swarm_primitives.py` while retaining phase-specific cases under `brain_stem/tests/cases/`.
- Updated the root README and one-shot sentinel to resolve the new layout. The sentinel checks charter completeness and rejects duplicate legacy root trees, separate roadmap/result ledgers, and autonomous-loop artifacts.
- Root invocation `PYTHONDONTWRITEBYTECODE=1 python3 verify_parity_gates.py`: **45 tests passed; `ARCHITECTURE_CLEARANCE=true`**.

**2026-10-02: Stacey Core v0 empty neural skeleton**

- Added the independent package `brain_stem/models/stacey/core/` with configurable encoder-decoder network, UTF-8 byte codec, four-matrix ingress types, JSONL decision types/validator, reproducible random initialization, and SHA-256-bound atomic checkpoint save/weights-only load.
- Extended the existing `brain_stem/schemas/contracts.schema.json` with Stacey ingress and Core-decision schema definitions; it remains the single shared wire contract.
- Added separate `brain_stem/training/stacey/` teacher-forcing objective and explicitly synthetic one-step smoke function. No trainer or dataset reader is present.
- Added Phase 8 tests for schema-valid ingress/output, scope/capability/DAG/clarification rejection, real CPU tensor shapes, finite loss/gradients, deterministic initialization, bounded generation, and checkpoint round-trip/tamper rejection.
- Added confidence-threshold and context-aware specialist-failure revision contract cases, plus candidate parameter-count/raw-weight-storage profile checks; these are contract/profile checks, not learned quality evidence.
- Runtime used for smoke validation: CPU-only PyTorch dependency from `brain_stem/models/stacey/requirements.txt`; no GPU or model weights.
- Verification: focused Phase 8 suite passed **22 tests**; full canonical suite passed **96 tests**. The parity sentinel also ran all 96 tests and reported only the preserved duplicate-tree clearance failures described above.
- Limits: this architecture is one v0 experiment hypothesis, not final model selection. Random weights are not intelligent; tests prove tensor/training plumbing only. No training dataset, production inference adapter, learned uncertainty calibration, semantic benchmark result, optimizer-resume checkpoint, GPU adapter, or trained Stacey weights exist.