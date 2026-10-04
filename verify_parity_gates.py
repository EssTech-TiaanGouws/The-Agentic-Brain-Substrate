from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQUIRED_FILES = (
    "brain_stem/charter.md",
    "brain_stem/docs/architecture_core.md",
    "brain_stem/schemas/contracts.schema.json",
    "brain_stem/substrate/contracts.py",
    "brain_stem/src/swarm_core/__init__.py",
    "brain_stem/src/swarm_core/lease_manager.py",
    "brain_stem/src/swarm_core/durable_ledger.py",
    "brain_stem/src/swarm_core/transaction_coordinator.py",
    "brain_stem/src/swarm_core/format_adapters.py",
    "brain_stem/src/swarm_core/model_catalog.py",
    "brain_stem/src/swarm_core/core_evaluation.py",
    "brain_stem/src/swarm_core/stacey_core_benchmark.py",
    "brain_stem/src/swarm_core/training_readiness.py",
    "brain_stem/src/swarm_core/hardware_profile.py",
    "brain_stem/src/swarm_core/identity.py",
    "brain_stem/src/swarm_core/operator_authority.py",
    "brain_stem/src/swarm_core/system_inspection.py",
    "brain_stem/src/swarm_core/stacey_bundle.py",
    "brain_stem/src/swarm_core/world_state.py",
    "brain_stem/src/swarm_core/provenance_graph.py",
    "brain_stem/src/swarm_core/workspace_writer.py",
    "brain_stem/src/swarm_core/workspace_action.py",
    "brain_stem/src/swarm_core/workspace_projection.py",
    "brain_stem/src/swarm_core/model_lifecycle.py",
    "brain_stem/src/swarm_core/training_governance.py",
    "brain_stem/src/swarm_core/foundry.py",
    "brain_stem/src/swarm_core/collective_scheduler.py",
    "brain_stem/src/swarm_core/collective_runtime.py",
    "brain_stem/src/swarm_core/response_composer.py",
    "brain_stem/src/swarm_core/stacey_core_backend.py",
    "brain_stem/src/swarm_core/resource_adapters.py",
    "brain_stem/src/swarm_core/volatility_sensor.py",
    "brain_stem/src/swarm_core/relational_context.py",
    "brain_stem/src/swarm_core/structured_ingress.py",
    "brain_stem/src/swarm_core/visual_ingress.py",
    "brain_stem/src/swarm_core/audio_telemetry.py",
    "brain_stem/src/swarm_core/temporal_chrono.py",
    "brain_stem/requirements.txt",
    "brain_stem/models/__init__.py",
    "brain_stem/models/stacey/__init__.py",
    "brain_stem/models/stacey/requirements.txt",
    "brain_stem/models/stacey/core/__init__.py",
    "brain_stem/models/stacey/core/config.py",
    "brain_stem/models/stacey/core/tokens.py",
    "brain_stem/models/stacey/core/inputs.py",
    "brain_stem/models/stacey/core/outputs.py",
    "brain_stem/models/stacey/core/network.py",
    "brain_stem/models/stacey/core/inference.py",
    "brain_stem/models/stacey/core/initialize.py",
    "brain_stem/models/stacey/core/checkpoint.py",
    "brain_stem/training/stacey/objectives.py",
    "brain_stem/training/stacey/smoke.py",
    "brain_stem/training/stacey/corpus.py",
    "brain_stem/training/stacey/trainer.py",
    "brain_stem/training/stacey/resume_checkpoint.py",
    "brain_stem/training/__init__.py",
    "brain_stem/training/stacey/__init__.py",
    "verify_parity_gates.py",
    "brain_stem/tests/test_swarm_primitives.py",
    "brain_stem/tests/cases/phase1_cases.py",
    "brain_stem/tests/cases/phase2_cases.py",
    "brain_stem/tests/cases/phase3_cases.py",
    "brain_stem/tests/cases/phase4_cases.py",
    "brain_stem/tests/cases/phase5_cases.py",
    "brain_stem/tests/cases/phase6_cases.py",
    "brain_stem/tests/cases/phase7_cases.py",
    "brain_stem/tests/cases/phase8_cases.py",
    "brain_stem/tests/cases/phase9_cases.py",
    "brain_stem/tests/cases/phase10_cases.py",
    "brain_stem/tests/cases/phase11_cases.py",
    "brain_stem/tests/cases/phase12_cases.py",
    "brain_stem/tests/cases/phase13_cases.py",
    "brain_stem/tests/cases/phase14_cases.py",
    "brain_stem/tests/cases/phase15_cases.py",
    "brain_stem/tests/cases/phase16_cases.py",
    "brain_stem/tests/cases/phase17_cases.py",
    "brain_stem/tests/cases/phase18_cases.py",
    "brain_stem/tests/cases/phase19_cases.py",
    "brain_stem/tests/cases/phase20_cases.py",
    "brain_stem/tests/cases/phase21_cases.py",
    "brain_stem/tests/cases/phase22_cases.py",
    "brain_stem/tests/cases/phase23_cases.py",
    "brain_stem/tests/cases/phase24_cases.py",
    "brain_stem/tests/cases/phase25_cases.py",
    "brain_stem/tests/cases/phase26_cases.py",
    "brain_stem/tests/cases/phase27_cases.py",
    "brain_stem/tests/cases/phase28_cases.py",
)
REQUIRED_MASTER_MARKERS = (
    "### Code-Completion Register",
    "### Block-by-Block Completion Gates",
    "### Dependency-Ordered Completion Work Packages",
    "### Composite Stacey Collective: Model Parts and Lifecycle",
    "### Stacey Core v0 Contract",
    "## 3A. Stacey Core v0 Skeleton",
    "### Stacey Core v0 First Benchmark",
    "### Phase 3: Transaction Coordinator",
    "### Phase 4: Transient Document Format Adapters",
    "### Operational Verification",
    "### Training Toolchain and Link Register",
    "### Training Run Envelope and Readiness Gates",
    "### Staged Model Build Order",
)
REQUIRED_CHARTER_MARKERS = (
    "End Goal: The Polymorphic Neural Swarm Engine",
    "Automated AI Foundry",
    "brain_stem/src/swarm_core/",
    "SURFACE A:",
    "SURFACE B:",
    "SURFACE C:",
    "5. NON-NEGOTABLE AUTONOMOUS CODING RULES",
    "LAW I:",
    "LAW II:",
    "LAW III:",
)


def evaluate_local_state() -> list[str]:
    issues = [f"Required file is missing: {name}" for name in REQUIRED_FILES if not (ROOT / name).is_file()]
    charter_path = ROOT / "brain_stem/charter.md"
    if charter_path.is_file():
        charter = charter_path.read_text(encoding="utf-8")
        issues.extend(
            f"Frozen charter is missing required section: {marker}"
            for marker in REQUIRED_CHARTER_MARKERS
            if marker not in charter
        )
        issues.extend(
            f"Frozen charter is missing taxonomy entry BLOCK {block_id}"
            for block_id in range(16)
            if f"BLOCK {block_id}:" not in charter
        )
    architecture_path = ROOT / "brain_stem/docs/architecture_core.md"
    if architecture_path.is_file():
        architecture = architecture_path.read_text(encoding="utf-8")
        issues.extend(
            f"Master architecture is missing required marker: {marker}"
            for marker in REQUIRED_MASTER_MARKERS
            if marker not in architecture
        )
    suite_entries = {path.name for path in (ROOT / "brain_stem/tests").glob("test_*.py")}
    if suite_entries != {"test_swarm_primitives.py"}:
        issues.append("brain_stem/tests must expose only test_swarm_primitives.py as its suite entrypoint")
    schema_path = ROOT / "brain_stem/schemas/contracts.schema.json"
    if schema_path.is_file():
        try:
            json.loads(schema_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            issues.append(f"Intent schema is invalid JSON: {error}")
    if (ROOT / "brain_stem/docs/roadmap.md").exists():
        issues.append("A separate roadmap violates the single-source planning rule")
    obsolete_paths = (
        "brain_stem/src/brain_core",
        "brain_stem/schemas/substrate_contracts.json",
    )
    issues.extend(
        f"Obsolete project path remains after re-indexing: {name}"
        for name in obsolete_paths
        if (ROOT / name).exists()
    )
    issues.extend(
        f"Legacy root directory duplicates brain_stem content: {name}"
        for name in ("docs", "src", "substrate", "tests")
        if (ROOT / name).exists()
    )
    forbidden_runtime_files = (
        "master_parity_ledger.json",
        "brain_stem/master_parity_ledger.json",
        "run_autonomous_loop.sh",
        "run_autonomous_loop.ps1",
    )
    issues.extend(
        f"Forbidden ledger or autonomous loop artifact exists: {name}"
        for name in forbidden_runtime_files
        if (ROOT / name).exists()
    )
    return issues


def main() -> int:
    issues = evaluate_local_state()
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    import_root = os.fspath(ROOT / "brain_stem")
    prior_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (import_root, prior_python_path) if value
    )
    test_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "brain_stem/tests",
            "-p",
            "test_swarm_primitives.py",
            "-v",
        ],
        cwd=ROOT,
        env=environment,
        check=False,
    )
    if test_result.returncode != 0:
        issues.append(f"Full test suite failed with exit code {test_result.returncode}")
    if issues:
        for issue in issues:
            print(f"PARITY_FAILURE: {issue}", file=sys.stderr)
        print("ARCHITECTURE_CLEARANCE=false")
        return 1
    print("ARCHITECTURE_CLEARANCE=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())