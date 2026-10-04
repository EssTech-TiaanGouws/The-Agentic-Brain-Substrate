# The Agentic Brain Substrate

The frozen project blueprint and live master architecture are indexed under `brain_stem/`. The master architecture remains the live source for decisions, implementation status, and verification results:

- [Frozen Charter](brain_stem/charter.md)
- [Master Architecture](brain_stem/docs/architecture_core.md)

This repository contains the local-first Agentic Brain Substrate. Install the declared runtime and model-test dependencies, then run the one-shot verification from the repository root:

```sh
python3 -m pip install -r brain_stem/requirements.txt -r brain_stem/models/stacey/requirements.txt
python3 verify_parity_gates.py
```