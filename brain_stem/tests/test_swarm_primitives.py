from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def load_tests(
    loader: unittest.TestLoader,
    standard_tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    cases_directory = Path(__file__).parent / "cases"
    suites: list[unittest.TestSuite] = []
    for index, case_path in enumerate(sorted(cases_directory.glob("*_cases.py"))):
        module_name = f"_swarm_primitive_cases_{index}"
        spec = importlib.util.spec_from_file_location(module_name, case_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load test cases from {case_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        suites.append(loader.loadTestsFromModule(module))
    return unittest.TestSuite(suites)


if __name__ == "__main__":
    unittest.main()