"""Shared pytest configuration for the ALAS unit-test suite.

Puts ``src/`` on ``sys.path`` so tests can ``import ai.*`` /
``import navigation.*`` exactly as the running system does, and exposes the
canonical fixture and output directories so every test reads inputs from one
place and writes artifacts to one place.

Layout (mirrors ``src/``):
    tests/
        fixtures/            committed input assets (images, etc.)
        ai/                  tests for src/ai
        navigation/
            local_planner/   tests for src/navigation/local_planner
            sensors/         tests for src/navigation/sensors

Test artifacts are written under ``outputs/tests/<module>/`` (git-ignored).

The standalone test scripts also work without pytest (``python tests/...``);
this file only adds convenience for ``pytest`` runs.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "tests"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def output_dir(module: str) -> Path:
    """Return (and create) the output directory for a module's test artifacts."""
    path = OUTPUT_ROOT / module
    path.mkdir(parents=True, exist_ok=True)
    return path
