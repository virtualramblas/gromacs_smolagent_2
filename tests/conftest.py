"""
Shared pytest fixtures and helpers used across all test sub-phases.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Directory fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_work_dir(tmp_path: Path) -> Path:
    """Provide a clean temporary working directory for each test."""
    work_dir = tmp_path / "gmx_run"
    work_dir.mkdir()
    return work_dir


@pytest.fixture
def tmp_mdp_file(tmp_path: Path) -> Path:
    """Write a minimal valid .mdp file and return its path."""
    mdp = tmp_path / "test.mdp"
    mdp.write_text(
        "; Test MDP\n"
        "integrator              = steep\n"
        "emtol                   = 1000.0\n"
        "emstep                  = 0.01\n"
        "nsteps                  = 5000\n"
        "nstlog                  = 500\n"
        "coulombtype             = PME\n"
        "pbc                     = xyz\n"
    )
    return mdp


@pytest.fixture
def tmp_state_file(tmp_path: Path) -> Path:
    """Return a path for a pipeline state JSON file (not yet created)."""
    return tmp_path / "pipeline_state.json"