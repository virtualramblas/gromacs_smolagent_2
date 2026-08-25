from __future__ import annotations

from pathlib import Path
from typing import Callable


# ---------------------------------------------------------------------------
# Correct patch targets — one per tool module
# ---------------------------------------------------------------------------

# Use these constants in tests instead of hardcoding the module path
GMX_TOOLS_MODULE      = "agent.tools.gmx_tools.run_gmx_command"
FILE_TOOLS_MODULE     = "agent.tools.file_tools.run_gmx_command"
ANALYSIS_TOOLS_MODULE = "agent.tools.analysis_tools.run_gmx_command"


# ---------------------------------------------------------------------------
# Generic mock factory
# ---------------------------------------------------------------------------

def make_gmx_mock(
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
    create_files: list[Path] | None = None,
) -> Callable:
    """
    Return a mock for run_gmx_command that:
        - Creates specified output files (empty, to simulate GMX output)
        - Returns (returncode, stdout, stderr)
    """
    def _mock(args, work_dir, stdin_text=None, timeout=3600):
        if create_files:
            for f in create_files:
                Path(f).parent.mkdir(parents=True, exist_ok=True)
                Path(f).write_text("mock GROMACS output")
        return returncode, stdout, stderr
    return _mock


def make_failing_mock(
    stderr: str = "Fatal error: something went wrong",
    returncode: int = 1,
) -> Callable:
    """Return a mock that simulates a GROMACS failure."""
    def _mock(args, work_dir, stdin_text=None, timeout=3600):
        return returncode, "", stderr
    return _mock


def make_warning_mock(
    create_files: list[Path],
    warning_text: str = "WARNING: 1-4 interaction not set",
) -> Callable:
    """Return a mock that succeeds but emits warnings."""
    def _mock(args, work_dir, stdin_text=None, timeout=3600):
        for f in create_files:
            Path(f).parent.mkdir(parents=True, exist_ok=True)
            Path(f).write_text("mock output")
        return 0, "", warning_text
    return _mock


def make_timeout_mock() -> Callable:
    """Return a mock that raises TimeoutExpired."""
    import subprocess
    def _mock(args, work_dir, stdin_text=None, timeout=3600):
        raise subprocess.TimeoutExpired(cmd=args, timeout=timeout)
    return _mock


def make_env_error_mock() -> Callable:
    """Return a mock that raises EnvironmentError (GMX not found)."""
    def _mock(args, work_dir, stdin_text=None, timeout=3600):
        raise EnvironmentError(
            "GROMACS binary ('gmx' or 'gmx_mpi') not found in PATH."
        )
    return _mock