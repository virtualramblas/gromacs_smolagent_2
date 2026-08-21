"""
Base classes and shared utilities for GROMACS SmolAgent tools.
"""

import subprocess
import logging
import shutil
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from smolagents import Tool

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("gromacs_agent")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class GMXResult:
    """
    Standardised return object for every GROMACS tool call.
    The agent reads .summary to decide next steps.
    """
    success: bool
    command: str
    returncode: int
    stdout: str
    stderr: str
    output_files: dict[str, Path] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    summary: str = ""

    def to_agent_string(self) -> str:
        """
        Serialise result to a plain string the LLM can reason about.
        Keeps token count manageable by truncating raw stdout/stderr.
        """
        lines = [
            f"SUCCESS: {self.success}",
            f"COMMAND: {self.command}",
            f"RETURN_CODE: {self.returncode}",
        ]
        if self.output_files:
            lines.append("OUTPUT_FILES:")
            for label, path in self.output_files.items():
                exists = Path(path).exists()
                lines.append(f"  {label}: {path} (exists={exists})")
        if self.warnings:
            lines.append("WARNINGS:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        if self.errors:
            lines.append("ERRORS:")
            for e in self.errors:
                lines.append(f"  - {e}")
        lines.append(f"SUMMARY: {self.summary}")
        # Truncated raw output for debugging — last 40 lines only
        if self.stderr:
            tail = "\n".join(self.stderr.splitlines()[-40:])
            lines.append(f"STDERR_TAIL:\n{tail}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Shared subprocess runner
# ---------------------------------------------------------------------------

def run_gmx_command(
    args: list[str],
    work_dir: Path,
    stdin_text: Optional[str] = None,
    timeout: int = 3600,
) -> tuple[int, str, str]:
    """
    Execute a GROMACS command in *work_dir*.

    Args:
        args:        Full command as list, e.g. ['gmx', 'pdb2gmx', '-f', ...]
        work_dir:    Working directory for the subprocess
        stdin_text:  Optional text piped to stdin (e.g. force-field index)
        timeout:     Seconds before the process is killed (default 1 h)

    Returns:
        (returncode, stdout, stderr)
    """
    gmx_bin = shutil.which("gmx") or shutil.which("gmx_mpi")
    if gmx_bin is None:
        raise EnvironmentError(
            "GROMACS binary ('gmx' or 'gmx_mpi') not found in PATH."
        )

    # Replace placeholder 'gmx' with the resolved binary path
    if args[0] == "gmx":
        args = [gmx_bin] + args[1:]

    logger.info("Running: %s  (cwd=%s)", " ".join(args), work_dir)

    result = subprocess.run(
        args,
        cwd=work_dir,
        input=stdin_text,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    logger.debug("STDOUT: %s", result.stdout[:500])
    logger.debug("STDERR: %s", result.stderr[:500])

    return result.returncode, result.stdout, result.stderr


# ---------------------------------------------------------------------------
# Log parser helpers  (used by multiple tools)
# ---------------------------------------------------------------------------

def extract_warnings(text: str) -> list[str]:
    """Pull WARNING lines from GROMACS stderr/stdout."""
    return [
        line.strip()
        for line in text.splitlines()
        if "WARNING" in line.upper() or "warning" in line.lower()
    ]


def extract_errors(text: str) -> list[str]:
    """Pull ERROR / Fatal error lines from GROMACS stderr/stdout."""
    return [
        line.strip()
        for line in text.splitlines()
        if any(kw in line for kw in ("ERROR", "Fatal error", "fatal error", "Error"))
    ]


def assert_files_exist(files: dict[str, Path]) -> list[str]:
    """
    Return a list of missing-file error strings.
    Empty list means all files are present.
    """
    missing = []
    for label, path in files.items():
        if not Path(path).exists():
            missing.append(f"Expected output file not found: {label} -> {path}")
    return missing


# ---------------------------------------------------------------------------
# Base Tool
# ---------------------------------------------------------------------------

class GMXBaseTool(Tool):
    """
    Abstract base for all GROMACS tools.

    Subclasses must implement:
        _run_gmx(**kwargs) -> GMXResult

    The public forward() method calls _run_gmx and always returns
    a plain string so SmolAgent's CodeAgent can parse it.
    """

    # SmolAgent requires these class-level attributes
    name: str = "gmx_base_tool"
    description: str = "Base GROMACS tool — do not use directly."
    inputs: dict = {}
    output_type: str = "string"

    def __init__(self, work_dir: str | Path = "."):
        super().__init__()
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Subclasses override this
    # ------------------------------------------------------------------

    def _run_gmx(self, **kwargs) -> GMXResult:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # SmolAgent entry point
    # ------------------------------------------------------------------

    def forward(self, **kwargs) -> str:
        try:
            result = self._run_gmx(**kwargs)
        except EnvironmentError as exc:
            return GMXResult(
                success=False,
                command="",
                returncode=-1,
                stdout="",
                stderr=str(exc),
                errors=[str(exc)],
                summary=f"Environment error: {exc}",
            ).to_agent_string()
        except subprocess.TimeoutExpired:
            return GMXResult(
                success=False,
                command="",
                returncode=-1,
                stdout="",
                stderr="Process timed out.",
                errors=["Process timed out."],
                summary="GMX command exceeded timeout limit.",
            ).to_agent_string()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error in tool %s", self.name)
            return GMXResult(
                success=False,
                command="",
                returncode=-1,
                stdout="",
                stderr=str(exc),
                errors=[str(exc)],
                summary=f"Unexpected error: {exc}",
            ).to_agent_string()

        return result.to_agent_string()