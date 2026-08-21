"""
File I/O, structure validation, and MDP management tools.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from smolagents import Tool

from .base import (
    GMXBaseTool,
    GMXResult,
    run_gmx_command,
    extract_warnings,
    extract_errors,
    logger,
)


# ===========================================================================
# 7. ReadFileTool
# ===========================================================================

class ReadFileTool(Tool):
    """
    Read text files relevant to the GROMACS pipeline.
    Supports .gro, .top, .mdp, .log, .itp files.
    Returns file content (truncated if large).
    """

    name = "read_file"
    description = (
        "Read the content of a text file (e.g. .gro, .top, .mdp, .log). "
        "Input: file_path (str), max_lines (int, default 200). "
        "Returns file content as a string, truncated to max_lines."
    )
    inputs = {
        "file_path": {
            "type": "string",
            "description": "Path to the file to read.",
        },
        "max_lines": {
            "type": "integer",
            "description": "Maximum number of lines to return. Default: 200.",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, file_path: str, max_lines: int = 200) -> str:
        path = Path(file_path)
        if not path.exists():
            return f"ERROR: File not found: {file_path}"
        try:
            lines = path.read_text(errors="replace").splitlines()
            total = len(lines)
            truncated = lines[:max_lines]
            result = "\n".join(truncated)
            if total > max_lines:
                result += f"\n... [truncated: showing {max_lines}/{total} lines]"
            return result
        except Exception as exc:  # noqa: BLE001
            return f"ERROR reading {file_path}: {exc}"


# ===========================================================================
# 8. WriteFileTool
# ===========================================================================

class WriteFileTool(Tool):
    """
    Write or overwrite a text file.
    Primary use: create or modify .mdp parameter files.
    """

    name = "write_file"
    description = (
        "Write text content to a file. Used to create or modify .mdp files. "
        "Inputs: file_path (str), content (str). "
        "Returns confirmation or error string."
    )
    inputs = {
        "file_path": {
            "type": "string",
            "description": "Destination file path.",
        },
        "content": {
            "type": "string",
            "description": "Full text content to write to the file.",
        },
    }
    output_type = "string"

    def forward(self, file_path: str, content: str) -> str:
        path = Path(file_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            return f"SUCCESS: Written {len(content)} characters to {file_path}"
        except Exception as exc:  # noqa: BLE001
            return f"ERROR writing {file_path}: {exc}"


# ===========================================================================
# 9. ValidateStructureTool
# ===========================================================================

class ValidateStructureTool(GMXBaseTool):
    """
    Wraps: gmx check

    Performs basic structural validation on .gro or .pdb files.
    Also checks topology consistency.
    """

    name = "validate_structure"
    description = (
        "Validate a GROMACS structure or trajectory file using gmx check. "
        "Input: input_file (path to .gro, .tpr, or .xtc). "
        "Returns a summary of any structural issues found."
    )
    inputs = {
        "input_file": {
            "type": "string",
            "description": "Path to structure or trajectory file to validate.",
        },
    }
    output_type = "string"

    def _run_gmx(self, input_file: str) -> GMXResult:
        in_file = Path(input_file).resolve()

        cmd = ["gmx", "check", "-f", str(in_file)]
        rc, stdout, stderr = run_gmx_command(cmd, self.work_dir)

        combined = stdout + stderr
        warnings = extract_warnings(combined)
        errors = extract_errors(combined)

        success = rc == 0

        summary = (
            f"Structure check {'passed' if success else 'FAILED'} "
            f"for {in_file.name}. "
            f"{len(warnings)} warning(s), {len(errors)} error(s)."
        )

        return GMXResult(
            success=success,
            command=" ".join(cmd),
            returncode=rc,
            stdout=stdout,
            stderr=stderr,
            warnings=warnings,
            errors=errors,
            summary=summary,
        )


# ===========================================================================
# 10. ParseGmxLogTool
# ===========================================================================

class ParseGmxLogTool(Tool):
    """
    Parse GROMACS .log files to extract key simulation metrics.
    Critical for the agent to assess convergence and decide next steps.
    """

    name = "parse_gmx_log"
    description = (
        "Parse a GROMACS .log file to extract convergence metrics, "
        "performance data, warnings, and errors. "
        "Input: log_file (path to .log). "
        "Returns structured summary string with key metrics."
    )
    inputs = {
        "log_file": {
            "type": "string",
            "description": "Path to GROMACS .log file.",
        },
    }
    output_type = "string"

    # Patterns for key GROMACS log entries
    _EM_CONVERGED   = re.compile(r"converged to Fmax\s*=\s*([\d.eE+\-]+)", re.I)
    _EM_NOT_CONV    = re.compile(r"did not converge", re.I)
    _EPOT           = re.compile(r"Potential Energy\s*=\s*([\d.eE+\-]+)")
    _FMAX           = re.compile(r"Maximum force\s*=\s*([\d.eE+\-]+)")
    _PERFORMANCE    = re.compile(r"Performance:\s*([\d.]+)\s*ns/day")
    _STEP           = re.compile(r"Step\s+(\d+),\s+time\s+([\d.]+)")

    def forward(self, log_file: str) -> str:
        path = Path(log_file)
        if not path.exists():
            return f"ERROR: Log file not found: {log_file}"

        text = path.read_text(errors="replace")
        lines = text.splitlines()

        results: dict = {
            "file": str(path),
            "em_converged": None,
            "final_epot": None,
            "final_fmax": None,
            "performance_ns_per_day": None,
            "last_step": None,
            "warnings": [],
            "errors": [],
            "fatal": False,
        }

        for line in lines:
            if m := self._EM_CONVERGED.search(line):
                results["em_converged"] = True
                results["final_fmax"] = float(m.group(1))
            if self._EM_NOT_CONV.search(line):
                results["em_converged"] = False
            if m := self._EPOT.search(line):
                results["final_epot"] = float(m.group(1))
            if m := self._FMAX.search(line):
                results["final_fmax"] = float(m.group(1))
            if m := self._PERFORMANCE.search(line):
                results["performance_ns_per_day"] = float(m.group(1))
            if m := self._STEP.search(line):
                results["last_step"] = int(m.group(1))
            if "WARNING" in line:
                results["warnings"].append(line.strip())
            if "Fatal error" in line or "ERROR" in line:
                results["errors"].append(line.strip())
                if "Fatal error" in line:
                    results["fatal"] = True

        # Deduplicate
        results["warnings"] = list(dict.fromkeys(results["warnings"]))
        results["errors"]   = list(dict.fromkeys(results["errors"]))

        # Build human-readable summary
        lines_out = [f"LOG PARSE RESULTS: {path.name}"]
        lines_out.append(f"  EM converged      : {results['em_converged']}")
        lines_out.append(f"  Final Epot        : {results['final_epot']}")
        lines_out.append(f"  Final Fmax        : {results['final_fmax']}")
        lines_out.append(f"  Last step         : {results['last_step']}")
        lines_out.append(f"  Performance ns/day: {results['performance_ns_per_day']}")
        lines_out.append(f"  Fatal error       : {results['fatal']}")
        if results["warnings"]:
            lines_out.append(f"  Warnings ({len(results['warnings'])}):")
            for w in results["warnings"][:10]:
                lines_out.append(f"    - {w}")
        if results["errors"]:
            lines_out.append(f"  Errors ({len(results['errors'])}):")
            for e in results["errors"][:10]:
                lines_out.append(f"    - {e}")

        return "\n".join(lines_out)