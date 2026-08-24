"""
Post-simulation analysis tools wrapping gmx energy, rms, rmsf, gyrate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .base import (
    GMXBaseTool,
    GMXResult,
    run_gmx_command,
    extract_warnings,
    extract_errors,
    logger,
)


# ===========================================================================
# 11. EnergyAnalysisTool
# ===========================================================================

class EnergyAnalysisTool(GMXBaseTool):
    """
    Wraps: gmx energy

    Extracts energy terms (potential, kinetic, temperature, pressure, etc.)
    from a GROMACS .edr file and writes them to an .xvg file.
    """

    name = "energy_analysis"
    description = (
        "Extract energy terms from a .edr file using gmx energy. "
        "Inputs: edr_file, output_xvg, energy_terms (list of term names "
        "e.g. ['Potential', 'Temperature', 'Pressure']). "
        "Returns status and path to output .xvg file."
    )
    inputs = {
        "edr_file": {
            "type": "string",
            "description": "Path to GROMACS .edr energy file.",
        },
        "output_xvg": {
            "type": "string",
            "description": "Path for output .xvg file.",
            "nullable": True,
        },
        "energy_terms": {
            "type": "array",
            "description": (
                "List of energy term names to extract, "
                "e.g. ['Potential', 'Kinetic-En.', 'Temperature']."
            ),
            "nullable": True,
        },
    }
    output_type = "string"

    def _run_gmx(
        self,
        edr_file: str,
        output_xvg: Optional[str] = None,
        energy_terms: Optional[list[str]] = None,
    ) -> GMXResult:

        edr = Path(edr_file).resolve()
        out_xvg = Path(output_xvg) if output_xvg else self.work_dir / "energy.xvg"
        terms = energy_terms or ["Potential"]

        # gmx energy reads term selection from stdin, one per line, terminated by 0
        stdin_input = "\n".join(terms) + "\n0\n"

        cmd = [
            "gmx", "energy",
            "-f", str(edr),
            "-o", str(out_xvg),
        ]

        rc, stdout, stderr = run_gmx_command(
            cmd, self.work_dir, stdin_text=stdin_input
        )

        output_files = {"xvg": out_xvg}
        warnings = extract_warnings(stderr)
        errors = extract_errors(stderr)
        from .base import assert_files_exist
        missing = assert_files_exist(output_files)
        errors.extend(missing)

        success = rc == 0 and not missing

        summary = (
            f"gmx energy {'succeeded' if success else 'FAILED'} "
            f"(terms={terms}). "
            f"Output: {out_xvg}."
        )

        return GMXResult(
            success=success,
            command=" ".join(cmd),
            returncode=rc,
            stdout=stdout,
            stderr=stderr,
            output_files=output_files,
            warnings=warnings,
            errors=errors,
            summary=summary,
        )
    
    # Explicit forward()
    def forward(
        self,
        edr_file: str,
        output_xvg: Optional[str] = None,
        energy_terms: Optional[list[str]] = None,
    ) -> str:
        return self._safe_run(
            edr_file=edr_file,
            output_xvg=output_xvg,
            energy_terms=energy_terms,
        )


# ===========================================================================
# 12. RMSDAnalysisTool
# ===========================================================================

class RMSDAnalysisTool(GMXBaseTool):
    """
    Wraps: gmx rms

    Calculates RMSD of the protein backbone over the trajectory.
    """

    name = "rmsd_analysis"
    description = (
        "Calculate backbone RMSD over a trajectory using gmx rms. "
        "Inputs: tpr_file, trajectory_file (xtc or trr), output_xvg. "
        "Automatically selects backbone group for both fitting and RMSD. "
        "Returns status and path to output .xvg file."
    )
    inputs = {
        "tpr_file": {
            "type": "string",
            "description": "Path to .tpr reference structure file.",
        },
        "trajectory_file": {
            "type": "string",
            "description": "Path to trajectory .xtc or .trr file.",
        },
        "output_xvg": {
            "type": "string",
            "description": "Path for output RMSD .xvg file.",
            "nullable": True,
        },
        "group": {
            "type": "string",
            "description": "Atom group for RMSD, e.g. 'Backbone', 'C-alpha'.",
            "nullable": True,
        },
    }
    output_type = "string"

    def _run_gmx(
        self,
        tpr_file: str,
        trajectory_file: str,
        output_xvg: Optional[str] = None,
        group: str = "Backbone",
    ) -> GMXResult:

        tpr = Path(tpr_file).resolve()
        traj = Path(trajectory_file).resolve()
        out_xvg = Path(output_xvg) if output_xvg else self.work_dir / "rmsd.xvg"

        # stdin: first selection = fitting group, second = RMSD group
        stdin_input = f"{group}\n{group}\n"

        cmd = [
            "gmx", "rms",
            "-s", str(tpr),
            "-f", str(traj),
            "-o", str(out_xvg),
        ]

        rc, stdout, stderr = run_gmx_command(
            cmd, self.work_dir, stdin_text=stdin_input
        )

        output_files = {"xvg": out_xvg}
        warnings = extract_warnings(stderr)
        errors = extract_errors(stderr)
        from .base import assert_files_exist
        missing = assert_files_exist(output_files)
        errors.extend(missing)

        success = rc == 0 and not missing

        summary = (
            f"gmx rms {'succeeded' if success else 'FAILED'} "
            f"(group={group}). Output: {out_xvg}."
        )

        return GMXResult(
            success=success,
            command=" ".join(cmd),
            returncode=rc,
            stdout=stdout,
            stderr=stderr,
            output_files=output_files,
            warnings=warnings,
            errors=errors,
            summary=summary,
        )

    # Explicit forward()
    def forward(
        self,
        tpr_file: str,
        trajectory_file: str,
        output_xvg: Optional[str] = None,
        group: str = "Backbone",
    ) -> str:
        return self._safe_run(
            tpr_file=tpr_file,
            trajectory_file=trajectory_file,
            output_xvg=output_xvg,
            group=group,
        )