"""
GROMACS pipeline tools — one SmolAgent Tool per GMX command.

Pipeline order:
    Pdb2GmxTool → EditconfTool → SolvateTool → GenionTool
    → GromppTool → MdrunTool
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from smolagents import tool   # used for the @tool decorator approach where simpler

from .base import (
    GMXBaseTool,
    GMXResult,
    run_gmx_command,
    extract_warnings,
    extract_errors,
    assert_files_exist,
    logger,
)


# ===========================================================================
# 1. Pdb2GmxTool
# ===========================================================================

class Pdb2GmxTool(GMXBaseTool):

    name = "pdb2gmx"
    description = (
        "Convert a PDB file to GROMACS format. "
        "Generates conf.gro (coordinates), topol.top (topology), "
        "and posre.itp (position restraints). "
        "Inputs: pdb_file, force_field (e.g. 'amber99sb-ildn'), "
        "water_model (e.g. 'tip3p'), output_gro, output_top. "
        "Returns a status string with output file paths and any warnings."
    )
    inputs = {
        "pdb_file": {
            "type": "string",
            "description": "Path to the input .pdb file.",
        },
        "force_field": {
            "type": "string",
            "description": "GROMACS force field name, e.g. 'amber99sb-ildn'.",
            "nullable": True,
        },
        "water_model": {
            "type": "string",
            "description": "Water model, e.g. 'tip3p', 'tip4p', 'spc'.",
            "nullable": True,
        },
        "output_gro": {
            "type": "string",
            "description": "Path for output .gro coordinate file.",
            "nullable": True,
        },
        "output_top": {
            "type": "string",
            "description": "Path for output topology .top file.",
            "nullable": True,
        },
        "ignore_hydrogens": {
            "type": "boolean",
            "description": "Pass -ignh flag to strip existing H atoms.",
            "nullable": True,
        },
    }
    output_type = "string"

    # ---- Explicit forward() — signature matches inputs keys exactly ----
    def forward(
        self,
        pdb_file: str,
        force_field: str = "amber99sb-ildn",
        water_model: str = "tip3p",
        output_gro: Optional[str] = None,
        output_top: Optional[str] = None,
        ignore_hydrogens: bool = True,
    ) -> str:
        return self._safe_run(
            pdb_file=pdb_file,
            force_field=force_field,
            water_model=water_model,
            output_gro=output_gro,
            output_top=output_top,
            ignore_hydrogens=ignore_hydrogens,
        )

    def _run_gmx(
        self,
        pdb_file: str,
        force_field: str = "amber99sb-ildn",
        water_model: str = "tip3p",
        output_gro: Optional[str] = None,
        output_top: Optional[str] = None,
        ignore_hydrogens: bool = True,
    ) -> GMXResult:

        pdb_path = Path(pdb_file).resolve()
        out_gro  = Path(output_gro) if output_gro else self.work_dir / "conf.gro"
        out_top  = Path(output_top) if output_top else self.work_dir / "topol.top"
        out_posre = self.work_dir / "posre.itp"

        cmd = [
            "gmx", "pdb2gmx",
            "-f", str(pdb_path),
            "-o", str(out_gro),
            "-p", str(out_top),
            "-ff", force_field,
            "-water", water_model,
        ]
        if ignore_hydrogens:
            cmd.append("-ignh")

        rc, stdout, stderr = run_gmx_command(cmd, self.work_dir)

        output_files = {"gro": out_gro, "top": out_top, "posre": out_posre}
        warnings = extract_warnings(stderr)
        errors   = extract_errors(stderr)
        missing  = assert_files_exist({"gro": out_gro, "top": out_top})
        errors.extend(missing)
        success = rc == 0 and not missing

        return GMXResult(
            success=success,
            command=" ".join(cmd),
            returncode=rc,
            stdout=stdout,
            stderr=stderr,
            output_files=output_files,
            warnings=warnings,
            errors=errors,
            summary=(
                f"pdb2gmx {'succeeded' if success else 'FAILED'} "
                f"(ff={force_field}, water={water_model}). "
                f"{len(warnings)} warning(s), {len(errors)} error(s)."
            ),
        )


# ===========================================================================
# 2. EditconfTool
# ===========================================================================

class EditconfTool(GMXBaseTool):

    name = "editconf"
    description = (
        "Define the simulation box around the protein using gmx editconf. "
        "Inputs: input_gro, output_gro, box_type ('cubic', 'dodecahedron', "
        "'triclinic'), distance (nm from protein to box edge, e.g. 1.0). "
        "Returns status string with output file path."
    )
    inputs = {
        "input_gro": {
            "type": "string",
            "description": "Path to input .gro file from pdb2gmx.",
        },
        "output_gro": {
            "type": "string",
            "description": "Path for output .gro file with box defined.",
            "nullable": True,
        },
        "box_type": {
            "type": "string",
            "description": (
                "Box geometry: 'cubic', 'dodecahedron', or 'triclinic'."
            ),
            "nullable": True,
        },
        "distance": {
            "type": "number",
            "description": "Minimum distance in nm between protein and box edge.",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(
        self,
        input_gro: str,
        output_gro: Optional[str] = None,
        box_type: str = "dodecahedron",
        distance: float = 1.0,
    ) -> str:
        return self._safe_run(
            input_gro=input_gro,
            output_gro=output_gro,
            box_type=box_type,
            distance=distance,
        )

    def _run_gmx(
        self,
        input_gro: str,
        output_gro: Optional[str] = None,
        box_type: str = "dodecahedron",
        distance: float = 1.0,
    ) -> GMXResult:

        in_gro  = Path(input_gro).resolve()
        out_gro = Path(output_gro) if output_gro else self.work_dir / "conf_box.gro"

        cmd = [
            "gmx", "editconf",
            "-f", str(in_gro),
            "-o", str(out_gro),
            "-bt", box_type,
            "-d", str(distance),
        ]

        rc, stdout, stderr = run_gmx_command(cmd, self.work_dir)

        output_files = {"gro": out_gro}
        warnings = extract_warnings(stderr)
        errors   = extract_errors(stderr)
        missing  = assert_files_exist(output_files)
        errors.extend(missing)
        success = rc == 0 and not missing

        return GMXResult(
            success=success,
            command=" ".join(cmd),
            returncode=rc,
            stdout=stdout,
            stderr=stderr,
            output_files=output_files,
            warnings=warnings,
            errors=errors,
            summary=(
                f"editconf {'succeeded' if success else 'FAILED'} "
                f"(box={box_type}, d={distance} nm). "
                f"{len(warnings)} warning(s), {len(errors)} error(s)."
            ),
        )


# ===========================================================================
# 3. SolvateTool
# ===========================================================================

class SolvateTool(GMXBaseTool):

    name = "solvate"
    description = (
        "Solvate the protein box with water using gmx solvate. "
        "Inputs: input_gro (boxed protein), topology_top, output_gro. "
        "The topology file is updated in-place with solvent molecule count. "
        "Returns status string with output file paths."
    )
    inputs = {
        "input_gro": {
            "type": "string",
            "description": "Path to boxed protein .gro file (from editconf).",
        },
        "topology_top": {
            "type": "string",
            "description": "Path to topology .top file (updated in-place).",
        },
        "output_gro": {
            "type": "string",
            "description": "Path for solvated output .gro file.",
            "nullable": True,
        },
        "solvent_model": {
            "type": "string",
            "description": "Solvent configuration file, default 'spc216.gro'.",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(
        self,
        input_gro: str,
        topology_top: str,
        output_gro: Optional[str] = None,
        solvent_model: str = "spc216.gro",
    ) -> str:
        return self._safe_run(
            input_gro=input_gro,
            topology_top=topology_top,
            output_gro=output_gro,
            solvent_model=solvent_model,
        )

    def _run_gmx(
        self,
        input_gro: str,
        topology_top: str,
        output_gro: Optional[str] = None,
        solvent_model: str = "spc216.gro",
    ) -> GMXResult:

        in_gro  = Path(input_gro).resolve()
        top     = Path(topology_top).resolve()
        out_gro = Path(output_gro) if output_gro else self.work_dir / "conf_solv.gro"

        cmd = [
            "gmx", "solvate",
            "-cp", str(in_gro),
            "-cs", solvent_model,
            "-o",  str(out_gro),
            "-p",  str(top),
        ]

        rc, stdout, stderr = run_gmx_command(cmd, self.work_dir)

        warnings = extract_warnings(stderr)
        errors   = extract_errors(stderr)
        missing  = assert_files_exist({"gro": out_gro})
        errors.extend(missing)
        success = rc == 0 and not missing

        n_water = "unknown"
        for line in (stdout + stderr).splitlines():
            if "Number of solvent molecules" in line or "SOL" in line:
                n_water = line.strip()
                break

        return GMXResult(
            success=success,
            command=" ".join(cmd),
            returncode=rc,
            stdout=stdout,
            stderr=stderr,
            output_files={"gro": out_gro, "top": top},
            warnings=warnings,
            errors=errors,
            summary=(
                f"solvate {'succeeded' if success else 'FAILED'}. "
                f"Solvent info: {n_water}. "
                f"{len(warnings)} warning(s), {len(errors)} error(s)."
            ),
        )


# ===========================================================================
# 4. GenionTool
# ===========================================================================

class GenionTool(GMXBaseTool):

    name = "genion"
    description = (
        "Add ions to neutralise the system using gmx genion. "
        "Requires a .tpr file (built from ions.mdp via grompp). "
        "Inputs: input_tpr, output_gro, topology_top, "
        "concentration (mol/L), neutral (bool). "
        "Automatically selects the SOL group for ion replacement. "
        "Returns status string."
    )
    inputs = {
        "input_tpr": {
            "type": "string",
            "description": "Path to .tpr file prepared for genion step.",
        },
        "topology_top": {
            "type": "string",
            "description": "Path to topology .top file (updated in-place).",
        },
        "output_gro": {
            "type": "string",
            "description": "Path for output .gro file with ions added.",
            "nullable": True,
        },
        "concentration": {
            "type": "number",
            "description": "Salt concentration in mol/L (e.g. 0.15).",
            "nullable": True,
        },
        "neutral": {
            "type": "boolean",
            "description": "Whether to add ions to neutralise net charge.",
            "nullable": True,
        },
        "positive_ion": {
            "type": "string",
            "description": "Positive ion type, e.g. 'NA' (default) or 'K'.",
            "nullable": True,
        },
        "negative_ion": {
            "type": "string",
            "description": "Negative ion type, e.g. 'CL' (default).",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(
        self,
        input_tpr: str,
        topology_top: str,
        output_gro: Optional[str] = None,
        concentration: float = 0.15,
        neutral: bool = True,
        positive_ion: str = "NA",
        negative_ion: str = "CL",
    ) -> str:
        return self._safe_run(
            input_tpr=input_tpr,
            topology_top=topology_top,
            output_gro=output_gro,
            concentration=concentration,
            neutral=neutral,
            positive_ion=positive_ion,
            negative_ion=negative_ion,
        )

    def _run_gmx(
        self,
        input_tpr: str,
        topology_top: str,
        output_gro: Optional[str] = None,
        concentration: float = 0.15,
        neutral: bool = True,
        positive_ion: str = "NA",
        negative_ion: str = "CL",
    ) -> GMXResult:

        in_tpr  = Path(input_tpr).resolve()
        top     = Path(topology_top).resolve()
        out_gro = Path(output_gro) if output_gro else self.work_dir / "conf_ions.gro"

        cmd = [
            "gmx", "genion",
            "-s",      str(in_tpr),
            "-o",      str(out_gro),
            "-p",      str(top),
            "-pname",  positive_ion,
            "-nname",  negative_ion,
            "-conc",   str(concentration),
        ]
        if neutral:
            cmd.append("-neutral")

        rc, stdout, stderr = run_gmx_command(
            cmd, self.work_dir, stdin_text="SOL\n"
        )

        warnings = extract_warnings(stderr)
        errors   = extract_errors(stderr)
        missing  = assert_files_exist({"gro": out_gro})
        errors.extend(missing)
        success = rc == 0 and not missing

        return GMXResult(
            success=success,
            command=" ".join(cmd),
            returncode=rc,
            stdout=stdout,
            stderr=stderr,
            output_files={"gro": out_gro, "top": top},
            warnings=warnings,
            errors=errors,
            summary=(
                f"genion {'succeeded' if success else 'FAILED'} "
                f"(conc={concentration} M, neutral={neutral}). "
                f"{len(warnings)} warning(s), {len(errors)} error(s)."
            ),
        )


# ===========================================================================
# 5. GromppTool
# ===========================================================================

class GromppTool(GMXBaseTool):

    name = "grompp"
    description = (
        "Run gmx grompp to generate a .tpr run input file. "
        "Inputs: mdp_file, input_gro, topology_top, output_tpr, "
        "index_file (optional), max_warnings (int, default 0). "
        "Returns status string with .tpr path and any warnings/errors."
    )
    inputs = {
        "mdp_file": {
            "type": "string",
            "description": "Path to .mdp simulation parameter file.",
        },
        "input_gro": {
            "type": "string",
            "description": "Path to input coordinate .gro file.",
        },
        "topology_top": {
            "type": "string",
            "description": "Path to topology .top file.",
        },
        "output_tpr": {
            "type": "string",
            "description": "Path for output .tpr file.",
            "nullable": True,
        },
        "index_file": {
            "type": "string",
            "description": "Optional path to index .ndx file.",
            "nullable": True,
        },
        "checkpoint_file": {
            "type": "string",
            "description": "Optional .cpt checkpoint for continuation runs.",
            "nullable": True,
        },
        "max_warnings": {
            "type": "integer",
            "description": "Maximum allowed grompp warnings. Default 0.",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(
        self,
        mdp_file: str,
        input_gro: str,
        topology_top: str,
        output_tpr: Optional[str] = None,
        index_file: Optional[str] = None,
        checkpoint_file: Optional[str] = None,
        max_warnings: int = 0,
    ) -> str:
        return self._safe_run(
            mdp_file=mdp_file,
            input_gro=input_gro,
            topology_top=topology_top,
            output_tpr=output_tpr,
            index_file=index_file,
            checkpoint_file=checkpoint_file,
            max_warnings=max_warnings,
        )

    def _run_gmx(
        self,
        mdp_file: str,
        input_gro: str,
        topology_top: str,
        output_tpr: Optional[str] = None,
        index_file: Optional[str] = None,
        checkpoint_file: Optional[str] = None,
        max_warnings: int = 0,
    ) -> GMXResult:

        mdp     = Path(mdp_file).resolve()
        gro     = Path(input_gro).resolve()
        top     = Path(topology_top).resolve()
        out_tpr = Path(output_tpr) if output_tpr else self.work_dir / "topol.tpr"

        cmd = [
            "gmx", "grompp",
            "-f",       str(mdp),
            "-c",       str(gro),
            "-p",       str(top),
            "-o",       str(out_tpr),
            "-maxwarn", str(max_warnings),
        ]
        if index_file:
            cmd += ["-n", str(Path(index_file).resolve())]
        if checkpoint_file:
            cmd += ["-t", str(Path(checkpoint_file).resolve())]

        rc, stdout, stderr = run_gmx_command(cmd, self.work_dir)

        warnings = extract_warnings(stderr)
        errors   = extract_errors(stderr)
        missing  = assert_files_exist({"tpr": out_tpr})
        errors.extend(missing)
        success = rc == 0 and not missing

        return GMXResult(
            success=success,
            command=" ".join(cmd),
            returncode=rc,
            stdout=stdout,
            stderr=stderr,
            output_files={"tpr": out_tpr},
            warnings=warnings,
            errors=errors,
            summary=(
                f"grompp {'succeeded' if success else 'FAILED'} "
                f"(mdp={mdp.name}, maxwarn={max_warnings}). "
                f"{len(warnings)} warning(s), {len(errors)} error(s)."
            ),
        )


# ===========================================================================
# 6. MdrunTool
# ===========================================================================

class MdrunTool(GMXBaseTool):

    name = "mdrun"
    description = (
        "Run a GROMACS simulation using gmx mdrun. "
        "Inputs: input_tpr, run_label (prefix for output files), "
        "n_threads (int), use_gpu (bool), checkpoint_file (for continuation), "
        "extra_flags (list of additional gmx mdrun flags). "
        "Returns status string with output file paths (xtc, edr, log, cpt)."
    )
    inputs = {
        "input_tpr": {
            "type": "string",
            "description": "Path to .tpr run input file.",
        },
        "run_label": {
            "type": "string",
            "description": "Output file prefix, e.g. 'em', 'nvt', 'npt', 'md'.",
            "nullable": True,
        },
        "n_threads": {
            "type": "integer",
            "description": "Number of CPU threads. Default: 4.",
            "nullable": True,
        },
        "use_gpu": {
            "type": "boolean",
            "description": "Whether to offload non-bonded calculations to GPU.",
            "nullable": True,
        },
        "checkpoint_file": {
            "type": "string",
            "description": "Path to .cpt file for continuation run.",
            "nullable": True,
        },
        "extra_flags": {
            "type": "array",
            "description": "Additional mdrun flags as a list of strings.",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(
        self,
        input_tpr: str,
        run_label: str = "md",
        n_threads: int = 4,
        use_gpu: bool = False,
        checkpoint_file: Optional[str] = None,
        extra_flags: Optional[list[str]] = None,
    ) -> str:
        return self._safe_run(
            input_tpr=input_tpr,
            run_label=run_label,
            n_threads=n_threads,
            use_gpu=use_gpu,
            checkpoint_file=checkpoint_file,
            extra_flags=extra_flags,
        )

    def _run_gmx(
        self,
        input_tpr: str,
        run_label: str = "md",
        n_threads: int = 4,
        use_gpu: bool = False,
        checkpoint_file: Optional[str] = None,
        extra_flags: Optional[list[str]] = None,
    ) -> GMXResult:

        in_tpr  = Path(input_tpr).resolve()
        out_dir = self.work_dir

        cmd = [
            "gmx", "mdrun",
            "-s",      str(in_tpr),
            "-deffnm", run_label,
            "-ntmpi",  "1",
            "-ntomp",  str(n_threads),
        ]
        if use_gpu:
            cmd += ["-nb", "gpu"]
        if checkpoint_file:
            cmd += ["-cpi", str(Path(checkpoint_file).resolve())]
        if extra_flags:
            cmd.extend(extra_flags)

        rc, stdout, stderr = run_gmx_command(
            cmd, out_dir, timeout=86400
        )

        output_files = {
            "xtc": out_dir / f"{run_label}.xtc",
            "trr": out_dir / f"{run_label}.trr",
            "edr": out_dir / f"{run_label}.edr",
            "log": out_dir / f"{run_label}.log",
            "cpt": out_dir / f"{run_label}.cpt",
            "gro": out_dir / f"{run_label}.gro",
        }

        warnings = extract_warnings(stderr)
        errors   = extract_errors(stderr)
        missing  = assert_files_exist({
            "edr": output_files["edr"],
            "log": output_files["log"],
        })
        errors.extend(missing)
        success = rc == 0 and not missing

        return GMXResult(
            success=success,
            command=" ".join(cmd),
            returncode=rc,
            stdout=stdout,
            stderr=stderr,
            output_files=output_files,
            warnings=warnings,
            errors=errors,
            summary=(
                f"mdrun {'succeeded' if success else 'FAILED'} "
                f"(label={run_label}, threads={n_threads}, gpu={use_gpu}). "
                f"{len(warnings)} warning(s), {len(errors)} error(s)."
            ),
        )