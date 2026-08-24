"""
GROMACS Agentic MD Pipeline — Entry Point

Usage:
    python run.py eiwit.pdb
    python run.py eiwit.pdb --config config.yaml
    python run.py eiwit.pdb --config config.yaml --resume
    python run.py eiwit.pdb --ff charmm36m --water tip4p --threads 8
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from agent.orchestrator import build_agent, load_config
from agent.tools.state_tools import PipelineStateTool


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO", log_file: str = "agent_run.log") -> None:
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="a"),
    ]
    logging.basicConfig(level=getattr(logging, level.upper()), format=fmt,
                        handlers=handlers)


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agentic GROMACS MD pipeline powered by SmolAgents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "pdb_file",
        type=str,
        help="Path to input .pdb file.",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to config.yaml.",
    )
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Resume from existing pipeline state if available.",
    )
    parser.add_argument(
        "--ff",
        type=str,
        default=None,
        help="Force field override, e.g. 'charmm36m', 'oplsaa'.",
    )
    parser.add_argument(
        "--water",
        type=str,
        default=None,
        help="Water model override, e.g. 'tip4p', 'spc'.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of CPU threads for mdrun.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable GPU offloading for mdrun.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the agent task prompt and exit without running.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Task prompt builder
# ---------------------------------------------------------------------------

def build_task_prompt(
    pdb_file: str,
    config: dict,
    resume: bool,
    overrides: dict,
) -> str:
    """
    Construct the natural-language task string passed to agent.run().
    Merges config defaults with CLI overrides.
    """
    sim = config.get("simulation", {})
    pipe = config.get("pipeline", {})

    ff       = overrides.get("ff")      or sim.get("force_field",      "amber99sb-ildn")
    water    = overrides.get("water")   or sim.get("water_model",       "tip3p")
    threads  = overrides.get("threads") or sim.get("n_threads",         4)
    use_gpu  = overrides.get("gpu")     or sim.get("use_gpu",           False)
    box_type = sim.get("box_type",       "dodecahedron")
    box_d    = sim.get("box_distance",   1.0)
    conc     = sim.get("salt_concentration", 0.15)
    temp     = sim.get("temperature",    300)
    work_dir = pipe.get("work_dir",      "gmx_run")
    tmpl_dir = pipe.get("templates_dir", "mdp_templates")
    max_rec  = pipe.get("max_recovery_attempts", 3)

    pdb_path = Path(pdb_file).resolve()

    resume_instruction = (
        "\nIMPORTANT: A previous run was detected. "
        "Start by calling pipeline_state(action='read') to check completed steps "
        "and resume from where the pipeline left off."
        if resume
        else
        "\nThis is a fresh run. "
        "Start by calling pipeline_state(action='reset') to initialise state."
    )

    return f"""
Run a complete GROMACS MD simulation pipeline for the protein in:
    {pdb_path}

## Configuration
- Working directory : {work_dir}
- MDP templates dir : {tmpl_dir}
- Force field       : {ff}
- Water model       : {water}
- Box type          : {box_type}
- Box distance      : {box_d} nm
- Salt concentration: {conc} mol/L
- Temperature       : {temp} K
- CPU threads       : {threads}
- GPU offloading    : {use_gpu}
- Max recovery tries: {max_rec}

## Instructions
{resume_instruction}

Follow the pipeline order defined in your system prompt exactly.
After each step:
  1. Check SUCCESS field in tool output.
  2. If a simulation step, call parse_gmx_log on the output .log file.
  3. Update pipeline state with completed step and output file paths.
  4. If SEVERITY is RECOVERABLE, apply MDP_PATCHES and retry (max {max_rec} times).
  5. If SEVERITY is FATAL or NEEDS_HUMAN_REVIEW, stop and report clearly.

Use MDP templates from {tmpl_dir}/ as starting points.
Copy each template to {work_dir}/ before modifying it.

When the full pipeline completes, provide a structured summary report
covering all steps, output files, convergence metrics, and any issues
encountered and resolved.
""".strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args   = parse_args()
    config = load_config(args.config)

    log_cfg = config.get("logging", {})
    setup_logging(
        level=log_cfg.get("level", "INFO"),
        log_file=log_cfg.get("log_file", "agent_run.log"),
    )
    logger = logging.getLogger("gromacs_agent.run")

    pdb_path = Path(args.pdb_file)
    if not pdb_path.exists():
        logger.error("Input PDB file not found: %s", pdb_path)
        return 1

    overrides: dict = {}
    if args.ff:      overrides["ff"]      = args.ff
    if args.water:   overrides["water"]   = args.water
    if args.threads: overrides["threads"] = args.threads
    if args.gpu:     overrides["gpu"]     = True

    try:
        agent, _ = build_agent(config)          # needs_prepend always False

        task_prompt = build_task_prompt(
            pdb_file=str(pdb_path),
            config=config,
            resume=args.resume,
            overrides=overrides,
            # prepend_system_prompt and system_prompt args
            # can be removed from build_task_prompt entirely
        )

        if args.dry_run:
            print("=" * 70)
            print("DRY RUN — Task prompt:")
            print("=" * 70)
            print(task_prompt)
            return 0

        logger.info("Starting pipeline for: %s", pdb_path)
        result = agent.run(task_prompt)

        print("\n" + "=" * 70)
        print("AGENT PIPELINE COMPLETE")
        print("=" * 70)
        print(result)
        return 0

    except KeyboardInterrupt:
        logger.warning("Interrupted. Run with --resume to continue.")
        return 130

    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        return 1

if __name__ == "__main__":
    sys.exit(main())