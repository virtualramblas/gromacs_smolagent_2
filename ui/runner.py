"""
UI-7: Runner Integration

Connects the Gradio UI to the GROMACS agent.
Runs the agent in a background thread so the UI stays responsive.

Key responsibilities:
    1. Build the agent config from UIState
    2. Install logging bridge (UILogHandler)
    3. Install tool call interceptor (captures step events for progress tab)
    4. Build and run the CodeAgent
    5. Parse tool outputs to update results (energy, RMSD, metrics)
    6. Handle stop requests gracefully
    7. Update UIState.status on completion / failure
"""

from __future__ import annotations

import json
import logging
import re
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from ui.log_handler      import install_ui_log_handler
from ui.progress_tracker import ProgressTracker
from ui.results_writer   import ResultsWriter
from ui.state            import RecoveryRecord, RunStatus, UIState

logger = logging.getLogger("gromacs_ui.runner")


# ---------------------------------------------------------------------------
# Task prompt template
# ---------------------------------------------------------------------------

def _build_task_prompt(ui_state: UIState) -> str:
    """
    Build the task prompt sent to the agent.
    Includes all simulation parameters so the agent does not
    need to ask for them.
    """
    work_dir = ui_state.work_dir
    pdb_path = ui_state.pdb_path

    return f"""
Run a complete GROMACS MD simulation pipeline for the following system.

## Input

- PDB file:          {pdb_path}
- Working directory: {work_dir}

## Simulation Parameters

- Force field:        {ui_state.force_field}
- Water model:        {ui_state.water_model}
- Box type:           {ui_state.box_type}
- Box distance:       {ui_state.box_distance} nm
- Temperature:        300 K
- Pressure:           1.0 bar
- CPU threads:        {ui_state.n_threads}
- GPU:                {"yes" if ui_state.use_gpu else "no"}

## MDP Templates

All MDP files are located in: {_find_mdp_dir()}

Use these exact MDP files:
- Energy minimisation: {_find_mdp_dir()}/em.mdp
- Ions preparation:    {_find_mdp_dir()}/ions.mdp
- NVT equilibration:   {_find_mdp_dir()}/nvt.mdp
- NPT equilibration:   {_find_mdp_dir()}/npt.mdp
- Production MD:       {_find_mdp_dir()}/md.mdp

## Instructions

1. Follow the pipeline order in the system prompt exactly.
2. Use {work_dir} as the working directory for all output files.
3. Update pipeline_state after every step.
4. Call parse_gmx_log after every mdrun step.
5. Apply recovery actions if parse_gmx_log returns RECOVERABLE.
6. Call final_answer() when the pipeline completes or fails.
""".strip()


def _build_resume_prompt(ui_state: UIState) -> str:
    """
    Build the resume task prompt.
    Reads current state and instructs the agent to continue
    from where it left off.
    """
    state_file = ui_state.state_file
    state_json = "{}"

    if state_file and Path(state_file).exists():
        try:
            state_json = Path(state_file).read_text()
        except Exception:
            pass

    return f"""
Resume an interrupted GROMACS MD simulation pipeline.

## Current State

{state_json}

## Instructions

1. Read the pipeline state above carefully.
2. Identify which steps are in completed_steps.
3. Skip all completed steps.
4. Resume from current_step.
5. Use the file paths in state["files"] — do not regenerate files
   that already exist.
6. Continue following the pipeline order from the system prompt.
7. Call final_answer() when the pipeline completes or fails.
""".strip()


def _find_mdp_dir() -> str:
    """Find the MDP templates directory relative to the project root."""
    candidates = [
        Path("benchmarks/mdp"),
        Path("mdp_templates"),
        Path("agent/mdp_templates"),
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    # Fallback — caller must ensure MDP files exist
    return str(Path("benchmarks/mdp").resolve())


# ---------------------------------------------------------------------------
# Tool call interceptor
# ---------------------------------------------------------------------------

class ToolCallInterceptor:
    """
    Wraps each tool's forward() method to intercept calls and
    update UIState step records automatically.

    This means the agent does not need to explicitly call
    ProgressTracker — the interceptor handles it transparently.
    """

    # Maps tool name → pipeline step name(s)
    _TOOL_TO_STEP: dict[str, str] = {
        "pdb2gmx":         "pdb2gmx",
        "editconf":        "editconf",
        "solvate":         "solvate",
        "genion":          "genion",
        "grompp":          None,       # resolved dynamically from context
        "mdrun":           None,       # resolved dynamically from run_label
        "parse_gmx_log":   None,       # resolved dynamically from log_file
        "energy_analysis": "energy_analysis",
        "rmsd_analysis":   "rmsd_analysis",
        "pipeline_state":  None,       # not tracked as a step
        "read_file":       None,
        "write_file":      None,
        "validate_structure": None,
    }

    def __init__(
        self,
        tools:    list,
        tracker:  ProgressTracker,
        writer:   ResultsWriter,
        ui_state: UIState,
    ):
        self.tracker  = tracker
        self.writer   = writer
        self.ui_state = ui_state
        self._install(tools)

    def _install(self, tools: list) -> None:
        """Wrap each tool's forward() with the interceptor."""
        for tool in tools:
            original_forward = tool.forward
            tool_name        = tool.name

            def _make_wrapper(orig, name):
                def _wrapper(*args, **kwargs):
                    return self._intercept(name, orig, args, kwargs)
                return _wrapper

            tool.forward = _make_wrapper(original_forward, tool_name)
            logger.debug("Interceptor installed on tool: %s", tool_name)

    def _intercept(
        self,
        tool_name: str,
        original:  Any,
        args:      tuple,
        kwargs:    dict,
    ) -> str:
        """
        Called instead of tool.forward().
        Determines step name, marks it running, calls the real
        forward(), then marks it complete or failed.
        """
        step_name = self._resolve_step_name(tool_name, kwargs)

        # Log the tool call
        self.ui_state.append_log(
            f"TOOL    | → {tool_name}"
            + (f" (step: {step_name})" if step_name else "")
        )

        # Mark step as running
        if step_name:
            self.tracker.start(step_name)

        # Call the real tool
        try:
            result = original(*args, **kwargs)
        except Exception as exc:
            if step_name:
                self.tracker.fail(step_name, str(exc))
            self.ui_state.append_log(
                f"TOOL    | ✗ {tool_name} raised: {exc}"
            )
            raise

        # Log result summary
        success = "SUCCESS: True" in result if isinstance(result, str) else True
        self.ui_state.append_log(
            f"TOOL    | {'✓' if success else '✗'} {tool_name} "
            f"→ {'OK' if success else 'FAILED'}"
        )

        # Post-process result
        if isinstance(result, str):
            self._post_process(tool_name, step_name, result, kwargs, success)

        return result

    def _resolve_step_name(self, tool_name: str, kwargs: dict) -> str | None:
        """Resolve the pipeline step name from tool name + kwargs."""
        static = self._TOOL_TO_STEP.get(tool_name)
        if static is not None:
            return static or None

        if tool_name == "mdrun":
            label = kwargs.get("run_label", "md")
            return f"mdrun_{label}"

        if tool_name == "grompp":
            mdp = kwargs.get("mdp_file", "")
            for key in ("em", "nvt", "npt", "md", "ions"):
                if key in str(mdp).lower():
                    return f"grompp_{key}"
            return "grompp_em"

        if tool_name == "parse_gmx_log":
            log = kwargs.get("log_file", "")
            for key in ("em", "nvt", "npt", "md"):
                if key in str(log).lower():
                    return f"parse_{key}"
            return "parse_em"

        return None

    def _post_process(
        self,
        tool_name: str,
        step_name: str | None,
        result:    str,
        kwargs:    dict,
        success:   bool,
    ) -> None:
        """
        Parse tool output and update UIState with results.
        Called after every successful tool invocation.
        """
        if step_name:
            if success:
                # Extract a short message for the step card
                msg = self._extract_step_message(tool_name, result)
                self.tracker.complete(step_name, msg)
            else:
                err = self._extract_error_message(result)
                self.tracker.fail(step_name, err)

        # ── parse_gmx_log → extract metrics + recovery events ─────────────
        if tool_name == "parse_gmx_log":
            self._process_diagnosis(result, step_name or "", kwargs)

        # ── energy_analysis → extract energy data ─────────────────────────
        elif tool_name == "energy_analysis":
            self._process_energy(result, kwargs)

        # ── rmsd_analysis → extract RMSD data ─────────────────────────────
        elif tool_name == "rmsd_analysis":
            self._process_rmsd(result, kwargs)

    # ── Result parsers ────────────────────────────────────────────────────

    def _process_diagnosis(
        self,
        result:    str,
        step_name: str,
        kwargs:    dict,
    ) -> None:
        """Parse parse_gmx_log output and update UIState metrics."""

        def _field(name: str) -> str:
            m = re.search(rf"^{name}:\s*(.+)$", result, re.MULTILINE)
            return m.group(1).strip() if m else ""

        diagnosis = _field("DIAGNOSIS")
        severity  = _field("SEVERITY")
        action    = _field("PRIMARY_ACTION")

        # EM metrics
        if "em" in step_name.lower() or "em" in kwargs.get("log_file", ""):
            fmax_m = re.search(r"Fmax[=:\s]+([\d.eE+\-]+)", result)
            epot_m = re.search(r"Epot[=:\s]+([-\d.eE+]+)", result)
            fmax   = float(fmax_m.group(1)) if fmax_m else None
            epot   = float(epot_m.group(1)) if epot_m else None
            conv   = "SUCCESS_CONVERGED" in diagnosis

            if fmax is not None or epot is not None:
                self.writer.record_em_results(
                    fmax      = fmax,
                    epot      = epot,
                    converged = conv,
                )

        # NVT metrics
        if "nvt" in step_name.lower() or "nvt" in kwargs.get("log_file", ""):
            temp_m = re.search(
                r"temperature_mean[=:\s]+([\d.]+)", result, re.IGNORECASE
            )
            if temp_m:
                self.writer.record_nvt_results(
                    temp_mean = float(temp_m.group(1))
                )

        # NPT metrics
        if "npt" in step_name.lower() or "npt" in kwargs.get("log_file", ""):
            pres_m = re.search(
                r"pressure_mean[=:\s]+([-\d.]+)", result, re.IGNORECASE
            )
            if pres_m:
                self.writer.record_npt_results(
                    pres_mean = float(pres_m.group(1))
                )

        # Recovery event
        if severity == "RECOVERABLE":
            patches = self._extract_patches(result)
            self.tracker.record_recovery(
                step_name = step_name,
                diagnosis = result,
                action    = action,
                patches   = patches,
                success   = True,   # updated later if retry fails
            )

    def _process_energy(self, result: str, kwargs: dict) -> None:
        """Parse energy_analysis output and push to UIState."""
        xvg_file = kwargs.get("output_xvg", "")
        if xvg_file and Path(xvg_file).exists():
            values = self.writer.parse_energy_xvg(xvg_file)
            if values:
                term = kwargs.get("energy_terms", ["Potential"])[0]
                self.writer.record_energy_data({term: values})

    def _process_rmsd(self, result: str, kwargs: dict) -> None:
        """Parse rmsd_analysis output and push to UIState."""
        xvg_file = kwargs.get("output_xvg", "")
        if xvg_file and Path(xvg_file).exists():
            rmsd_data = self.writer.parse_rmsd_xvg(xvg_file)
            if rmsd_data:
                self.writer.record_rmsd_data(rmsd_data)

    @staticmethod
    def _extract_step_message(tool_name: str, result: str) -> str:
        """Extract a short success message from tool output."""
        # Look for output file names
        files = re.findall(r"[\w/.\-]+\.(?:gro|top|tpr|edr|xtc|log|cpt)", result)
        if files:
            names = [Path(f).name for f in files[:3]]
            return ", ".join(names)
        # Look for key metrics
        fmax_m = re.search(r"Fmax[=:\s]+([\d.eE+\-]+)", result)
        if fmax_m:
            return f"Fmax = {float(fmax_m.group(1)):.1f}"
        return ""

    @staticmethod
    def _extract_error_message(result: str) -> str:
        """Extract a short error message from tool output."""
        for line in result.splitlines():
            if "ERROR" in line or "Fatal" in line or "error" in line.lower():
                return line.strip()[:100]
        return result[:100]

    @staticmethod
    def _extract_patches(diagnosis: str) -> list[str]:
        """
        Extract MDP patch descriptions from diagnosis output.
        Looks for lines in the MDP_PATCHES section.
        """
        patches = []
        in_section = False
        for line in diagnosis.splitlines():
            if "MDP_PATCHES" in line:
                in_section = True
                continue
            if in_section:
                stripped = line.strip()
                if not stripped or stripped.startswith(
                    ("RERUN", "AGENT", "FALLBACK", "PRIMARY", "DIAGNOSIS",
                     "SEVERITY", "PHASE", "EVIDENCE")
                ):
                    break
                if stripped.startswith("-"):
                    stripped = stripped[1:].strip()
                if stripped:
                    patches.append(stripped)
        return patches


# ---------------------------------------------------------------------------
# Agent thread
# ---------------------------------------------------------------------------

def _run_agent_thread(ui_state: UIState, resume: bool) -> None:
    """
    Main agent execution function — runs in a background thread.

    Steps:
        1. Set up logging bridge
        2. Build config and agent
        3. Install tool interceptor
        4. Run agent
        5. Update UIState on completion
    """
    try:
        ui_state.append_log("RUNNER  | Agent thread started.")
        ui_state.append_log(
            f"RUNNER  | Mode: {'resume' if resume else 'fresh run'}"
        )

        # ── 1. Logging bridge ─────────────────────────────────────────────
        install_ui_log_handler(ui_state, logger_name="gromacs_agent")
        install_ui_log_handler(ui_state, logger_name="gromacs_ui")
        install_ui_log_handler(ui_state, logger_name="smolagents")

        # ── 2. Build config ───────────────────────────────────────────────
        work_dir   = Path(ui_state.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        state_file = work_dir / "pipeline_state.json"

        with ui_state.lock:
            ui_state.state_file = str(state_file)

        config = _build_agent_config(ui_state, state_file)
        ui_state.append_log(
            f"RUNNER  | Work dir:  {work_dir}"
        )
        ui_state.append_log(
            f"RUNNER  | LLM:       {ui_state.llm_backend} / "
            f"{ui_state.llm_model}"
        )

        # ── 3. Build agent ────────────────────────────────────────────────
        from agent.orchestrator import build_agent
        from agent.tools        import get_all_tools

        tools = get_all_tools(work_dir=str(work_dir))
        ui_state.append_log(
            f"RUNNER  | Tools loaded: "
            f"{[t.name for t in tools]}"
        )

        # ── 4. Install interceptor ────────────────────────────────────────
        tracker     = ProgressTracker(ui_state)
        writer      = ResultsWriter(ui_state)
        interceptor = ToolCallInterceptor(tools, tracker, writer, ui_state)

        # ── 5. Build CodeAgent ────────────────────────────────────────────
        from agent.orchestrator import load_model, load_system_prompt
        from smolagents         import CodeAgent

        model  = load_model(config["llm"])
        prompt = load_system_prompt()

        agent = CodeAgent(
            tools        = tools,
            model        = model,
            instructions = prompt,
            max_steps    = ui_state.max_steps,
            additional_authorized_imports = [
                "json", "re", "math", "pathlib",
                "statistics", "collections",
                "datetime", "itertools",
                "agent.utils.mdp_utils",
                "agent.recovery",
            ],
        )

        ui_state.append_log("RUNNER  | CodeAgent built. Starting run...")

        # ── 6. Build task prompt ──────────────────────────────────────────
        if resume:
            task = _build_resume_prompt(ui_state)
        else:
            task = _build_task_prompt(ui_state)

        ui_state.append_log(
            f"RUNNER  | Task prompt ({len(task)} chars) sent to agent."
        )

        # ── 7. Run agent ──────────────────────────────────────────────────
        with ui_state.lock:
            ui_state.status = RunStatus.RUNNING

        result = agent.run(
            task,
            reset = not resume,
        )

        # ── 8. Handle completion ──────────────────────────────────────────
        if ui_state.stop_requested:
            _finish(ui_state, RunStatus.STOPPED, "Run stopped by user.")
        else:
            writer.record_final_answer(str(result) if result else "")
            _finish(ui_state, RunStatus.COMPLETED, "Pipeline completed.")
            ui_state.append_log(
                f"RUNNER  | ✅ Run complete. "
                f"Final answer: {str(result)[:200]}"
            )

    except Exception as exc:
        tb = traceback.format_exc()
        ui_state.append_log(f"RUNNER  | ❌ Agent thread error: {exc}")
        ui_state.append_log(f"RUNNER  | Traceback:\n{tb}")
        _finish(ui_state, RunStatus.FAILED, str(exc))


def _finish(
    ui_state: UIState,
    status:   RunStatus,
    message:  str,
) -> None:
    """Mark the run as finished and record end time."""
    with ui_state.lock:
        ui_state.status   = status
        ui_state.ended_at = datetime.now().isoformat(timespec="seconds")
    ui_state.append_log(
        f"RUNNER  | Status → {status.value}. {message}"
    )


def _build_agent_config(
    ui_state:   UIState,
    state_file: Path,
) -> dict:
    """Build the agent config dict from UIState."""
    return {
        "llm": {
            "backend":     ui_state.llm_backend,
            "model_id":    ui_state.llm_model,
            "temperature": ui_state.temperature,
            "max_tokens":  4096,
        },
        "pipeline": {
            "work_dir":              str(ui_state.work_dir),
            "templates_dir":         _find_mdp_dir(),
            "state_file":            str(state_file),
            "max_recovery_attempts": 3,
        },
        "simulation": {
            "force_field":        ui_state.force_field,
            "water_model":        ui_state.water_model,
            "box_type":           ui_state.box_type,
            "box_distance":       ui_state.box_distance,
            "temperature":        300,
            "n_threads":          ui_state.n_threads,
            "use_gpu":            ui_state.use_gpu,
        },
    }


# ---------------------------------------------------------------------------
# Public API — called by UI-2 run/resume buttons
# ---------------------------------------------------------------------------

def start_run(ui_state: UIState, resume: bool = False) -> None:
    """
    Launch the agent in a background thread.
    Called by the Run Configuration tab when the user clicks
    ▶ Start Run or ⏩ Resume.

    Guards against starting a second run while one is active.
    """
    with ui_state.lock:
        if ui_state.status == RunStatus.RUNNING:
            ui_state.append_log(
                "RUNNER  | ⚠️  Run already in progress — ignoring start request."
            )
            return

    # Reset state for a fresh run (not for resume)
    if not resume:
        ui_state.reset_for_new_run()
        ui_state.append_log("RUNNER  | State reset for fresh run.")
    else:
        with ui_state.lock:
            ui_state.status          = RunStatus.RUNNING
            ui_state.stop_requested  = False
            ui_state.started_at      = datetime.now().isoformat(
                timespec="seconds"
            )
        ui_state.append_log("RUNNER  | Resuming from existing state.")

    thread = threading.Thread(
        target = _run_agent_thread,
        args   = (ui_state, resume),
        daemon = True,       # thread dies if main process exits
        name   = "gromacs-agent-runner",
    )
    thread.start()
    ui_state.append_log(
        f"RUNNER  | Background thread started "
        f"(thread id: {thread.ident})."
    )