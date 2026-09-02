"""
Shared UI state — passed between tab builders so all tabs
can read and write the same run context.

Design:
    UIState is a plain dataclass holding Gradio-compatible
    mutable state. It is instantiated once in build_app()
    and passed to every tab builder.

    Agent runs execute in a background thread. The UI polls
    for updates via Gradio's every= timer mechanism.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class RunStatus(str, Enum):
    IDLE      = "idle"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"
    STOPPED   = "stopped"


@dataclass
class StepRecord:
    """Record of a single pipeline step execution."""
    name:       str
    status:     str          = "pending"   # pending|running|ok|failed|skipped
    started_at: str          = ""
    ended_at:   str          = ""
    duration_s: float        = 0.0
    message:    str          = ""


@dataclass
class RecoveryRecord:
    """Record of a recovery action triggered during the run."""
    step:        str
    diagnosis:   str
    action:      str
    patches:     list[str]   = field(default_factory=list)
    success:     bool        = False
    timestamp:   str         = ""


@dataclass
class UIState:
    """
    Central mutable state shared across all UI tabs.
    All writes from the agent thread must acquire `lock`.
    """

    # ── Run control ───────────────────────────────────────────────────────
    status:         RunStatus    = RunStatus.IDLE
    lock:           threading.Lock = field(default_factory=threading.Lock)
    stop_requested: bool         = False

    # ── Configuration (set by Run Config tab) ─────────────────────────────
    pdb_path:       str          = ""
    work_dir:       str          = ""
    force_field:    str          = "amber99sb-ildn"
    water_model:    str          = "tip3p"
    box_type:       str          = "dodecahedron"
    box_distance:   float        = 1.0
    n_threads:      int          = 4
    use_gpu:        bool         = False
    llm_backend:    str          = "ollama"
    llm_model:      str          = "qwen2.5:14b"
    temperature:    float        = 0.1
    max_steps:      int          = 60

    # ── Run metadata ──────────────────────────────────────────────────────
    run_id:         str          = ""
    started_at:     str          = ""
    ended_at:       str          = ""
    total_duration: float        = 0.0

    # ── Live log ──────────────────────────────────────────────────────────
    log_lines:      list[str]    = field(default_factory=list)
    log_max_lines:  int          = 2000

    # ── Pipeline steps ────────────────────────────────────────────────────
    steps:          list[StepRecord] = field(default_factory=list)
    current_step:   str          = ""

    # ── Recovery events ───────────────────────────────────────────────────
    recovery_events: list[RecoveryRecord] = field(default_factory=list)

    # ── Results ───────────────────────────────────────────────────────────
    em_fmax:        float | None = None
    em_epot:        float | None = None
    nvt_temp_mean:  float | None = None
    npt_pres_mean:  float | None = None
    rmsd_data:      list[tuple[float, float]] = field(default_factory=list)
    energy_data:    dict[str, list[float]]    = field(default_factory=dict)
    final_answer:   str          = ""

    # ── Pipeline state file ───────────────────────────────────────────────
    state_file:     str          = ""

    # ── Initialise step records ───────────────────────────────────────────
    def __post_init__(self) -> None:
        self._init_steps()

    def _init_steps(self) -> None:
        step_names = [
            "pdb2gmx", "editconf", "solvate",
            "grompp_ions", "genion",
            "grompp_em",   "mdrun_em",   "parse_em",
            "grompp_nvt",  "mdrun_nvt",  "parse_nvt",
            "grompp_npt",  "mdrun_npt",  "parse_npt",
            "grompp_md",   "mdrun_md",   "parse_md",
            "energy_analysis", "rmsd_analysis",
        ]
        self.steps = [StepRecord(name=n) for n in step_names]

    # ── Thread-safe helpers ───────────────────────────────────────────────

    def append_log(self, line: str) -> None:
        with self.lock:
            ts = datetime.now().strftime("%H:%M:%S")
            self.log_lines.append(f"[{ts}] {line}")
            if len(self.log_lines) > self.log_max_lines:
                self.log_lines = self.log_lines[-self.log_max_lines:]

    def set_step_status(
        self,
        step_name: str,
        status: str,
        message: str = "",
    ) -> None:
        with self.lock:
            for step in self.steps:
                if step.name == step_name:
                    step.status = status
                    step.message = message
                    now = datetime.now().isoformat(timespec="seconds")
                    if status == "running":
                        step.started_at = now
                    elif status in ("ok", "failed"):
                        step.ended_at = now
                    break

    def add_recovery_event(self, record: RecoveryRecord) -> None:
        with self.lock:
            self.recovery_events.append(record)

    def get_log_text(self) -> str:
        with self.lock:
            return "\n".join(self.log_lines)

    def reset_for_new_run(self) -> None:
        with self.lock:
            self.status          = RunStatus.RUNNING
            self.stop_requested  = False
            self.log_lines       = []
            self.recovery_events = []
            self.em_fmax         = None
            self.em_epot         = None
            self.nvt_temp_mean   = None
            self.npt_pres_mean   = None
            self.rmsd_data       = []
            self.energy_data     = {}
            self.final_answer    = ""
            self.started_at      = datetime.now().isoformat(timespec="seconds")
            self.ended_at        = ""
            self._init_steps()