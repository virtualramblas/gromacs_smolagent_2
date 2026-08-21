"""
Pipeline state persistence — lets the agent track progress
and resume interrupted runs.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from smolagents import Tool

logger_name = "gromacs_agent.state"


class PipelineStateTool(Tool):
    """
    Read and write the pipeline state JSON file.
    The agent uses this to track completed steps, current files,
    and any flags (e.g. em_converged) across the full workflow.
    """

    name = "pipeline_state"
    description = (
        "Read or update the pipeline state JSON file. "
        "Use action='read' to get current state. "
        "Use action='update' with updates dict to modify specific fields. "
        "Use action='reset' to start fresh. "
        "Returns the current state as a formatted string."
    )
    inputs = {
        "action": {
            "type": "string",
            "description": "'read', 'update', or 'reset'.",
        },
        "updates": {
            "type": "object",
            "description": (
                "Dict of key-value pairs to update in the state. "
                "Only used when action='update'."
            ),
            "nullable": True,
        },
    }
    output_type = "string"

    # Default state schema
    _DEFAULT_STATE = {
        "input_pdb": None,
        "work_dir": None,
        "completed_steps": [],
        "current_step": None,
        "files": {
            "pdb":      None,
            "gro":      None,
            "gro_box":  None,
            "gro_solv": None,
            "gro_ions": None,
            "top":      None,
            "tpr_em":   None,
            "tpr_nvt":  None,
            "tpr_npt":  None,
            "tpr_md":   None,
            "cpt":      None,
            "edr_em":   None,
            "edr_md":   None,
            "xtc":      None,
        },
        "em_converged": None,
        "nvt_complete": None,
        "npt_complete": None,
        "md_complete":  None,
        "warnings": [],
        "errors": [],
        "last_updated": None,
    }

    def __init__(self, state_file: str | Path = "pipeline_state.json"):
        super().__init__()
        self.state_file = Path(state_file)

    def _load(self) -> dict:
        if self.state_file.exists():
            return json.loads(self.state_file.read_text())
        return dict(self._DEFAULT_STATE)  # fresh copy

    def _save(self, state: dict) -> None:
        state["last_updated"] = datetime.now(timezone.utc).isoformat()
        self.state_file.write_text(json.dumps(state, indent=2, default=str))

    def forward(self, action: str, updates: dict | None = None) -> str:
        action = action.strip().lower()

        if action == "reset":
            state = dict(self._DEFAULT_STATE)
            self._save(state)
            return "State reset to defaults.\n" + json.dumps(state, indent=2)

        if action == "read":
            state = self._load()
            return json.dumps(state, indent=2, default=str)

        if action == "update":
            if not updates:
                return "ERROR: 'update' action requires a non-empty 'updates' dict."
            state = self._load()
            # Deep merge for nested 'files' dict
            for key, value in updates.items():
                if key == "files" and isinstance(value, dict):
                    state.setdefault("files", {}).update(value)
                elif key in ("warnings", "errors") and isinstance(value, list):
                    state.setdefault(key, []).extend(value)
                else:
                    state[key] = value
            self._save(state)
            return "State updated.\n" + json.dumps(state, indent=2, default=str)

        return f"ERROR: Unknown action '{action}'. Use 'read', 'update', or 'reset'."