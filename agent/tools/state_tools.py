"""
Pipeline state persistence — lets the agent track progress
and resume interrupted runs.
"""

from __future__ import annotations

import copy
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from smolagents import Tool

logger = logging.getLogger("gromacs_agent.state")


class PipelineStateTool(Tool):
    """
    Read and write the pipeline state JSON file.
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

    # ------------------------------------------------------------------
    # Default state schema — NOTE: use _fresh_default() everywhere,
    # never mutate this directly (deep copy required for nested dicts)
    # ------------------------------------------------------------------
    _DEFAULT_STATE = {
        "input_pdb":       None,
        "work_dir":        None,
        "completed_steps": [],
        "current_step":    None,
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
        "em_converged":  None,
        "nvt_complete":  None,
        "npt_complete":  None,
        "md_complete":   None,
        "warnings":      [],
        "errors":        [],
        "last_updated":  None,
    }

    def __init__(self, state_file: str | Path = "pipeline_state.json"):
        super().__init__()
        self.state_file = Path(state_file)

    def _fresh_default(self) -> dict:
        """
        Return a deep copy of _DEFAULT_STATE.
        Deep copy is essential — the nested 'files' dict must be
        independent between calls, otherwise mutations in one test
        or pipeline run bleed into subsequent ones.
        """
        return copy.deepcopy(self._DEFAULT_STATE)

    def _load(self) -> dict:
        """
        Load state from disk.
        Returns a fresh default if the file does not exist or
        contains invalid JSON (graceful recovery).
        Does NOT write to disk — read is non-destructive.
        """
        if not self.state_file.exists():
            return self._fresh_default()
        try:
            return json.loads(self.state_file.read_text())
        except json.JSONDecodeError:
            logger.warning(
                "State file %s contains invalid JSON. "
                "Returning default state.",
                self.state_file,
            )
            return self._fresh_default()

    def _save(self, state: dict) -> None:
        """
        Persist state to disk as indented JSON.
        Creates parent directories if needed.
        """
        state["last_updated"] = datetime.now(timezone.utc).isoformat()
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(
            json.dumps(state, indent=2, default=str)
        )

    # ------------------------------------------------------------------
    # Public SmolAgent entry point
    # ------------------------------------------------------------------

    def forward(self, action: str, updates: dict | None = None) -> str:
        action = action.strip().lower()

        # ---- RESET -------------------------------------------------------
        if action == "reset":
            state = self._fresh_default()
            self._save(state)
            # Return pure JSON so callers can json.loads() the result
            return json.dumps(state, indent=2, default=str)

        # ---- READ --------------------------------------------------------
        if action == "read":
            state = self._load()
            # Persist on first read so the file always exists after read
            self._save(state)
            return json.dumps(state, indent=2, default=str)

        # ---- UPDATE ------------------------------------------------------
        if action == "update":
            if not updates:
                return "ERROR: 'update' action requires a non-empty 'updates' dict."

            state = self._load()

            for key, value in updates.items():
                if key == "files" and isinstance(value, dict):
                    # Deep merge: update individual file keys, not replace dict
                    state.setdefault("files", {}).update(value)

                elif key in ("warnings", "errors") and isinstance(value, list):
                    # Append semantics: extend existing list
                    state.setdefault(key, []).extend(value)

                else:
                    # Scalar / list replacement
                    state[key] = value

            self._save(state)
            return json.dumps(state, indent=2, default=str)

        # ---- UNKNOWN -----------------------------------------------------
        return (
            f"ERROR: Unknown action '{action}'. "
            "Use 'read', 'update', or 'reset'."
        )