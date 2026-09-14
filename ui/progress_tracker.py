"""
ui/progress_tracker.py

Provides ProgressTracker — a helper the agent runner uses to
update UIState step records as the pipeline executes.

Usage (in ui/runner.py):
    tracker = ProgressTracker(ui_state)
    tracker.start("pdb2gmx")
    # ... run the step ...
    tracker.complete("pdb2gmx", message="conf.gro, topol.top written")
    # or
    tracker.fail("pdb2gmx", message="Fatal error: unknown residue LIG")
"""

from __future__ import annotations

import re
from datetime import datetime

from ui.state import RecoveryRecord, UIState


class ProgressTracker:
    """
    Thread-safe helper for updating pipeline step statuses in UIState.
    """

    def __init__(self, ui_state: UIState):
        self.ui_state = ui_state

    def start(self, step_name: str) -> None:
        """Mark a step as running."""
        self.ui_state.set_step_status(step_name, "running")
        self.ui_state.append_log(f"STEP    | ▶ Starting: {step_name}")
        with self.ui_state.lock:
            self.ui_state.current_step = step_name

    def complete(self, step_name: str, message: str = "") -> None:
        """Mark a step as successfully completed."""
        self.ui_state.set_step_status(step_name, "ok", message)
        self.ui_state.append_log(
            f"STEP    | ✅ Completed: {step_name}"
            + (f" — {message[:80]}" if message else "")
        )

    def fail(self, step_name: str, message: str = "") -> None:
        """Mark a step as failed."""
        self.ui_state.set_step_status(step_name, "failed", message)
        self.ui_state.append_log(
            f"STEP    | ❌ Failed: {step_name}"
            + (f" — {message[:80]}" if message else "")
        )

    def skip(self, step_name: str, reason: str = "already completed") -> None:
        """Mark a step as skipped (resume mode)."""
        self.ui_state.set_step_status(step_name, "skipped", reason)
        self.ui_state.append_log(
            f"STEP    | ⏭️  Skipped: {step_name} ({reason})"
        )

    def record_recovery(
        self,
        step_name:  str,
        diagnosis:  str,
        action:     str,
        patches:    list[str],
        success:    bool,
    ) -> None:
        """Record a recovery event in UIState."""
        record = RecoveryRecord(
            step      = step_name,
            diagnosis = diagnosis,
            action    = action,
            patches   = patches,
            success   = success,
            timestamp = datetime.now().isoformat(timespec="seconds"),
        )
        self.ui_state.add_recovery_event(record)
        icon = "✅" if success else "❌"
        self.ui_state.append_log(
            f"RECOVER | {icon} {step_name}: {action} "
            f"({'succeeded' if success else 'failed'})"
        )

    def parse_diagnosis_message(self, diagnosis_str: str) -> str:
        """
        Extract a short display message from a parse_gmx_log output string.
        Returns e.g. "SUCCESS_CONVERGED | Fmax=987.6 kJ/mol/nm"
        """
        code    = self._extract_field(diagnosis_str, "DIAGNOSIS")
        severity = self._extract_field(diagnosis_str, "SEVERITY")
        action  = self._extract_field(diagnosis_str, "PRIMARY_ACTION")

        parts = [p for p in [code, severity, action] if p]
        return " | ".join(parts) if parts else diagnosis_str[:80]

    @staticmethod
    def _extract_field(text: str, field: str) -> str:
        """Extract a single-line field value from diagnosis output."""
        pattern = rf"^{field}:\s*(.+)$"
        match   = re.search(pattern, text, re.MULTILINE)
        return match.group(1).strip() if match else ""