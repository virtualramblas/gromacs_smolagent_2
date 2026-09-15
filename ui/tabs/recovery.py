"""
UI-6: Recovery Events Viewer

Displays a live timeline of all recovery actions triggered
during the run. Each event card shows:
    - Timestamp and step name
    - Diagnosis code and severity
    - Primary action taken
    - MDP patches applied (parameter → old value → new value)
    - Whether the recovery succeeded
    - Fallback action (if primary failed)

A statistics panel at the top summarises:
    - Total recovery attempts
    - Success rate
    - Most common diagnosis
    - Most common action
    - Steps that required recovery
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime

import gradio as gr

from ui.state import RecoveryRecord, RunStatus, UIState


# ---------------------------------------------------------------------------
# Severity styling
# ---------------------------------------------------------------------------

_SEVERITY_COLOUR: dict[str, str] = {
    "OK":                 "#059669",
    "RECOVERABLE":        "#d97706",
    "ASSISTED":           "#7c3aed",
    "FATAL":              "#dc2626",
    "NEEDS_HUMAN_REVIEW": "#dc2626",
}

_SEVERITY_BG: dict[str, str] = {
    "OK":                 "#f0fdf4",
    "RECOVERABLE":        "#fffbeb",
    "ASSISTED":           "#f5f3ff",
    "FATAL":              "#fef2f2",
    "NEEDS_HUMAN_REVIEW": "#fef2f2",
}

_SEVERITY_BORDER: dict[str, str] = {
    "OK":                 "#059669",
    "RECOVERABLE":        "#d97706",
    "ASSISTED":           "#7c3aed",
    "FATAL":              "#dc2626",
    "NEEDS_HUMAN_REVIEW": "#dc2626",
}

_SEVERITY_ICON: dict[str, str] = {
    "OK":                 "✅",
    "RECOVERABLE":        "🔧",
    "ASSISTED":           "🤝",
    "FATAL":              "💀",
    "NEEDS_HUMAN_REVIEW": "👤",
}

# Human-readable action labels
_ACTION_LABELS: dict[str, str] = {
    "NONE":                              "No action needed",
    "REDUCE_EMSTEP":                     "Reduce emstep",
    "INCREASE_NSTEPS":                   "Increase nsteps",
    "REDUCE_EMSTEP_AND_INCREASE_NSTEPS": "Reduce emstep + increase nsteps",
    "SWITCH_INTEGRATOR_SD":              "Switch integrator to SD",
    "REDUCE_DT":                         "Reduce timestep (dt)",
    "INCREASE_LINCS_ORDER":              "Increase LINCS order",
    "DISABLE_LINCS":                     "Disable LINCS",
    "INCREASE_NEIGHBOUR_FREQ":           "Increase neighbour list frequency",
    "REBUILD_TOPOLOGY":                  "Rebuild topology",
    "RERUN_GENION":                      "Re-run genion",
    "SWITCH_TO_CPU":                     "Switch to CPU execution",
    "RESUME_FROM_CHECKPOINT":            "Resume from checkpoint",
    "ESCALATE_TO_USER":                  "Escalate to user",
}

# Diagnosis code descriptions
_DIAGNOSIS_DESC: dict[str, str] = {
    "SUCCESS_CONVERGED":       "EM converged successfully",
    "SUCCESS_COMPLETED":       "Simulation completed successfully",
    "EM_NOT_CONVERGED":        "EM did not converge",
    "EM_STEP_LIMIT_HIT":       "EM step limit reached",
    "EM_LINCS_ERROR":          "LINCS constraint error during EM",
    "EM_EXPLODED":             "System exploded (Epot >> 0)",
    "EQUIL_TEMP_UNSTABLE":     "Temperature unstable during equilibration",
    "EQUIL_PRESSURE_UNSTABLE": "Pressure unstable during equilibration",
    "EQUIL_DRIFT_TOO_HIGH":    "Energy drift too high",
    "LINCS_WARNING":           "LINCS bond angle warning",
    "SETTLE_ERROR":            "SETTLE constraint error",
    "PARTICLE_ESCAPED_BOX":    "Particle escaped simulation box",
    "NAN_DETECTED":            "NaN detected in forces/energies",
    "NEIGHBOUR_LIST_ERROR":    "Neighbour list error",
    "MISSING_PARAMETERS":      "Missing force field parameters",
    "CHARGE_IMBALANCE":        "Net charge in system",
    "TOPOLOGY_MISMATCH":       "Topology inconsistency",
    "GPU_ERROR":               "GPU/CUDA error",
    "MPI_ERROR":               "MPI communication error",
    "DISK_FULL":               "Disk full",
    "TIMEOUT":                 "Simulation timed out",
    "UNKNOWN_ERROR":           "Unrecognised error",
    "NEEDS_HUMAN_REVIEW":      "Requires manual review",
}


# ---------------------------------------------------------------------------
# HTML builders
# ---------------------------------------------------------------------------

def _patch_row_html(patch: str) -> str:
    """
    Render a single MDP patch as a table row.
    Expected format: "parameter: old_value → new_value (reason)"
    or just a plain string.
    """
    import re
    # Try to parse structured patch string
    m = re.match(
        r"^([^:]+):\s*(.+?)\s*→\s*(.+?)(?:\s*\((.+)\))?$",
        patch.strip(),
    )
    if m:
        param     = m.group(1).strip()
        old_val   = m.group(2).strip()
        new_val   = m.group(3).strip()
        reason    = m.group(4).strip() if m.group(4) else ""
        reason_td = (
            f'<td style="padding:4px 8px; color:#9ca3af; '
            f'font-size:0.72rem; font-style:italic;">{reason}</td>'
            if reason else
            '<td></td>'
        )
        return f"""
        <tr>
            <td style="padding:4px 8px; font-family:monospace;
                       font-size:0.78rem; color:#60a5fa;
                       font-weight:600;">{param}</td>
            <td style="padding:4px 8px; font-family:monospace;
                       font-size:0.78rem; color:#ef4444;
                       text-decoration:line-through;">{old_val}</td>
            <td style="padding:4px 8px; color:#9ca3af;
                       font-size:0.78rem;">→</td>
            <td style="padding:4px 8px; font-family:monospace;
                       font-size:0.78rem; color:#34d399;
                       font-weight:600;">{new_val}</td>
            {reason_td}
        </tr>
        """
    else:
        # Plain string — render as single cell
        return f"""
        <tr>
            <td colspan="5" style="padding:4px 8px;
                font-family:monospace; font-size:0.78rem;
                color:#cbd5e1;">{patch}</td>
        </tr>
        """


def _patches_table_html(patches: list[str]) -> str:
    """Render the MDP patches as an HTML table."""
    if not patches:
        return ""

    rows = "".join(_patch_row_html(p) for p in patches)

    return f"""
    <div style="margin-top: 10px;">
        <div style="font-size:0.75rem; color:#94a3b8;
                    font-weight:600; margin-bottom:4px;
                    text-transform:uppercase; letter-spacing:0.05em;">
            MDP Patches Applied
        </div>
        <table style="
            width: 100%;
            border-collapse: collapse;
            background: #0f172a;
            border-radius: 6px;
            overflow: hidden;
        ">
            <thead>
                <tr style="background:#1e293b;">
                    <th style="padding:5px 8px; text-align:left;
                               color:#64748b; font-size:0.72rem;">
                        Parameter</th>
                    <th style="padding:5px 8px; text-align:left;
                               color:#64748b; font-size:0.72rem;">
                        Old value</th>
                    <th style="padding:5px 8px;"></th>
                    <th style="padding:5px 8px; text-align:left;
                               color:#64748b; font-size:0.72rem;">
                        New value</th>
                    <th style="padding:5px 8px; text-align:left;
                               color:#64748b; font-size:0.72rem;">
                        Reason</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
    </div>
    """


def _recovery_card_html(
    record: RecoveryRecord,
    index:  int,
) -> str:
    """Render a single recovery event as an HTML card."""
    # Extract severity from diagnosis string if available
    import re
    severity_match = re.search(
        r"SEVERITY:\s*(\w+)", record.diagnosis
    )
    severity = severity_match.group(1) if severity_match else "RECOVERABLE"

    # Extract diagnosis code
    code_match = re.search(
        r"DIAGNOSIS:\s*(\w+)", record.diagnosis
    )
    diag_code = code_match.group(1) if code_match else record.diagnosis

    colour  = _SEVERITY_COLOUR.get(severity, "#d97706")
    bg      = _SEVERITY_BG.get(severity,     "#fffbeb")
    border  = _SEVERITY_BORDER.get(severity, "#d97706")
    icon    = _SEVERITY_ICON.get(severity,   "🔧")

    desc    = _DIAGNOSIS_DESC.get(diag_code, diag_code)
    action  = _ACTION_LABELS.get(record.action, record.action)

    success_badge = (
        '<span style="color:#059669; font-weight:600; '
        'font-size:0.78rem;">✅ Recovered</span>'
        if record.success else
        '<span style="color:#dc2626; font-weight:600; '
        'font-size:0.78rem;">❌ Recovery failed</span>'
    )

    patches_html = _patches_table_html(record.patches)

    # Format timestamp
    ts = record.timestamp
    try:
        dt = datetime.fromisoformat(ts)
        ts = dt.strftime("%H:%M:%S")
    except Exception:
        pass

    return f"""
    <div style="
        border: 1px solid {border};
        border-left: 5px solid {border};
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
        background: {bg};
    ">
        <!-- Header row -->
        <div style="
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 10px;
        ">
            <div style="display:flex; align-items:center; gap:10px;">
                <span style="font-size:1.2rem;">{icon}</span>
                <div>
                    <span style="
                        font-weight: 700;
                        font-size: 0.9rem;
                        color: {colour};
                        font-family: monospace;
                    ">#{index} — {record.step}</span>
                    <span style="
                        margin-left: 10px;
                        font-size: 0.78rem;
                        color: #6b7280;
                    ">{ts}</span>
                </div>
            </div>
            {success_badge}
        </div>

        <!-- Diagnosis row -->
        <div style="
            display: flex;
            gap: 16px;
            flex-wrap: wrap;
            margin-bottom: 8px;
        ">
            <div>
                <span style="font-size:0.72rem; color:#9ca3af;
                             text-transform:uppercase;
                             letter-spacing:0.05em;">Diagnosis</span>
                <div style="font-family:monospace; font-size:0.82rem;
                            color:{colour}; font-weight:600;
                            margin-top:2px;">{diag_code}</div>
                <div style="font-size:0.75rem; color:#6b7280;
                            margin-top:1px;">{desc}</div>
            </div>
            <div>
                <span style="font-size:0.72rem; color:#9ca3af;
                             text-transform:uppercase;
                             letter-spacing:0.05em;">Severity</span>
                <div style="font-family:monospace; font-size:0.82rem;
                            color:{colour}; font-weight:600;
                            margin-top:2px;">{severity}</div>
            </div>
            <div>
                <span style="font-size:0.72rem; color:#9ca3af;
                             text-transform:uppercase;
                             letter-spacing:0.05em;">Action taken</span>
                <div style="font-family:monospace; font-size:0.82rem;
                            color:#60a5fa; font-weight:600;
                            margin-top:2px;">{record.action}</div>
                <div style="font-size:0.75rem; color:#6b7280;
                            margin-top:1px;">{action}</div>
            </div>
        </div>

        <!-- Patches table -->
        {patches_html}
    </div>
    """


def _statistics_html(events: list[RecoveryRecord]) -> str:
    """Render the recovery statistics summary panel."""
    if not events:
        return ""

    n_total   = len(events)
    n_success = sum(1 for e in events if e.success)
    n_failed  = n_total - n_success
    rate      = int((n_success / n_total) * 100) if n_total else 0

    # Most common diagnosis
    import re
    diag_codes = []
    for e in events:
        m = re.search(r"DIAGNOSIS:\s*(\w+)", e.diagnosis)
        if m:
            diag_codes.append(m.group(1))
        else:
            diag_codes.append(e.diagnosis[:30])

    most_common_diag = (
        Counter(diag_codes).most_common(1)[0][0]
        if diag_codes else "—"
    )

    # Most common action
    actions = [e.action for e in events if e.action]
    most_common_action = (
        Counter(actions).most_common(1)[0][0]
        if actions else "—"
    )

    # Steps that needed recovery
    affected_steps = sorted(set(e.step for e in events))

    # Rate bar colour
    bar_colour = (
        "#059669" if rate >= 80 else
        "#d97706" if rate >= 50 else
        "#dc2626"
    )

    def _stat(value: str, label: str, colour: str = "#0d9488") -> str:
        return f"""
        <div style="
            text-align: center;
            padding: 10px 14px;
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 8px;
            min-width: 90px;
            flex: 1;
        ">
            <div style="font-size:1.2rem; font-weight:700;
                        color:{colour}; font-family:monospace;">
                {value}
            </div>
            <div style="font-size:0.72rem; color:#64748b;
                        margin-top:3px;">{label}</div>
        </div>
        """

    stats_row = f"""
    <div style="display:flex; gap:10px; flex-wrap:wrap; margin-bottom:14px;">
        {_stat(str(n_total),   "Total events",   "#94a3b8")}
        {_stat(str(n_success), "Succeeded",      "#059669")}
        {_stat(str(n_failed),  "Failed",
               "#dc2626" if n_failed else "#64748b")}
        {_stat(f"{rate}%",     "Success rate",   bar_colour)}
    </div>
    """

    rate_bar = f"""
    <div style="margin-bottom:14px;">
        <div style="display:flex; justify-content:space-between;
                    font-size:0.75rem; color:#64748b; margin-bottom:4px;">
            <span>Recovery success rate</span>
            <span>{n_success} / {n_total}</span>
        </div>
        <div style="background:#334155; border-radius:999px;
                    height:8px; overflow:hidden;">
            <div style="
                background:{bar_colour};
                width:{rate}%;
                height:100%;
                border-radius:999px;
                transition:width 0.5s ease;
            "></div>
        </div>
    </div>
    """

    details = f"""
    <div style="
        display: flex;
        gap: 20px;
        flex-wrap: wrap;
        font-size: 0.78rem;
        color: #94a3b8;
        margin-bottom: 4px;
    ">
        <div>
            <span style="color:#64748b;">Most common diagnosis: </span>
            <span style="font-family:monospace; color:#f59e0b;
                         font-weight:600;">{most_common_diag}</span>
        </div>
        <div>
            <span style="color:#64748b;">Most common action: </span>
            <span style="font-family:monospace; color:#60a5fa;
                         font-weight:600;">{most_common_action}</span>
        </div>
        <div>
            <span style="color:#64748b;">Affected steps: </span>
            <span style="font-family:monospace; color:#a78bfa;">
                {", ".join(affected_steps) if affected_steps else "—"}
            </span>
        </div>
    </div>
    """

    return f"""
    <div style="
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px 20px;
        background: #1e293b;
        margin-bottom: 20px;
    ">
        <div style="font-size:0.85rem; font-weight:700; color:#f1f5f9;
                    margin-bottom:12px;">
            📊 Recovery Statistics
        </div>
        {stats_row}
        {rate_bar}
        {details}
    </div>
    """


def _empty_state_html() -> str:
    """Render the empty state when no recovery events exist."""
    return """
    <div style="
        text-align: center;
        padding: 60px 20px;
        color: #64748b;
    ">
        <div style="font-size: 2rem; margin-bottom: 12px;">🔧</div>
        <div style="font-size: 1rem; font-weight: 600;
                    color: #94a3b8;">
            No recovery events yet
        </div>
        <div style="font-size: 0.85rem; margin-top: 8px;
                    color: #64748b; max-width: 400px;
                    margin-left: auto; margin-right: auto;">
            Recovery events appear here when the agent detects
            a simulation failure and applies automatic fixes.
            A clean run with no failures will leave this panel empty.
        </div>
    </div>
    """


def build_recovery_html(ui_state: UIState) -> str:
    """
    Build the full recovery panel HTML from current UIState.
    Called on every timer tick.
    """
    with ui_state.lock:
        events = list(ui_state.recovery_events)
        status = ui_state.status

    if not events:
        if status == RunStatus.COMPLETED:
            return """
            <div style="
                text-align: center;
                padding: 40px 20px;
                color: #059669;
            ">
                <div style="font-size: 2rem; margin-bottom: 10px;">🎉</div>
                <div style="font-size: 1rem; font-weight: 700;">
                    Clean run — no recovery needed
                </div>
                <div style="font-size: 0.85rem; margin-top: 6px;
                            color: #64748b;">
                    The pipeline completed without triggering
                    any recovery actions.
                </div>
            </div>
            """
        return _empty_state_html()

    # Statistics panel
    html = _statistics_html(events)

    # Timeline header
    html += f"""
    <div style="
        font-size: 0.85rem;
        font-weight: 700;
        color: #94a3b8;
        margin-bottom: 12px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    ">
        Recovery Timeline — {len(events)} event{"s" if len(events) != 1 else ""}
    </div>
    """

    # Event cards (most recent first)
    for i, record in enumerate(reversed(events), start=1):
        html += _recovery_card_html(record, len(events) - i + 1)

    return html


# ---------------------------------------------------------------------------
# Tab builder
# ---------------------------------------------------------------------------

def build_recovery_tab(ui_state: UIState) -> dict:
    """Build the Recovery Events tab. Returns component references."""

    gr.Markdown("## 🔧 Recovery Events")

    # ── Description ───────────────────────────────────────────────────────
    gr.Markdown(
        "_Automatic recovery actions triggered when the agent detects "
        "simulation failures. Updates every 4 seconds during a run._"
    )

    # ── Controls row ──────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=3):
            filter_dd = gr.Dropdown(
                label   = "Filter by outcome",
                choices = ["All", "Succeeded", "Failed"],
                value   = "All",
            )
        with gr.Column(scale=1):
            refresh_btn = gr.Button(
                "🔄  Refresh",
                variant = "secondary",
                size    = "sm",
            )
        with gr.Column(scale=1):
            export_btn = gr.Button(
                "💾  Export CSV",
                variant = "secondary",
                size    = "sm",
            )

    # ── Main recovery display ─────────────────────────────────────────────
    recovery_html = gr.HTML(
        value   = build_recovery_html(ui_state),
        elem_id = "recovery-panel",
    )

    # ── Export file ───────────────────────────────────────────────────────
    export_file = gr.File(
        label   = "Recovery log CSV",
        visible = False,
    )

    # ── Diagnosis reference accordion ─────────────────────────────────────
    with gr.Accordion("📖 Diagnosis Code Reference", open=False):
        ref_rows = "".join(
            f"<tr>"
            f"<td style='padding:5px 10px; font-family:monospace; "
            f"font-size:0.78rem; color:#60a5fa; font-weight:600;'>{code}</td>"
            f"<td style='padding:5px 10px; font-size:0.78rem; "
            f"color:#94a3b8;'>{desc}</td>"
            f"</tr>"
            for code, desc in _DIAGNOSIS_DESC.items()
        )
        gr.HTML(f"""
        <table style="
            width:100%; border-collapse:collapse;
            background:#1e293b; border-radius:8px; overflow:hidden;
        ">
            <thead>
                <tr style="background:#0f172a;">
                    <th style="padding:7px 10px; text-align:left;
                               color:#64748b; font-size:0.75rem;
                               font-weight:600;">Code</th>
                    <th style="padding:7px 10px; text-align:left;
                               color:#64748b; font-size:0.75rem;
                               font-weight:600;">Description</th>
                </tr>
            </thead>
            <tbody>{ref_rows}</tbody>
        </table>
        """)

    # ── Action reference accordion ─────────────────────────────────────────
    with gr.Accordion("⚡ Recovery Action Reference", open=False):
        action_rows = "".join(
            f"<tr>"
            f"<td style='padding:5px 10px; font-family:monospace; "
            f"font-size:0.78rem; color:#f59e0b; font-weight:600;'>{action}</td>"
            f"<td style='padding:5px 10px; font-size:0.78rem; "
            f"color:#94a3b8;'>{label}</td>"
            f"</tr>"
            for action, label in _ACTION_LABELS.items()
        )
        gr.HTML(f"""
        <table style="
            width:100%; border-collapse:collapse;
            background:#1e293b; border-radius:8px; overflow:hidden;
        ">
            <thead>
                <tr style="background:#0f172a;">
                    <th style="padding:7px 10px; text-align:left;
                               color:#64748b; font-size:0.75rem;
                               font-weight:600;">Action</th>
                    <th style="padding:7px 10px; text-align:left;
                               color:#64748b; font-size:0.75rem;
                               font-weight:600;">Description</th>
                </tr>
            </thead>
            <tbody>{action_rows}</tbody>
        </table>
        """)

    # ── Timer — polls every 4 seconds ─────────────────────────────────────
    timer = gr.Timer(value=4)

    # ── Poll and event handlers ───────────────────────────────────────────

    def poll_recovery(filter_choice: str) -> str:
        with ui_state.lock:
            events = list(ui_state.recovery_events)
            status = ui_state.status

        if filter_choice == "Succeeded":
            events = [e for e in events if e.success]
        elif filter_choice == "Failed":
            events = [e for e in events if not e.success]

        # Rebuild with filtered events — create a temporary state view
        class _FilteredState:
            def __init__(self):
                self.recovery_events = events
                self.status          = status
                self.lock            = ui_state.lock

        return build_recovery_html(_FilteredState())

    def on_refresh(filter_choice: str) -> str:
        return poll_recovery(filter_choice)

    def on_export() -> tuple[gr.File, str]:
        """Export recovery events to CSV."""
        import csv
        import tempfile
        from pathlib import Path

        with ui_state.lock:
            events = list(ui_state.recovery_events)

        if not events:
            return gr.File(visible=False), ""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"recovery_events_{timestamp}.csv"
        tmp_path  = Path(tempfile.mkdtemp()) / filename

        with tmp_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "step", "diagnosis",
                "action", "patches", "success",
            ])
            for e in events:
                import re
                code_m = re.search(r"DIAGNOSIS:\s*(\w+)", e.diagnosis)
                code   = code_m.group(1) if code_m else e.diagnosis[:40]
                writer.writerow([
                    e.timestamp,
                    e.step,
                    code,
                    e.action,
                    " | ".join(e.patches),
                    e.success,
                ])

        return gr.File(value=str(tmp_path), visible=True), ""

    # ── Wire timer ────────────────────────────────────────────────────────

    timer.tick(
        fn      = poll_recovery,
        inputs  = [filter_dd],
        outputs = [recovery_html],
    )

    refresh_btn.click(
        fn      = on_refresh,
        inputs  = [filter_dd],
        outputs = [recovery_html],
    )

    filter_dd.change(
        fn      = poll_recovery,
        inputs  = [filter_dd],
        outputs = [recovery_html],
    )

    export_btn.click(
        fn      = on_export,
        inputs  = [],
        outputs = [export_file, recovery_html],
    )

    return {
        "recovery_html": recovery_html,
        "filter_dd":     filter_dd,
        "refresh_btn":   refresh_btn,
        "export_btn":    export_btn,
        "export_file":   export_file,
        "timer":         timer,
    }