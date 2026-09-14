"""
UI-4: Pipeline Progress Tracker

Renders a live step-by-step status board that updates every 3 seconds.
Each pipeline step is shown as a card with:
    - Status icon  (⏳ pending | 🔄 running | ✅ ok | ❌ failed | ⏭ skipped)
    - Step name
    - Duration (once completed)
    - Message / diagnosis code (if available)

A summary row at the top shows overall progress metrics.
"""

from __future__ import annotations

from datetime import datetime

import gradio as gr

from ui.state import RunStatus, StepRecord, UIState


# ---------------------------------------------------------------------------
# Step metadata — display names and descriptions
# ---------------------------------------------------------------------------

_STEP_META: dict[str, dict] = {
    "pdb2gmx":         {"label": "pdb2gmx",          "desc": "Generate topology"},
    "editconf":        {"label": "editconf",          "desc": "Define box"},
    "solvate":         {"label": "solvate",           "desc": "Add water"},
    "grompp_ions":     {"label": "grompp (ions)",     "desc": "Prepare for genion"},
    "genion":          {"label": "genion",            "desc": "Add ions"},
    "grompp_em":       {"label": "grompp (EM)",       "desc": "Prepare EM"},
    "mdrun_em":        {"label": "mdrun (EM)",        "desc": "Energy minimisation"},
    "parse_em":        {"label": "parse_gmx_log (EM)","desc": "Diagnose EM"},
    "grompp_nvt":      {"label": "grompp (NVT)",      "desc": "Prepare NVT"},
    "mdrun_nvt":       {"label": "mdrun (NVT)",       "desc": "NVT equilibration"},
    "parse_nvt":       {"label": "parse_gmx_log (NVT)","desc": "Diagnose NVT"},
    "grompp_npt":      {"label": "grompp (NPT)",      "desc": "Prepare NPT"},
    "mdrun_npt":       {"label": "mdrun (NPT)",       "desc": "NPT equilibration"},
    "parse_npt":       {"label": "parse_gmx_log (NPT)","desc": "Diagnose NPT"},
    "grompp_md":       {"label": "grompp (MD)",       "desc": "Prepare production"},
    "mdrun_md":        {"label": "mdrun (MD)",        "desc": "Production MD"},
    "parse_md":        {"label": "parse_gmx_log (MD)","desc": "Diagnose production"},
    "energy_analysis": {"label": "energy_analysis",   "desc": "Extract energies"},
    "rmsd_analysis":   {"label": "rmsd_analysis",     "desc": "Calculate RMSD"},
}

_STATUS_ICON: dict[str, str] = {
    "pending": "⏳",
    "running": "🔄",
    "ok":      "✅",
    "failed":  "❌",
    "skipped": "⏭️",
}

_STATUS_COLOUR: dict[str, str] = {
    "pending": "#6b7280",
    "running": "#d97706",
    "ok":      "#059669",
    "failed":  "#dc2626",
    "skipped": "#9ca3af",
}

_STATUS_BG: dict[str, str] = {
    "pending": "#f9fafb",
    "running": "#fffbeb",
    "ok":      "#f0fdf4",
    "failed":  "#fef2f2",
    "skipped": "#f3f4f6",
}

_STATUS_BORDER: dict[str, str] = {
    "pending": "#e5e7eb",
    "running": "#d97706",
    "ok":      "#059669",
    "failed":  "#dc2626",
    "skipped": "#d1d5db",
}

# Pipeline stage groupings for visual separation
_STAGE_GROUPS: list[dict] = [
    {
        "label": "🔬 System Preparation",
        "steps": ["pdb2gmx", "editconf", "solvate",
                  "grompp_ions", "genion"],
    },
    {
        "label": "⚡ Energy Minimisation",
        "steps": ["grompp_em", "mdrun_em", "parse_em"],
    },
    {
        "label": "🌡️  NVT Equilibration",
        "steps": ["grompp_nvt", "mdrun_nvt", "parse_nvt"],
    },
    {
        "label": "🔩 NPT Equilibration",
        "steps": ["grompp_npt", "mdrun_npt", "parse_npt"],
    },
    {
        "label": "🚀 Production MD",
        "steps": ["grompp_md", "mdrun_md", "parse_md"],
    },
    {
        "label": "📊 Analysis",
        "steps": ["energy_analysis", "rmsd_analysis"],
    },
]


# ---------------------------------------------------------------------------
# HTML builders
# ---------------------------------------------------------------------------

def _step_card_html(step: StepRecord) -> str:
    """Render a single step as an HTML card."""
    meta    = _STEP_META.get(step.name, {"label": step.name, "desc": ""})
    icon    = _STATUS_ICON.get(step.status,   "⏳")
    colour  = _STATUS_COLOUR.get(step.status, "#6b7280")
    bg      = _STATUS_BG.get(step.status,     "#f9fafb")
    border  = _STATUS_BORDER.get(step.status, "#e5e7eb")

    # Duration string
    duration_str = ""
    if step.started_at and step.ended_at:
        try:
            start = datetime.fromisoformat(step.started_at)
            end   = datetime.fromisoformat(step.ended_at)
            secs  = (end - start).total_seconds()
            if secs < 60:
                duration_str = f"{secs:.1f}s"
            else:
                m = int(secs // 60)
                s = int(secs % 60)
                duration_str = f"{m}m {s:02d}s"
        except Exception:
            pass
    elif step.started_at and step.status == "running":
        try:
            start = datetime.fromisoformat(step.started_at)
            secs  = (datetime.now() - start).total_seconds()
            duration_str = f"{secs:.0f}s…"
        except Exception:
            pass

    # Message (truncated)
    msg_html = ""
    if step.message:
        msg = step.message[:120] + ("…" if len(step.message) > 120 else "")
        msg_html = f"""
        <div style="
            font-size: 0.75rem;
            color: #6b7280;
            margin-top: 4px;
            font-family: monospace;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        ">{msg}</div>
        """

    # Running spinner animation
    spinner = ""
    if step.status == "running":
        spinner = """
        <style>
        @keyframes spin {
            0%   { transform: rotate(0deg);   }
            100% { transform: rotate(360deg); }
        }
        .spin { display: inline-block; animation: spin 1.5s linear infinite; }
        </style>
        """

    return f"""
    {spinner}
    <div style="
        border: 1px solid {border};
        border-left: 4px solid {border};
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 6px;
        background: {bg};
        display: flex;
        align-items: center;
        gap: 12px;
    ">
        <span style="font-size: 1.1rem; min-width: 24px; text-align: center;">
            {'<span class="spin">🔄</span>' if step.status == "running" else icon}
        </span>
        <div style="flex: 1; min-width: 0;">
            <div style="display: flex; align-items: baseline; gap: 8px;">
                <span style="
                    font-weight: 600;
                    font-size: 0.85rem;
                    color: {colour};
                    font-family: monospace;
                ">{meta['label']}</span>
                <span style="
                    font-size: 0.75rem;
                    color: #9ca3af;
                ">{meta['desc']}</span>
            </div>
            {msg_html}
        </div>
        <div style="
            font-size: 0.75rem;
            color: #9ca3af;
            white-space: nowrap;
            font-family: monospace;
            min-width: 60px;
            text-align: right;
        ">{duration_str}</div>
    </div>
    """


def _stage_group_html(
    group:      dict,
    steps_dict: dict[str, StepRecord],
) -> str:
    """Render a stage group (label + its step cards)."""
    step_cards = "".join(
        _step_card_html(steps_dict[name])
        for name in group["steps"]
        if name in steps_dict
    )

    # Compute group status
    group_steps  = [
        steps_dict[n] for n in group["steps"] if n in steps_dict
    ]
    n_ok      = sum(1 for s in group_steps if s.status == "ok")
    n_failed  = sum(1 for s in group_steps if s.status == "failed")
    n_running = sum(1 for s in group_steps if s.status == "running")
    n_total   = len(group_steps)

    if n_failed > 0:
        group_colour = "#dc2626"
        group_badge  = f"<span style='color:#dc2626'>❌ {n_failed} failed</span>"
    elif n_running > 0:
        group_colour = "#d97706"
        group_badge  = f"<span style='color:#d97706'>🔄 running</span>"
    elif n_ok == n_total:
        group_colour = "#059669"
        group_badge  = f"<span style='color:#059669'>✅ complete</span>"
    else:
        group_colour = "#9ca3af"
        group_badge  = (
            f"<span style='color:#9ca3af'>{n_ok}/{n_total} done</span>"
        )

    return f"""
    <div style="margin-bottom: 20px;">
        <div style="
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 8px;
            padding-bottom: 4px;
            border-bottom: 2px solid {group_colour}22;
        ">
            <span style="
                font-weight: 700;
                font-size: 0.9rem;
                color: {group_colour};
            ">{group['label']}</span>
            <span style="font-size: 0.78rem;">{group_badge}</span>
        </div>
        {step_cards}
    </div>
    """


def _summary_bar_html(
    steps:   list[StepRecord],
    status:  RunStatus,
    started: str,
    ended:   str,
) -> str:
    """Render the top summary metrics bar."""
    n_total   = len(steps)
    n_ok      = sum(1 for s in steps if s.status == "ok")
    n_failed  = sum(1 for s in steps if s.status == "failed")
    n_running = sum(1 for s in steps if s.status == "running")
    n_pending = sum(1 for s in steps if s.status == "pending")
    n_skipped = sum(1 for s in steps if s.status == "skipped")

    pct = int((n_ok / n_total) * 100) if n_total > 0 else 0

    # Progress bar colour
    if n_failed > 0:
        bar_colour = "#dc2626"
    elif status == RunStatus.COMPLETED:
        bar_colour = "#059669"
    else:
        bar_colour = "#0d9488"

    # Elapsed
    elapsed = "—"
    if started:
        try:
            start = datetime.fromisoformat(started)
            end   = datetime.fromisoformat(ended) if ended else datetime.now()
            secs  = int((end - start).total_seconds())
            h, r  = divmod(secs, 3600)
            m, s  = divmod(r, 60)
            elapsed = f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"
        except Exception:
            pass

    # Status label
    status_labels = {
        RunStatus.IDLE:      ("⚪", "#6b7280", "Idle"),
        RunStatus.RUNNING:   ("🟡", "#d97706", "Running"),
        RunStatus.COMPLETED: ("🟢", "#059669", "Completed"),
        RunStatus.FAILED:    ("🔴", "#dc2626", "Failed"),
        RunStatus.STOPPED:   ("🟠", "#d97706", "Stopped"),
    }
    icon, colour, label = status_labels.get(
        status, ("⚪", "#6b7280", "Unknown")
    )

    def _metric(value: str, label: str, colour: str = "#0d9488") -> str:
        return f"""
        <div style="
            text-align: center;
            padding: 10px 16px;
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            min-width: 80px;
        ">
            <div style="
                font-size: 1.3rem;
                font-weight: 700;
                color: {colour};
            ">{value}</div>
            <div style="
                font-size: 0.72rem;
                color: #9ca3af;
                margin-top: 2px;
            ">{label}</div>
        </div>
        """

    metrics_html = f"""
    <div style="
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-bottom: 16px;
        align-items: center;
    ">
        {_metric(f"{icon} {label}", "Status",    colour)}
        {_metric(str(n_ok),         "Completed", "#059669")}
        {_metric(str(n_running),    "Running",   "#d97706")}
        {_metric(str(n_failed),     "Failed",    "#dc2626" if n_failed else "#9ca3af")}
        {_metric(str(n_pending),    "Pending",   "#6b7280")}
        {_metric(elapsed,           "Elapsed",   "#0d9488")}
        {_metric(f"{pct}%",         "Progress",  bar_colour)}
    </div>
    """

    progress_bar = f"""
    <div style="margin-bottom: 20px;">
        <div style="
            display: flex;
            justify-content: space-between;
            font-size: 0.78rem;
            color: #6b7280;
            margin-bottom: 4px;
        ">
            <span>Pipeline progress</span>
            <span>{n_ok} / {n_total} steps complete</span>
        </div>
        <div style="
            background: #e5e7eb;
            border-radius: 999px;
            height: 10px;
            overflow: hidden;
        ">
            <div style="
                background: {bar_colour};
                width: {pct}%;
                height: 100%;
                border-radius: 999px;
                transition: width 0.5s ease;
            "></div>
        </div>
    </div>
    """

    return metrics_html + progress_bar


def build_progress_html(ui_state: UIState) -> str:
    """
    Build the full progress panel HTML from current UIState.
    Called on every timer tick.
    """
    with ui_state.lock:
        steps      = list(ui_state.steps)
        status     = ui_state.status
        started_at = ui_state.started_at
        ended_at   = ui_state.ended_at

    if status == RunStatus.IDLE and not any(
        s.status != "pending" for s in steps
    ):
        return """
        <div style="
            text-align: center;
            padding: 60px 20px;
            color: #9ca3af;
        ">
            <div style="font-size: 2rem; margin-bottom: 12px;">🧬</div>
            <div style="font-size: 1rem; font-weight: 600;">
                No run in progress
            </div>
            <div style="font-size: 0.85rem; margin-top: 6px;">
                Configure a run in the
                <strong>⚙️ Run Configuration</strong> tab and click
                <strong>▶ Start Run</strong>.
            </div>
        </div>
        """

    # Build steps lookup
    steps_dict = {s.name: s for s in steps}

    # Summary bar
    html = _summary_bar_html(steps, status, started_at, ended_at)

    # Stage groups
    for group in _STAGE_GROUPS:
        html += _stage_group_html(group, steps_dict)

    return html


# ---------------------------------------------------------------------------
# Tab builder
# ---------------------------------------------------------------------------

def build_progress_tab(ui_state: UIState) -> dict:
    """Build the Pipeline Progress tab. Returns component references."""

    gr.Markdown("## 📋 Pipeline Progress")

    # ── Top action row ────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown(
                "_Updates every 3 seconds during a run. "
                "Step durations are shown once each step completes._"
            )
        with gr.Column(scale=1):
            refresh_btn = gr.Button(
                "🔄  Refresh Now",
                variant = "secondary",
                size    = "sm",
            )

    # ── Main progress display ─────────────────────────────────────────────
    progress_html = gr.HTML(
        value    = build_progress_html(ui_state),
        elem_id  = "progress-panel",
    )

    # ── Step detail expander ──────────────────────────────────────────────
    with gr.Accordion("🔍 Step Detail Inspector", open=False):
        gr.Markdown(
            "Select a step name to see its full message and timing."
        )
        with gr.Row():
            step_selector = gr.Dropdown(
                label   = "Select step",
                choices = list(_STEP_META.keys()),
                value   = None,
            )
        step_detail_box = gr.Textbox(
            label       = "Step detail",
            lines       = 6,
            max_lines   = 6,
            interactive = False,
        )

    # ── Timer — polls every 3 seconds ─────────────────────────────────────
    timer = gr.Timer(value=3)

    # ── Poll function ─────────────────────────────────────────────────────

    def poll_progress() -> str:
        return build_progress_html(ui_state)

    def on_refresh() -> str:
        return build_progress_html(ui_state)

    def on_step_select(step_name: str) -> str:
        if not step_name:
            return ""
        with ui_state.lock:
            steps_dict = {s.name: s for s in ui_state.steps}
        step = steps_dict.get(step_name)
        if not step:
            return f"Step '{step_name}' not found."

        meta = _STEP_META.get(step_name, {"label": step_name, "desc": ""})
        lines = [
            f"Step:       {meta['label']}",
            f"Description:{meta['desc']}",
            f"Status:     {step.status}",
            f"Started:    {step.started_at or '—'}",
            f"Ended:      {step.ended_at   or '—'}",
            f"Message:    {step.message    or '—'}",
        ]
        return "\n".join(lines)

    # ── Wire timer ────────────────────────────────────────────────────────

    timer.tick(
        fn      = poll_progress,
        inputs  = [],
        outputs = [progress_html],
    )

    refresh_btn.click(
        fn      = on_refresh,
        inputs  = [],
        outputs = [progress_html],
    )

    step_selector.change(
        fn      = on_step_select,
        inputs  = [step_selector],
        outputs = [step_detail_box],
    )

    return {
        "progress_html":  progress_html,
        "refresh_btn":    refresh_btn,
        "step_selector":  step_selector,
        "step_detail_box": step_detail_box,
        "timer":          timer,
    }