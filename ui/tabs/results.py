"""
UI-5: Results Viewer

Displays simulation results after (or during) a run:
    - Key metrics summary (Fmax, temperature, pressure)
    - Potential energy profile (EM + production)
    - Temperature profile (NVT)
    - Pressure profile (NPT)
    - RMSD vs time (production)
    - Raw pipeline state JSON inspector
    - Output file browser

Plots are built with matplotlib and returned as gr.Plot objects.
The tab polls UIState every 5 seconds for new data.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import gradio as gr

from ui.state import RunStatus, UIState

# ---------------------------------------------------------------------------
# Optional matplotlib import — graceful fallback if not available
# ---------------------------------------------------------------------------

try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------

_PLOT_STYLE = {
    "figure.facecolor":  "#0f172a",
    "axes.facecolor":    "#1e293b",
    "axes.edgecolor":    "#334155",
    "axes.labelcolor":   "#cbd5e1",
    "axes.titlecolor":   "#f1f5f9",
    "xtick.color":       "#94a3b8",
    "ytick.color":       "#94a3b8",
    "grid.color":        "#334155",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "text.color":        "#f1f5f9",
    "lines.linewidth":   2.0,
    "figure.dpi":        110,
}

_COLOURS = {
    "potential":    "#34d399",   # green
    "kinetic":      "#60a5fa",   # blue
    "total":        "#a78bfa",   # purple
    "temperature":  "#f59e0b",   # amber
    "pressure":     "#fb7185",   # rose
    "rmsd":         "#38bdf8",   # sky
}


# ---------------------------------------------------------------------------
# Plot builders
# ---------------------------------------------------------------------------

def _apply_style() -> None:
    """Apply dark plot style."""
    for key, val in _PLOT_STYLE.items():
        try:
            plt.rcParams[key] = val
        except Exception:
            pass


def _no_data_figure(message: str = "No data yet"):
    """Return a placeholder figure when data is not available."""
    if not _HAS_MPL:
        return None
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.text(
        0.5, 0.5, message,
        transform   = ax.transAxes,
        ha          = "center",
        va          = "center",
        fontsize    = 13,
        color       = "#64748b",
        style       = "italic",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def build_energy_plot(energy_data: dict[str, list[float]]) -> Any:
    """
    Build potential / kinetic / total energy vs step plot.

    Args:
        energy_data: dict mapping term name → list of values.
                     Keys expected: "Potential", "Kinetic-En.", "Total-Energy"
                     Values are in kJ/mol.
    """
    if not _HAS_MPL:
        return None

    if not energy_data:
        return _no_data_figure("Energy data not yet available.\nRun will populate this after mdrun.")

    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 3.5))

    term_map = {
        "Potential":    ("Potential Energy",  _COLOURS["potential"]),
        "Kinetic-En.":  ("Kinetic Energy",    _COLOURS["kinetic"]),
        "Total-Energy": ("Total Energy",      _COLOURS["total"]),
    }

    plotted = False
    for term, (label, colour) in term_map.items():
        values = energy_data.get(term, [])
        if not values:
            continue
        steps = list(range(len(values)))
        ax.plot(steps, values, label=label, color=colour, linewidth=1.8)
        plotted = True

    if not plotted:
        return _no_data_figure("No energy terms found in data.")

    ax.set_xlabel("Frame", fontsize=10)
    ax.set_ylabel("Energy (kJ/mol)", fontsize=10)
    ax.set_title("Energy Profile", fontsize=12, fontweight="bold", pad=10)
    ax.legend(
        fontsize    = 8,
        framealpha  = 0.3,
        facecolor   = "#1e293b",
        edgecolor   = "#334155",
    )
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x/1000:.1f}k")
    )
    fig.tight_layout()
    return fig


def build_temperature_plot(
    temp_values: list[float],
    temp_target: float | None = 300.0,
) -> Any:
    """
    Build temperature vs frame plot with target reference line.
    """
    if not _HAS_MPL:
        return None

    if not temp_values:
        return _no_data_figure(
            "Temperature data not yet available.\nPopulated after NVT mdrun."
        )

    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 3.0))

    frames = list(range(len(temp_values)))
    ax.plot(
        frames, temp_values,
        color     = _COLOURS["temperature"],
        linewidth = 1.6,
        label     = "Temperature",
        alpha     = 0.9,
    )

    if temp_target is not None:
        ax.axhline(
            y         = temp_target,
            color     = "#f1f5f9",
            linestyle = "--",
            linewidth = 1.2,
            alpha     = 0.6,
            label     = f"Target ({temp_target:.0f} K)",
        )
        # Tolerance band ±5 K
        ax.axhspan(
            temp_target - 5, temp_target + 5,
            alpha     = 0.08,
            color     = _COLOURS["temperature"],
            label     = "±5 K tolerance",
        )

    ax.set_xlabel("Frame", fontsize=10)
    ax.set_ylabel("Temperature (K)", fontsize=10)
    ax.set_title("Temperature Profile (NVT)", fontsize=12,
                 fontweight="bold", pad=10)
    ax.legend(
        fontsize   = 8,
        framealpha = 0.3,
        facecolor  = "#1e293b",
        edgecolor  = "#334155",
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def build_pressure_plot(
    pres_values: list[float],
    pres_target: float | None = 1.0,
) -> Any:
    """
    Build pressure vs frame plot with target reference line.
    """
    if not _HAS_MPL:
        return None

    if not pres_values:
        return _no_data_figure(
            "Pressure data not yet available.\nPopulated after NPT mdrun."
        )

    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 3.0))

    frames = list(range(len(pres_values)))
    ax.plot(
        frames, pres_values,
        color     = _COLOURS["pressure"],
        linewidth = 1.6,
        label     = "Pressure",
        alpha     = 0.9,
    )

    if pres_target is not None:
        ax.axhline(
            y         = pres_target,
            color     = "#f1f5f9",
            linestyle = "--",
            linewidth = 1.2,
            alpha     = 0.6,
            label     = f"Target ({pres_target:.1f} bar)",
        )

    ax.set_xlabel("Frame", fontsize=10)
    ax.set_ylabel("Pressure (bar)", fontsize=10)
    ax.set_title("Pressure Profile (NPT)", fontsize=12,
                 fontweight="bold", pad=10)
    ax.legend(
        fontsize   = 8,
        framealpha = 0.3,
        facecolor  = "#1e293b",
        edgecolor  = "#334155",
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def build_rmsd_plot(rmsd_data: list[tuple[float, float]]) -> Any:
    """
    Build RMSD vs time plot.

    Args:
        rmsd_data: list of (time_ps, rmsd_nm) tuples.
    """
    if not _HAS_MPL:
        return None

    if not rmsd_data:
        return _no_data_figure(
            "RMSD data not yet available.\nPopulated after production MD."
        )

    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 3.5))

    times = [t for t, _ in rmsd_data]
    rmsds = [r for _, r in rmsd_data]

    ax.plot(
        times, rmsds,
        color     = _COLOURS["rmsd"],
        linewidth = 1.8,
        label     = "Backbone RMSD",
    )

    # Rolling mean overlay (window = 10% of data)
    window = max(1, len(rmsds) // 10)
    if len(rmsds) >= window * 2:
        rolling = [
            sum(rmsds[max(0, i - window): i + 1]) /
            len(rmsds[max(0, i - window): i + 1])
            for i in range(len(rmsds))
        ]
        ax.plot(
            times, rolling,
            color     = "#f1f5f9",
            linewidth = 1.2,
            linestyle = "--",
            alpha     = 0.7,
            label     = f"Rolling mean (w={window})",
        )

    ax.set_xlabel("Time (ps)", fontsize=10)
    ax.set_ylabel("RMSD (nm)", fontsize=10)
    ax.set_title("Backbone RMSD — Production MD", fontsize=12,
                 fontweight="bold", pad=10)
    ax.legend(
        fontsize   = 8,
        framealpha = 0.3,
        facecolor  = "#1e293b",
        edgecolor  = "#334155",
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Metric card HTML
# ---------------------------------------------------------------------------

def _metric_card(
    value:    str,
    label:    str,
    sublabel: str  = "",
    colour:   str  = "#0d9488",
    status:   str  = "",
) -> str:
    """Render a single metric card as HTML."""
    status_html = ""
    if status:
        status_colours = {
            "ok":      "#059669",
            "warning": "#d97706",
            "error":   "#dc2626",
            "pending": "#9ca3af",
        }
        sc = status_colours.get(status, "#9ca3af")
        status_icons = {
            "ok":      "✅",
            "warning": "⚠️",
            "error":   "❌",
            "pending": "⏳",
        }
        si = status_icons.get(status, "")
        status_html = f"""
        <div style="
            font-size: 0.72rem;
            color: {sc};
            margin-top: 4px;
            font-weight: 600;
        ">{si} {status.upper()}</div>
        """

    sublabel_html = ""
    if sublabel:
        sublabel_html = f"""
        <div style="
            font-size: 0.72rem;
            color: #64748b;
            margin-top: 2px;
        ">{sublabel}</div>
        """

    return f"""
    <div style="
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px 20px;
        background: #1e293b;
        text-align: center;
        min-width: 130px;
        flex: 1;
    ">
        <div style="
            font-size: 1.5rem;
            font-weight: 700;
            color: {colour};
            font-family: monospace;
        ">{value}</div>
        <div style="
            font-size: 0.78rem;
            color: #94a3b8;
            margin-top: 4px;
        ">{label}</div>
        {sublabel_html}
        {status_html}
    </div>
    """


def build_metrics_html(ui_state: UIState) -> str:
    """Build the key metrics summary row HTML."""
    with ui_state.lock:
        em_fmax       = ui_state.em_fmax
        em_epot       = ui_state.em_epot
        nvt_temp      = ui_state.nvt_temp_mean
        npt_pres      = ui_state.npt_pres_mean
        em_converged  = getattr(ui_state, "em_converged_flag", None)
        nvt_complete  = getattr(ui_state, "nvt_complete_flag", None)
        npt_complete  = getattr(ui_state, "npt_complete_flag", None)
        md_complete   = getattr(ui_state, "md_complete_flag",  None)

    # EM Fmax
    if em_fmax is not None:
        fmax_val    = f"{em_fmax:.1f}"
        fmax_status = "ok" if em_fmax < 1000.0 else "error"
        fmax_sub    = "kJ/mol/nm"
    else:
        fmax_val    = "—"
        fmax_status = "pending"
        fmax_sub    = "kJ/mol/nm"

    # EM Epot
    if em_epot is not None:
        epot_val = f"{em_epot/1000:.1f}k"
        epot_sub = "kJ/mol"
    else:
        epot_val = "—"
        epot_sub = "kJ/mol"

    # NVT temperature
    if nvt_temp is not None:
        temp_val    = f"{nvt_temp:.1f}"
        temp_status = "ok" if abs(nvt_temp - 300.0) < 5.0 else "warning"
        temp_sub    = "K (target 300)"
    else:
        temp_val    = "—"
        temp_status = "pending"
        temp_sub    = "K"

    # NPT pressure
    if npt_pres is not None:
        pres_val    = f"{npt_pres:.1f}"
        pres_status = "ok" if abs(npt_pres - 1.0) < 200.0 else "warning"
        pres_sub    = "bar (target 1.0)"
    else:
        pres_val    = "—"
        pres_status = "pending"
        pres_sub    = "bar"

    cards = "".join([
        _metric_card(fmax_val, "EM Fmax",      fmax_sub, "#34d399", fmax_status),
        _metric_card(epot_val, "EM Epot",      epot_sub, "#60a5fa"),
        _metric_card(temp_val, "NVT Temp",     temp_sub, "#f59e0b", temp_status),
        _metric_card(pres_val, "NPT Pressure", pres_sub, "#fb7185", pres_status),
    ])

    return f"""
    <div style="
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin-bottom: 8px;
    ">
        {cards}
    </div>
    """


# ---------------------------------------------------------------------------
# File browser HTML
# ---------------------------------------------------------------------------

def build_file_browser_html(ui_state: UIState) -> str:
    """Build an HTML table of output files from UIState."""
    with ui_state.lock:
        state_file = ui_state.state_file
        work_dir   = ui_state.work_dir

    if not state_file or not Path(state_file).exists():
        return """
        <div style="color:#64748b; font-style:italic; padding:12px;">
            No output files yet. Files will appear here after the run starts.
        </div>
        """

    try:
        state = json.loads(Path(state_file).read_text())
        files = state.get("files", {})
    except Exception:
        return "<div style='color:#ef4444;'>Could not read state file.</div>"

    if not any(v for v in files.values()):
        return """
        <div style="color:#64748b; font-style:italic; padding:12px;">
            No output files recorded yet.
        </div>
        """

    rows = ""
    file_labels = {
        "pdb":      "Input PDB",
        "gro":      "Coordinates (GRO)",
        "gro_box":  "Boxed coordinates",
        "gro_solv": "Solvated coordinates",
        "gro_ions": "Ionised coordinates",
        "top":      "Topology",
        "tpr_em":   "EM run input",
        "tpr_nvt":  "NVT run input",
        "tpr_npt":  "NPT run input",
        "tpr_md":   "MD run input",
        "cpt":      "Checkpoint",
        "edr_em":   "EM energy file",
        "edr_md":   "MD energy file",
        "xtc":      "Trajectory (XTC)",
    }

    for key, label in file_labels.items():
        path_str = files.get(key)
        if not path_str:
            continue
        path   = Path(path_str)
        exists = path.exists()
        size   = ""
        if exists:
            sz = path.stat().st_size
            if sz < 1024:
                size = f"{sz} B"
            elif sz < 1024 ** 2:
                size = f"{sz/1024:.1f} KB"
            else:
                size = f"{sz/1024**2:.1f} MB"

        icon   = "✅" if exists else "❌"
        colour = "#34d399" if exists else "#ef4444"
        rows  += f"""
        <tr>
            <td style="padding:6px 10px; color:#94a3b8;
                       font-size:0.78rem;">{label}</td>
            <td style="padding:6px 10px; font-family:monospace;
                       font-size:0.75rem; color:#cbd5e1;">
                {path.name}
            </td>
            <td style="padding:6px 10px; text-align:center;">{icon}</td>
            <td style="padding:6px 10px; color:{colour};
                       font-size:0.75rem; text-align:right;">{size}</td>
        </tr>
        """

    return f"""
    <table style="
        width: 100%;
        border-collapse: collapse;
        background: #1e293b;
        border-radius: 8px;
        overflow: hidden;
    ">
        <thead>
            <tr style="background:#0f172a;">
                <th style="padding:8px 10px; text-align:left;
                           color:#64748b; font-size:0.78rem;
                           font-weight:600;">File type</th>
                <th style="padding:8px 10px; text-align:left;
                           color:#64748b; font-size:0.78rem;
                           font-weight:600;">Filename</th>
                <th style="padding:8px 10px; text-align:center;
                           color:#64748b; font-size:0.78rem;
                           font-weight:600;">Exists</th>
                <th style="padding:8px 10px; text-align:right;
                           color:#64748b; font-size:0.78rem;
                           font-weight:600;">Size</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>
    """


# ---------------------------------------------------------------------------
# Tab builder
# ---------------------------------------------------------------------------

def build_results_tab(ui_state: UIState) -> dict:
    """Build the Results tab. Returns component references."""

    gr.Markdown("## 📊 Results")

    # ── Key metrics ───────────────────────────────────────────────────────
    gr.Markdown("### Key Metrics")
    metrics_html = gr.HTML(
        value   = build_metrics_html(ui_state),
        elem_id = "metrics-panel",
    )

    # ── Plots ─────────────────────────────────────────────────────────────
    gr.Markdown("### Energy & Dynamics Profiles")

    with gr.Tabs():

        with gr.TabItem("⚡ Potential Energy"):
            energy_plot = gr.Plot(
                value   = _no_data_figure(
                    "Energy data not yet available.\n"
                    "Populated after mdrun (EM or production)."
                ) if _HAS_MPL else None,
                label   = "Potential Energy Profile",
            )

        with gr.TabItem("🌡️  Temperature"):
            temp_plot = gr.Plot(
                value   = _no_data_figure(
                    "Temperature data not yet available.\n"
                    "Populated after NVT mdrun."
                ) if _HAS_MPL else None,
                label   = "Temperature Profile",
            )

        with gr.TabItem("🔩 Pressure"):
            pres_plot = gr.Plot(
                value   = _no_data_figure(
                    "Pressure data not yet available.\n"
                    "Populated after NPT mdrun."
                ) if _HAS_MPL else None,
                label   = "Pressure Profile",
            )

        with gr.TabItem("📈 RMSD"):
            rmsd_plot = gr.Plot(
                value   = _no_data_figure(
                    "RMSD data not yet available.\n"
                    "Populated after production MD."
                ) if _HAS_MPL else None,
                label   = "Backbone RMSD",
            )

    # ── Output files ──────────────────────────────────────────────────────
    gr.Markdown("### Output Files")
    with gr.Row():
        with gr.Column(scale=3):
            files_html = gr.HTML(
                value   = build_file_browser_html(ui_state),
                elem_id = "files-panel",
            )
        with gr.Column(scale=1):
            refresh_files_btn = gr.Button(
                "🔄  Refresh Files",
                variant = "secondary",
                size    = "sm",
            )

    # ── State inspector ───────────────────────────────────────────────────
    with gr.Accordion("🔍 Pipeline State Inspector (JSON)", open=False):
        gr.Markdown(
            "_Raw pipeline state JSON — useful for debugging "
            "and verifying state persistence._"
        )
        with gr.Row():
            refresh_state_btn = gr.Button(
                "🔄  Refresh State",
                variant = "secondary",
                size    = "sm",
            )
        state_json_box = gr.Code(
            value    = "{}",
            language = "json",
            label    = "Pipeline State",
            lines    = 20,
            interactive = False,
        )

    # ── Final answer ──────────────────────────────────────────────────────
    with gr.Accordion("🏁 Final Agent Answer", open=False):
        final_answer_box = gr.Textbox(
            label       = "Agent final_answer() output",
            lines       = 8,
            max_lines   = 8,
            interactive = False,
            value       = "No final answer yet.",
        )

    # ── Timer — polls every 5 seconds ─────────────────────────────────────
    timer = gr.Timer(value=5)

    # ── Poll functions ────────────────────────────────────────────────────

    def poll_results() -> tuple:
        """
        Called every 5 seconds.
        Returns updated values for all result components.
        """
        with ui_state.lock:
            energy_data  = dict(ui_state.energy_data)
            rmsd_data    = list(ui_state.rmsd_data)
            final_answer = ui_state.final_answer
            state_file   = ui_state.state_file

        # Plots
        e_plot = build_energy_plot(energy_data)
        r_plot = build_rmsd_plot(rmsd_data)

        # Temperature and pressure from state file
        t_plot = _no_data_figure(
            "Temperature data not yet available."
        )
        p_plot = _no_data_figure(
            "Pressure data not yet available."
        )

        if state_file and Path(state_file).exists():
            try:
                state = json.loads(Path(state_file).read_text())
                temp_vals = state.get("temperature_values", [])
                pres_vals = state.get("pressure_values",   [])
                temp_tgt  = state.get("temperature_target", 300.0)
                pres_tgt  = state.get("pressure_target",    1.0)
                if temp_vals:
                    t_plot = build_temperature_plot(temp_vals, temp_tgt)
                if pres_vals:
                    p_plot = build_pressure_plot(pres_vals, pres_tgt)
            except Exception:
                pass

        # Metrics HTML
        metrics = build_metrics_html(ui_state)

        # Files HTML
        files = build_file_browser_html(ui_state)

        # State JSON
        state_json = _read_state_json(state_file)

        # Final answer
        fa = final_answer if final_answer else "No final answer yet."

        return (
            metrics,
            e_plot, t_plot, p_plot, r_plot,
            files,
            state_json,
            fa,
        )

    def _read_state_json(state_file: str) -> str:
        if not state_file:
            return "{}"
        path = Path(state_file)
        if not path.exists():
            return "{}"
        try:
            data = json.loads(path.read_text())
            return json.dumps(data, indent=2)
        except Exception as exc:
            return f'{{"error": "{exc}"}}'

    def on_refresh_files() -> str:
        return build_file_browser_html(ui_state)

    def on_refresh_state() -> str:
        with ui_state.lock:
            sf = ui_state.state_file
        return _read_state_json(sf)

    # ── Wire timer ────────────────────────────────────────────────────────

    timer.tick(
        fn = poll_results,
        inputs  = [],
        outputs = [
            metrics_html,
            energy_plot, temp_plot, pres_plot, rmsd_plot,
            files_html,
            state_json_box,
            final_answer_box,
        ],
    )

    refresh_files_btn.click(
        fn      = on_refresh_files,
        inputs  = [],
        outputs = [files_html],
    )

    refresh_state_btn.click(
        fn      = on_refresh_state,
        inputs  = [],
        outputs = [state_json_box],
    )

    return {
        "metrics_html":      metrics_html,
        "energy_plot":       energy_plot,
        "temp_plot":         temp_plot,
        "pres_plot":         pres_plot,
        "rmsd_plot":         rmsd_plot,
        "files_html":        files_html,
        "state_json_box":    state_json_box,
        "final_answer_box":  final_answer_box,
        "refresh_files_btn": refresh_files_btn,
        "refresh_state_btn": refresh_state_btn,
        "timer":             timer,
    }