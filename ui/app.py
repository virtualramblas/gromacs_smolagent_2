"""
GROMACS Agent — Gradio UI entry point.

Launch:
    python ui/app.py
    python ui/app.py --port 7860 --share
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import gradio as gr

from ui.tabs.run_config   import build_run_config_tab
from ui.tabs.live_log     import build_live_log_tab
from ui.tabs.progress     import build_progress_tab
from ui.tabs.results      import build_results_tab
from ui.tabs.recovery     import build_recovery_tab
from ui.state             import UIState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("gromacs_ui")


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

THEME = gr.themes.Soft(
    primary_hue   = "teal",
    secondary_hue = "slate",
    neutral_hue   = "slate",
    font          = gr.themes.GoogleFont("Inter"),
).set(
    button_primary_background_fill         = "#0d9488",
    button_primary_background_fill_hover   = "#0f766e",
    button_primary_text_color              = "white",
    block_label_text_size                  = "sm",
    block_title_text_size                  = "md",
)


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
/* ── General ─────────────────────────────────────────────── */
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
}

/* ── Header ──────────────────────────────────────────────── */
.app-header {
    background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%);
    color: white;
    padding: 20px 28px;
    border-radius: 12px;
    margin-bottom: 16px;
}
.app-header h1 {
    margin: 0 0 4px 0;
    font-size: 1.6rem;
    font-weight: 700;
}
.app-header p {
    margin: 0;
    opacity: 0.85;
    font-size: 0.9rem;
}

/* ── Status badges ───────────────────────────────────────── */
.status-ok         { color: #059669; font-weight: 600; }
.status-running    { color: #d97706; font-weight: 600; }
.status-failed     { color: #dc2626; font-weight: 600; }
.status-pending    { color: #6b7280; font-weight: 600; }
.status-skipped    { color: #9ca3af; font-weight: 600; }

/* ── Step cards ──────────────────────────────────────────── */
.step-card {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 6px;
    background: #f9fafb;
    font-family: monospace;
    font-size: 0.85rem;
}
.step-card.running  { border-left: 4px solid #d97706; background: #fffbeb; }
.step-card.ok       { border-left: 4px solid #059669; background: #f0fdf4; }
.step-card.failed   { border-left: 4px solid #dc2626; background: #fef2f2; }

/* ── Log panel ───────────────────────────────────────────── */
.log-panel textarea {
    font-family: "JetBrains Mono", "Fira Code", monospace !important;
    font-size: 0.78rem !important;
    background: #0f172a !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}

/* ── Recovery panel ──────────────────────────────────────── */
.recovery-card {
    border: 1px solid #fbbf24;
    border-radius: 8px;
    padding: 12px 16px;
    background: #fffbeb;
    margin-bottom: 8px;
}

/* ── Metric cards ────────────────────────────────────────── */
.metric-card {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 14px;
    text-align: center;
    background: white;
}
.metric-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #0d9488;
}
.metric-label {
    font-size: 0.78rem;
    color: #6b7280;
    margin-top: 2px;
}

/* ── Buttons ─────────────────────────────────────────────── */
.btn-run    { background: #0d9488 !important; }
.btn-stop   { background: #dc2626 !important; }
.btn-resume { background: #d97706 !important; }
"""


# ---------------------------------------------------------------------------
# Header HTML
# ---------------------------------------------------------------------------

HEADER_HTML = """
<div class="app-header">
    <h1>🧬 GROMACS MD Simulation Agent</h1>
    <p>
        Agentic molecular dynamics pipeline powered by LLM reasoning
        and automated error recovery.
    </p>
</div>
"""


# ---------------------------------------------------------------------------
# App builder
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    ui_state = UIState()

    with gr.Blocks(
        theme=THEME,
        css=CSS,
        title="GROMACS Agent",
        analytics_enabled=False,
    ) as app:

        gr.HTML(HEADER_HTML)

        with gr.Tabs(elem_id="main-tabs") as tabs:

            with gr.TabItem("⚙️  Run Configuration", id="tab-config"):
                config_components = build_run_config_tab(ui_state)

            with gr.TabItem("📋  Pipeline Progress", id="tab-progress"):
                progress_components = build_progress_tab(ui_state)

            with gr.TabItem("📜  Live Log", id="tab-log"):
                log_components = build_live_log_tab(ui_state)

            with gr.TabItem("🔧  Recovery Events", id="tab-recovery"):
                recovery_components = build_recovery_tab(ui_state)

            with gr.TabItem("📊  Results", id="tab-results"):
                results_components = build_results_tab(ui_state)

        # ── Footer ────────────────────────────────────────────────────────
        gr.HTML("""
        <div style="text-align:center; color:#9ca3af;
                    font-size:0.75rem; margin-top:16px; padding:8px;">
            GROMACS Agent UI &nbsp;|&nbsp;
            smolagents 1.26.0 &nbsp;|&nbsp;
            <a href="http://go/gpteal/" style="color:#0d9488;">GPTeal</a>
        </div>
        """)

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="GROMACS Agent Gradio UI")
    parser.add_argument("--port",  type=int,  default=7860)
    parser.add_argument("--host",  type=str,  default="0.0.0.0")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio share link")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Starting GROMACS Agent UI on %s:%d", args.host, args.port)

    app = build_app()
    app.launch(
        server_name = args.host,
        server_port = args.port,
        share       = args.share,
        show_error  = True,
    )


if __name__ == "__main__":
    main()