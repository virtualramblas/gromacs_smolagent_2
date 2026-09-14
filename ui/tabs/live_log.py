"""
UI-3: Live Log Streaming Panel

Polls ui_state.log_lines every 2 seconds and updates the
log textbox. Provides filtering, search, and download.

Design notes:
    - Gradio's gr.Timer(every=N) triggers a server-side poll
      function that reads from UIState and returns the updated text.
    - Filtering and search are applied client-side on each poll
      so the full log is always preserved in UIState.
    - Download writes the full unfiltered log to a temp file
      and returns it as a gr.File download.
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import gradio as gr

from ui.state import RunStatus, UIState


# ---------------------------------------------------------------------------
# Log level colours (injected as HTML into the log display)
# ---------------------------------------------------------------------------

_LEVEL_COLOURS = {
    "ERROR":   "#ef4444",
    "WARNING": "#f59e0b",
    "INFO":    "#60a5fa",
    "DEBUG":   "#9ca3af",
    "SUCCESS": "#34d399",
    "STEP":    "#a78bfa",
}

_FILTER_CHOICES = ["ALL", "INFO", "WARNING", "ERROR", "DEBUG"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_filter(lines: list[str], level: str, keyword: str) -> list[str]:
    """
    Filter log lines by level and/or keyword.

    Args:
        lines:   Full list of log lines from UIState.
        level:   One of ALL / INFO / WARNING / ERROR / DEBUG.
        keyword: Free-text search string (case-insensitive).

    Returns:
        Filtered list of lines.
    """
    result = lines

    if level != "ALL":
        result = [l for l in result if level.upper() in l.upper()]

    if keyword.strip():
        kw = keyword.strip().lower()
        result = [l for l in result if kw in l.lower()]

    return result


def _format_log_text(lines: list[str]) -> str:
    """Join lines into a single string for the Textbox."""
    if not lines:
        return "Waiting for run to start..."
    return "\n".join(lines)


def _status_badge(status: RunStatus) -> str:
    """Return a Markdown status badge string."""
    badges = {
        RunStatus.IDLE:      "⚪ Idle",
        RunStatus.RUNNING:   "🟡 Running",
        RunStatus.COMPLETED: "🟢 Completed",
        RunStatus.FAILED:    "🔴 Failed",
        RunStatus.STOPPED:   "🟠 Stopped",
    }
    return badges.get(status, "⚪ Unknown")


# ---------------------------------------------------------------------------
# Tab builder
# ---------------------------------------------------------------------------

def build_live_log_tab(ui_state: UIState) -> dict:
    """Build the Live Log tab. Returns component references."""

    gr.Markdown("## 📜 Live Agent Log")

    # ── Controls row ──────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=1):
            status_badge = gr.Markdown(
                value = f"**Status:** {_status_badge(RunStatus.IDLE)}",
            )
        with gr.Column(scale=1):
            line_count_md = gr.Markdown(
                value = "**Lines:** 0",
            )
        with gr.Column(scale=1):
            elapsed_md = gr.Markdown(
                value = "**Elapsed:** —",
            )
        with gr.Column(scale=1):
            step_md = gr.Markdown(
                value = "**Current step:** —",
            )

    # ── Filter row ────────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=1):
            level_filter = gr.Dropdown(
                label   = "Filter by level",
                choices = _FILTER_CHOICES,
                value   = "ALL",
            )
        with gr.Column(scale=3):
            keyword_filter = gr.Textbox(
                label       = "Search keyword",
                placeholder = "e.g. ERROR, grompp, Fmax, WARNING",
                max_lines   = 1,
            )
        with gr.Column(scale=1):
            autoscroll_cb = gr.Checkbox(
                label = "Auto-scroll to bottom",
                value = True,
            )

    # ── Log display ───────────────────────────────────────────────────────
    log_box = gr.Textbox(
        label        = "Agent Log",
        lines        = 35,
        max_lines    = 35,
        interactive  = False,
        elem_classes = ["log-panel"],
        value        = "Waiting for run to start...",
    )

    # ── Action buttons ────────────────────────────────────────────────────
    with gr.Row():
        clear_btn    = gr.Button("🗑  Clear Display",  variant="secondary", scale=1)
        download_btn = gr.Button("💾  Download Log",   variant="secondary", scale=1)
        download_file = gr.File(
            label   = "Log file",
            visible = False,
        )

    # ── LLM step detail expander ──────────────────────────────────────────
    with gr.Accordion("🔍 Last LLM Output", open=False):
        llm_output_box = gr.Textbox(
            label        = "Last code block generated by LLM",
            lines        = 15,
            max_lines    = 15,
            interactive  = False,
            elem_classes = ["log-panel"],
            value        = "No LLM output yet.",
        )

    # ── Timer — polls every 2 seconds ─────────────────────────────────────
    timer = gr.Timer(value=2)

    # ── Poll function ─────────────────────────────────────────────────────

    def poll_log(level: str, keyword: str) -> tuple:
        """
        Called by the timer every 2 seconds.
        Reads UIState and returns updated component values.

        Returns:
            (log_text, status_badge, line_count, elapsed, current_step,
             llm_output)
        """
        with ui_state.lock:
            lines       = list(ui_state.log_lines)
            status      = ui_state.status
            current_step = ui_state.current_step
            started_at  = ui_state.started_at
            ended_at    = ui_state.ended_at

        # Filter
        filtered = _apply_filter(lines, level, keyword)
        log_text = _format_log_text(filtered)

        # Status badge
        badge = f"**Status:** {_status_badge(status)}"

        # Line count
        count_md = f"**Lines:** {len(lines):,} ({len(filtered):,} shown)"

        # Elapsed time
        elapsed = _compute_elapsed(started_at, ended_at, status)

        # Current step
        step = f"**Current step:** `{current_step}`" if current_step else "**Current step:** —"

        # Last LLM output — extract last code block from log
        llm_out = _extract_last_code_block(lines)

        return log_text, badge, count_md, elapsed, step, llm_out

    def _compute_elapsed(
        started_at: str,
        ended_at:   str,
        status:     RunStatus,
    ) -> str:
        if not started_at:
            return "**Elapsed:** —"
        try:
            start = datetime.fromisoformat(started_at)
            if ended_at:
                end = datetime.fromisoformat(ended_at)
            else:
                end = datetime.now()
            delta = end - start
            total = int(delta.total_seconds())
            h, rem = divmod(total, 3600)
            m, s   = divmod(rem, 60)
            if h > 0:
                return f"**Elapsed:** {h}h {m:02d}m {s:02d}s"
            return f"**Elapsed:** {m}m {s:02d}s"
        except Exception:
            return "**Elapsed:** —"

    def _extract_last_code_block(lines: list[str]) -> str:
        """
        Extract the last ```py ... ``` block from the log lines.
        These are the code blocks generated by the LLM.
        """
        full_text = "\n".join(lines)
        # Find all code blocks
        import re
        blocks = re.findall(r"```(?:py|python)\n(.*?)```", full_text, re.DOTALL)
        if blocks:
            return blocks[-1].strip()
        return "No LLM code block found in log yet."

    # ── Clear handler ─────────────────────────────────────────────────────

    def on_clear() -> str:
        """Clear the display only — does not clear UIState log."""
        return "Log display cleared. New entries will appear on next poll."

    # ── Download handler ──────────────────────────────────────────────────

    def on_download() -> tuple[gr.File, str]:
        """Write full log to a temp file and return for download."""
        with ui_state.lock:
            lines = list(ui_state.log_lines)

        if not lines:
            return gr.File(visible=False), "No log content to download."

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"gromacs_agent_log_{timestamp}.txt"
        tmp_path  = Path(tempfile.mkdtemp()) / filename
        tmp_path.write_text("\n".join(lines))

        return gr.File(value=str(tmp_path), visible=True), ""

    # ── Wire timer ────────────────────────────────────────────────────────

    timer.tick(
        fn = poll_log,
        inputs  = [level_filter, keyword_filter],
        outputs = [
            log_box,
            status_badge,
            line_count_md,
            elapsed_md,
            step_md,
            llm_output_box,
        ],
    )

    # ── Wire buttons ──────────────────────────────────────────────────────

    clear_btn.click(
        fn      = on_clear,
        inputs  = [],
        outputs = [log_box],
    )

    download_btn.click(
        fn      = on_download,
        inputs  = [],
        outputs = [download_file, log_box],
    )

    # ── Wire filter changes → immediate refresh ───────────────────────────

    def on_filter_change(level: str, keyword: str) -> str:
        with ui_state.lock:
            lines = list(ui_state.log_lines)
        filtered = _apply_filter(lines, level, keyword)
        return _format_log_text(filtered)

    level_filter.change(
        fn      = on_filter_change,
        inputs  = [level_filter, keyword_filter],
        outputs = [log_box],
    )

    keyword_filter.submit(
        fn      = on_filter_change,
        inputs  = [level_filter, keyword_filter],
        outputs = [log_box],
    )

    return {
        "log_box":        log_box,
        "level_filter":   level_filter,
        "keyword_filter": keyword_filter,
        "autoscroll_cb":  autoscroll_cb,
        "status_badge":   status_badge,
        "line_count_md":  line_count_md,
        "elapsed_md":     elapsed_md,
        "step_md":        step_md,
        "llm_output_box": llm_output_box,
        "clear_btn":      clear_btn,
        "download_btn":   download_btn,
        "download_file":  download_file,
        "timer":          timer,
    }