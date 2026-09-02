"""
UI-2: Run Configuration Tab

Allows the user to:
    - Upload a PDB file or enter a path
    - Configure simulation parameters
    - Configure LLM backend
    - Launch, stop, or resume a run
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import gradio as gr

from ui.state import UIState


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FORCE_FIELDS = [
    "amber99sb-ildn",
    "amber03",
    "amber96",
    "charmm36m",
    "charmm27",
    "gromos96 43a1",
    "gromos96 53a6",
    "oplsaa",
]

WATER_MODELS = [
    "tip3p",
    "tip4p",
    "tip5p",
    "spc",
    "spce",
]

BOX_TYPES = [
    "dodecahedron",
    "cubic",
    "triclinic",
    "octahedron",
]

LLM_BACKENDS = [
    "ollama",
    "litellm",
    "transformers",
]


# ---------------------------------------------------------------------------
# Tab builder
# ---------------------------------------------------------------------------

def build_run_config_tab(ui_state: UIState) -> dict:
    """Build the Run Configuration tab. Returns component references."""

    # ── Section: Input ────────────────────────────────────────────────────
    gr.Markdown("## 📂 Input Structure")

    with gr.Row():
        with gr.Column(scale=2):
            pdb_upload = gr.File(
                label       = "Upload PDB File",
                file_types  = [".pdb"],
                type        = "filepath",
            )
        with gr.Column(scale=3):
            pdb_path_box = gr.Textbox(
                label       = "Or enter PDB file path",
                placeholder = "/path/to/protein.pdb",
                info        = "Absolute path to an existing PDB file on disk.",
            )

    with gr.Row():
        work_dir_box = gr.Textbox(
            label       = "Working Directory",
            placeholder = "/tmp/gmx_run",
            value       = str(Path.home() / "gmx_agent_run"),
            info        = (
                "Directory where all simulation files will be written. "
                "Created automatically if it does not exist."
            ),
        )

    # ── Section: Simulation Parameters ───────────────────────────────────
    gr.Markdown("## ⚗️  Simulation Parameters")

    with gr.Row():
        with gr.Column():
            force_field_dd = gr.Dropdown(
                label   = "Force Field",
                choices = FORCE_FIELDS,
                value   = "amber99sb-ildn",
            )
            water_model_dd = gr.Dropdown(
                label   = "Water Model",
                choices = WATER_MODELS,
                value   = "tip3p",
            )
        with gr.Column():
            box_type_dd = gr.Dropdown(
                label   = "Box Type",
                choices = BOX_TYPES,
                value   = "dodecahedron",
            )
            box_distance_sl = gr.Slider(
                label   = "Box Distance (nm)",
                minimum = 0.5,
                maximum = 2.0,
                step    = 0.1,
                value   = 1.0,
                info    = "Minimum distance between protein and box edge.",
            )

    with gr.Row():
        with gr.Column():
            n_threads_sl = gr.Slider(
                label   = "CPU Threads",
                minimum = 1,
                maximum = 32,
                step    = 1,
                value   = 4,
            )
        with gr.Column():
            use_gpu_cb = gr.Checkbox(
                label = "Use GPU (if available)",
                value = False,
            )
        with gr.Column():
            max_steps_sl = gr.Slider(
                label   = "Max Agent Steps",
                minimum = 10,
                maximum = 120,
                step    = 5,
                value   = 60,
                info    = "Maximum number of LLM reasoning steps.",
            )

    # ── Section: LLM Configuration ───────────────────────────────────────
    gr.Markdown("## 🤖 LLM Configuration")

    with gr.Row():
        with gr.Column():
            llm_backend_dd = gr.Dropdown(
                label   = "Backend",
                choices = LLM_BACKENDS,
                value   = "ollama",
            )
        with gr.Column():
            llm_model_box = gr.Textbox(
                label       = "Model ID",
                value       = "qwen2.5:14b",
                placeholder = "e.g. qwen2.5:14b, llama3.1:8b",
                info        = "Model name as recognised by the selected backend.",
            )

    with gr.Row():
        with gr.Column():
            temperature_sl = gr.Slider(
                label   = "Temperature",
                minimum = 0.0,
                maximum = 1.0,
                step    = 0.05,
                value   = 0.1,
                info    = "Lower = more deterministic. Recommended: 0.0–0.2.",
            )
        with gr.Column():
            llm_url_box = gr.Textbox(
                label       = "API Base URL (optional)",
                placeholder = "http://localhost:11434",
                info        = "Leave blank to use the backend default.",
            )

    # ── Section: Run Control ─────────────────────────────────────────────
    gr.Markdown("## 🚀 Run Control")

    with gr.Row():
        run_btn    = gr.Button(
            "▶  Start Run",
            variant  = "primary",
            elem_classes = ["btn-run"],
            scale    = 2,
        )
        resume_btn = gr.Button(
            "⏩  Resume",
            variant  = "secondary",
            elem_classes = ["btn-resume"],
            scale    = 1,
        )
        stop_btn   = gr.Button(
            "⏹  Stop",
            variant  = "stop",
            elem_classes = ["btn-stop"],
            scale    = 1,
        )

    # ── Status bar ────────────────────────────────────────────────────────
    status_md = gr.Markdown(
        value = "_Status: idle — configure a run and click Start._",
        elem_id = "run-status-bar",
    )

    # ── Validation output ─────────────────────────────────────────────────
    validation_box = gr.Textbox(
        label     = "Validation",
        lines     = 3,
        max_lines = 3,
        interactive = False,
        visible   = False,
    )

    # ── Event handlers ────────────────────────────────────────────────────

    def on_pdb_upload(file_path: str) -> str:
        """When a file is uploaded, copy it to a stable location."""
        if not file_path:
            return ""
        return file_path

    def validate_config(
        pdb_upload_path: str,
        pdb_path:        str,
        work_dir:        str,
        llm_model:       str,
    ) -> tuple[bool, str]:
        """Validate configuration before starting a run."""
        errors = []

        # Resolve PDB path
        resolved_pdb = pdb_upload_path or pdb_path
        if not resolved_pdb:
            errors.append("❌ No PDB file specified.")
        elif not Path(resolved_pdb).exists():
            errors.append(f"❌ PDB file not found: {resolved_pdb}")

        if not work_dir.strip():
            errors.append("❌ Working directory is required.")

        if not llm_model.strip():
            errors.append("❌ LLM model ID is required.")

        if errors:
            return False, "\n".join(errors)
        return True, "✅ Configuration valid."

    def on_run_click(
        pdb_upload_path: str,
        pdb_path:        str,
        work_dir:        str,
        force_field:     str,
        water_model:     str,
        box_type:        str,
        box_distance:    float,
        n_threads:       int,
        use_gpu:         bool,
        max_steps:       int,
        llm_backend:     str,
        llm_model:       str,
        temperature:     float,
        llm_url:         str,
    ) -> tuple[str, str, bool]:
        """
        Validate config, update UIState, and signal the run to start.
        Returns: (status_markdown, validation_text, validation_visible)
        """
        resolved_pdb = pdb_upload_path or pdb_path
        valid, msg   = validate_config(
            pdb_upload_path, pdb_path, work_dir, llm_model
        )

        if not valid:
            return (
                "_Status: ⚠️ Configuration errors — see Validation._",
                msg,
                True,
            )

        # Update shared state
        with ui_state.lock:
            ui_state.pdb_path     = resolved_pdb
            ui_state.work_dir     = work_dir.strip()
            ui_state.force_field  = force_field
            ui_state.water_model  = water_model
            ui_state.box_type     = box_type
            ui_state.box_distance = box_distance
            ui_state.n_threads    = int(n_threads)
            ui_state.use_gpu      = use_gpu
            ui_state.max_steps    = int(max_steps)
            ui_state.llm_backend  = llm_backend
            ui_state.llm_model    = llm_model
            ui_state.temperature  = temperature

        # Signal the runner (UI-7 will wire this to the actual agent)
        from ui.runner import start_run
        start_run(ui_state, resume=False)

        return (
            f"_Status: 🟡 Running — {Path(resolved_pdb).name} "
            f"with {llm_model}_",
            msg,
            False,
        )

    def on_resume_click(
        pdb_upload_path: str,
        pdb_path:        str,
        work_dir:        str,
        force_field:     str,
        water_model:     str,
        box_type:        str,
        box_distance:    float,
        n_threads:       int,
        use_gpu:         bool,
        max_steps:       int,
        llm_backend:     str,
        llm_model:       str,
        temperature:     float,
        llm_url:         str,
    ) -> tuple[str, str, bool]:
        resolved_pdb = pdb_upload_path or pdb_path
        valid, msg   = validate_config(
            pdb_upload_path, pdb_path, work_dir, llm_model
        )
        if not valid:
            return (
                "_Status: ⚠️ Configuration errors — see Validation._",
                msg,
                True,
            )

        with ui_state.lock:
            ui_state.pdb_path     = resolved_pdb
            ui_state.work_dir     = work_dir.strip()
            ui_state.force_field  = force_field
            ui_state.water_model  = water_model
            ui_state.box_type     = box_type
            ui_state.box_distance = box_distance
            ui_state.n_threads    = int(n_threads)
            ui_state.use_gpu      = use_gpu
            ui_state.max_steps    = int(max_steps)
            ui_state.llm_backend  = llm_backend
            ui_state.llm_model    = llm_model
            ui_state.temperature  = temperature

        from ui.runner import start_run
        start_run(ui_state, resume=True)

        return (
            f"_Status: 🟡 Resuming — {Path(resolved_pdb).name}_",
            msg,
            False,
        )

    def on_stop_click() -> str:
        with ui_state.lock:
            ui_state.stop_requested = True
        return "_Status: 🔴 Stop requested — waiting for current step to finish..._"

    # Wire upload → path box
    pdb_upload.change(
        fn      = on_pdb_upload,
        inputs  = [pdb_upload],
        outputs = [pdb_path_box],
    )

    # Wire run button
    run_btn.click(
        fn = on_run_click,
        inputs = [
            pdb_upload, pdb_path_box, work_dir_box,
            force_field_dd, water_model_dd, box_type_dd, box_distance_sl,
            n_threads_sl, use_gpu_cb, max_steps_sl,
            llm_backend_dd, llm_model_box, temperature_sl, llm_url_box,
        ],
        outputs = [status_md, validation_box, validation_box],
    )

    # Wire resume button
    resume_btn.click(
        fn = on_resume_click,
        inputs = [
            pdb_upload, pdb_path_box, work_dir_box,
            force_field_dd, water_model_dd, box_type_dd, box_distance_sl,
            n_threads_sl, use_gpu_cb, max_steps_sl,
            llm_backend_dd, llm_model_box, temperature_sl, llm_url_box,
        ],
        outputs = [status_md, validation_box, validation_box],
    )

    # Wire stop button
    stop_btn.click(
        fn      = on_stop_click,
        inputs  = [],
        outputs = [status_md],
    )

    return {
        "pdb_upload":      pdb_upload,
        "pdb_path_box":    pdb_path_box,
        "work_dir_box":    work_dir_box,
        "force_field_dd":  force_field_dd,
        "water_model_dd":  water_model_dd,
        "box_type_dd":     box_type_dd,
        "box_distance_sl": box_distance_sl,
        "n_threads_sl":    n_threads_sl,
        "use_gpu_cb":      use_gpu_cb,
        "max_steps_sl":    max_steps_sl,
        "llm_backend_dd":  llm_backend_dd,
        "llm_model_box":   llm_model_box,
        "temperature_sl":  temperature_sl,
        "llm_url_box":     llm_url_box,
        "run_btn":         run_btn,
        "resume_btn":      resume_btn,
        "stop_btn":        stop_btn,
        "status_md":       status_md,
        "validation_box":  validation_box,
    }