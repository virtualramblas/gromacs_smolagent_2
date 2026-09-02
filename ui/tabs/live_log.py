import gradio as gr
from ui.state import UIState

def build_live_log_tab(ui_state: UIState) -> dict:
    gr.Markdown("## 📜 Live Agent Log")
    gr.Markdown("_Log streaming will appear here during a run._")
    log_box = gr.Textbox(
        label       = "Agent Log",
        lines       = 30,
        max_lines   = 30,
        interactive = False,
        elem_classes = ["log-panel"],
        value       = "Waiting for run to start...",
    )
    return {"log_box": log_box}