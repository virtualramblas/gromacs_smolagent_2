import gradio as gr
from ui.state import UIState

def build_progress_tab(ui_state: UIState) -> dict:
    gr.Markdown("## 📋 Pipeline Progress")
    gr.Markdown("_Step-by-step status will appear here during a run._")
    progress_md = gr.Markdown("Waiting for run to start...")
    return {"progress_md": progress_md}