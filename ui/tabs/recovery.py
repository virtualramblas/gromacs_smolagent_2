import gradio as gr
from ui.state import UIState

def build_recovery_tab(ui_state: UIState) -> dict:
    gr.Markdown("## 🔧 Recovery Events")
    gr.Markdown("_Recovery actions triggered during the run will appear here._")
    recovery_md = gr.Markdown("No recovery events yet.")
    return {"recovery_md": recovery_md}