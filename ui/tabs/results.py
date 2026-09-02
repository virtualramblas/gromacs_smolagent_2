import gradio as gr
from ui.state import UIState

def build_results_tab(ui_state: UIState) -> dict:
    gr.Markdown("## 📊 Results")
    gr.Markdown("_Energy plots and RMSD analysis will appear here after the run._")
    results_md = gr.Markdown("No results yet.")
    return {"results_md": results_md}