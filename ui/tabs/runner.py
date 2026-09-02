from ui.state import UIState

def start_run(ui_state: UIState, resume: bool = False) -> None:
    """Stub — wired to the agent in UI-7."""
    ui_state.append_log(
        f"Run {'resume' if resume else 'start'} requested — "
        "runner not yet connected (UI-7)."
    )