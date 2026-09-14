"""
ui/log_handler.py

A Python logging.Handler that writes log records to UIState.
Install it once in the agent thread so all agent logging
automatically appears in the Gradio live log panel.

Usage:
    from ui.log_handler import UILogHandler
    handler = UILogHandler(ui_state)
    logging.getLogger("gromacs_agent").addHandler(handler)
"""

from __future__ import annotations

import logging

from ui.state import UIState


class UILogHandler(logging.Handler):
    """
    Logging handler that appends formatted records to UIState.log_lines.
    Thread-safe — UIState.append_log() acquires the lock internally.
    """

    # Colour prefixes for different log levels
    _LEVEL_PREFIX = {
        logging.DEBUG:    "DEBUG   ",
        logging.INFO:     "INFO    ",
        logging.WARNING:  "WARNING ",
        logging.ERROR:    "ERROR   ",
        logging.CRITICAL: "CRITICAL",
    }

    def __init__(self, ui_state: UIState, level: int = logging.DEBUG):
        super().__init__(level)
        self.ui_state = ui_state
        self.setFormatter(logging.Formatter(
            "%(name)s — %(message)s"
        ))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            prefix = self._LEVEL_PREFIX.get(record.levelno, "LOG     ")
            msg    = self.format(record)
            self.ui_state.append_log(f"{prefix} | {msg}")
        except Exception:
            self.handleError(record)


def install_ui_log_handler(
    ui_state:    UIState,
    logger_name: str = "gromacs_agent",
    level:       int = logging.INFO,
) -> UILogHandler:
    """
    Attach a UILogHandler to the named logger.
    Safe to call multiple times — checks for existing handler first.

    Returns the installed handler.
    """
    target_logger = logging.getLogger(logger_name)

    # Remove any existing UILogHandler to avoid duplicates on re-run
    target_logger.handlers = [
        h for h in target_logger.handlers
        if not isinstance(h, UILogHandler)
    ]

    handler = UILogHandler(ui_state, level=level)
    target_logger.addHandler(handler)
    target_logger.setLevel(level)

    return handler