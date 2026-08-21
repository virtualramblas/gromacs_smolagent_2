"""
CodeAgent orchestrator — wires together LLM backend, tools,
system prompt, and configuration into a runnable agent.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import yaml
from smolagents import CodeAgent

from agent.tools import get_all_tools
from agent.backends import load_model

logger = logging.getLogger("gromacs_agent.orchestrator")


def load_config(config_path: str | Path = "config.yaml") -> dict:
    """Load and return the YAML configuration."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with path.open() as f:
        return yaml.safe_load(f)


def load_system_prompt(
    prompt_path: str | Path = "agent/prompts/system_prompt.md",
) -> str:
    """Load the system prompt from file."""
    path = Path(prompt_path)
    if not path.exists():
        raise FileNotFoundError(f"System prompt not found: {prompt_path}")
    return path.read_text()


def build_agent(config: dict) -> CodeAgent:
    """
    Instantiate the CodeAgent with the configured LLM backend and tools.

    Args:
        config: Parsed config.yaml dict.

    Returns:
        A fully configured SmolAgent CodeAgent.
    """
    llm_cfg  = config["llm"]
    pipe_cfg = config["pipeline"]
    work_dir = config["pipeline"]["work_dir"]

    # Resolve LLM model
    model = load_model(llm_cfg)
    logger.info(
        "LLM backend: %s | model: %s",
        llm_cfg["backend"], llm_cfg["model_id"]
    )

    # Load tools
    tools = get_all_tools(work_dir=work_dir)
    logger.info("Loaded %d tools: %s", len(tools), [t.name for t in tools])

    # Load system prompt
    system_prompt = load_system_prompt()

    # Build agent
    agent = CodeAgent(
        tools=tools,
        model=model,
        system_prompt=system_prompt,
        max_steps=60,               # Full pipeline needs ~20 steps + recovery headroom
        planning_interval=5,        # Re-plan every 5 steps
        additional_authorized_imports=[
            "pathlib", "json", "re", "statistics",
            "agent.utils.mdp_utils",
            "agent.recovery",
        ],
    )

    logger.info("CodeAgent built successfully.")
    return agent