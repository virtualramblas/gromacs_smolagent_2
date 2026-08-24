"""
CodeAgent orchestrator — smolagents 1.26.0 compatible.

Key findings from API introspection:
- prompt_templates: PromptTemplates | None  → nested TypedDict, complex to fill
- instructions: str | None                  → the correct kwarg for system prompt
- EMPTY_PROMPT_TEMPLATES                    → shows required nested structure
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from smolagents import CodeAgent

from agent.tools import get_all_tools
from agent.backends import load_model

logger = logging.getLogger("gromacs_agent.orchestrator")


def load_config(config_path: str | Path = "config.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with path.open() as f:
        return yaml.safe_load(f)


def load_system_prompt(
    prompt_path: str | Path = "agent/prompts/system_prompt.md",
) -> str:
    path = Path(prompt_path)
    if not path.exists():
        raise FileNotFoundError(f"System prompt not found: {prompt_path}")
    return path.read_text()


def build_agent(config: dict) -> tuple[CodeAgent, bool]:
    """
    Instantiate CodeAgent for smolagents 1.26.0.

    System prompt injection:
        Use the `instructions` kwarg on MultiStepAgent.__init__.
        This is the dedicated parameter for custom agent instructions
        in 1.26.0 — separate from the complex nested prompt_templates.

    Returns:
        (agent, needs_prompt_prepend)
        needs_prompt_prepend is always False with this approach.
    """
    llm_cfg  = config["llm"]
    pipe_cfg = config["pipeline"]
    work_dir = pipe_cfg["work_dir"]

    model = load_model(llm_cfg)
    logger.info("LLM: %s / %s", llm_cfg["backend"], llm_cfg["model_id"])

    tools = get_all_tools(work_dir=work_dir)
    logger.info("Tools loaded: %s", [t.name for t in tools])

    system_prompt = load_system_prompt()

    agent = CodeAgent(
        tools=tools,
        model=model,
        instructions=system_prompt,     # ← correct kwarg in 1.26.0
        max_steps=60,
        planning_interval=5,
        additional_authorized_imports=[
            "pathlib", "json", "re", "statistics",
            "agent.utils.mdp_utils",
            "agent.recovery",
        ],
    )

    logger.info("CodeAgent built successfully.")
    return agent, False                 # needs_prepend always False now