"""
CodeAgent orchestrator — compatible with smolagents 1.26.0
"""

from __future__ import annotations

import inspect
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


def _inspect_code_agent() -> dict:
    """
    Introspect CodeAgent at runtime.
    Returns a dict of findings used to adapt instantiation.
    """
    init_params = set(inspect.signature(CodeAgent.__init__).parameters.keys())

    # Check for prompt-related attributes on the class itself
    class_attrs = set(dir(CodeAgent))

    findings = {
        "init_params":          init_params,
        "has_system_prompt":    "system_prompt"    in init_params,
        "has_prompt_templates": "prompt_templates" in init_params,
        "has_planning_interval":"planning_interval" in init_params,
        "has_prompt_attr":      "prompt_template"  in class_attrs,
        "has_system_prompt_attr": "system_prompt"  in class_attrs,
    }

    logger.info("CodeAgent introspection: %s", findings)
    return findings


def _inject_system_prompt(
    agent: CodeAgent,
    system_prompt: str,
    findings: dict,
) -> None:
    """
    Inject the system prompt into an already-constructed CodeAgent.
    Tries all known injection points for smolagents 1.x compatibility.
    """

    # Strategy 1: Direct attribute assignment (1.26.0 pattern)
    # In 1.26.0, CodeAgent stores the system prompt as an attribute
    # that is read at agent.run() time
    if hasattr(agent, "system_prompt"):
        agent.system_prompt = system_prompt
        logger.info("System prompt injected via agent.system_prompt attribute.")
        return

    # Strategy 2: prompt_templates dict attribute
    if hasattr(agent, "prompt_templates"):
        if isinstance(agent.prompt_templates, dict):
            agent.prompt_templates["system_prompt"] = system_prompt
        else:
            agent.prompt_templates = {"system_prompt": system_prompt}
        logger.info("System prompt injected via agent.prompt_templates attribute.")
        return

    # Strategy 3: prompt_template (singular) attribute
    if hasattr(agent, "prompt_template"):
        agent.prompt_template = system_prompt
        logger.info("System prompt injected via agent.prompt_template attribute.")
        return

    # Strategy 4: No known injection point found — log warning
    # Task-level prepend will be used as fallback in run.py
    logger.warning(
        "No known system prompt injection point found on CodeAgent instance. "
        "System prompt will be prepended to the task string at agent.run() time. "
        "Agent attributes available: %s",
        [a for a in dir(agent) if "prompt" in a.lower()],
    )


def build_agent(config: dict) -> tuple[CodeAgent, bool]:
    """
    Instantiate the CodeAgent for smolagents 1.26.0.

    Returns:
        (agent, needs_prompt_prepend)
        needs_prompt_prepend: True if system prompt must be prepended
        to the task string in run.py (last-resort fallback).
    """
    llm_cfg  = config["llm"]
    pipe_cfg = config["pipeline"]
    work_dir = pipe_cfg["work_dir"]

    model = load_model(llm_cfg)
    logger.info("LLM: %s / %s", llm_cfg["backend"], llm_cfg["model_id"])

    tools = get_all_tools(work_dir=work_dir)
    logger.info("Tools loaded: %s", [t.name for t in tools])

    system_prompt = load_system_prompt()
    findings      = _inspect_code_agent()

    # ------------------------------------------------------------------
    # Build constructor kwargs — only include what this version supports
    # ------------------------------------------------------------------
    agent_kwargs: dict = {
        "tools": tools,
        "model": model,
        "max_steps": 60,
        "additional_authorized_imports": [
            "pathlib", "json", "re", "statistics",
            "agent.utils.mdp_utils",
            "agent.recovery",
        ],
    }

    # planning_interval — only if supported
    if findings["has_planning_interval"]:
        agent_kwargs["planning_interval"] = 5

    # system_prompt as constructor kwarg — only if supported
    # (older API, kept for backwards compat)
    if findings["has_system_prompt"]:
        agent_kwargs["system_prompt"] = system_prompt
        logger.info("Passing system_prompt as constructor kwarg.")

    elif findings["has_prompt_templates"]:
        agent_kwargs["prompt_templates"] = {"system_prompt": system_prompt}
        logger.info("Passing system_prompt via prompt_templates constructor kwarg.")

    # ------------------------------------------------------------------
    # Construct agent
    # ------------------------------------------------------------------
    agent = CodeAgent(**agent_kwargs)

    # ------------------------------------------------------------------
    # Post-construction prompt injection
    # (handles 1.26.0 where prompt is set as instance attribute)
    # ------------------------------------------------------------------
    needs_prepend = False

    if not findings["has_system_prompt"] and not findings["has_prompt_templates"]:
        _inject_system_prompt(agent, system_prompt, findings)

        # Check if injection succeeded
        injected = (
            (hasattr(agent, "system_prompt")    and agent.system_prompt    == system_prompt) or
            (hasattr(agent, "prompt_template")  and agent.prompt_template  == system_prompt) or
            (hasattr(agent, "prompt_templates") and isinstance(agent.prompt_templates, dict)
             and agent.prompt_templates.get("system_prompt") == system_prompt)
        )
        if not injected:
            logger.warning(
                "System prompt injection could not be verified. "
                "Falling back to task-level prepend."
            )
            needs_prepend = True

    logger.info("CodeAgent built. needs_prompt_prepend=%s", needs_prepend)
    return agent, needs_prepend