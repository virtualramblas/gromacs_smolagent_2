"""
CodeAgent orchestrator — compatible with smolagents 1.26.0
"""

from __future__ import annotations

import inspect
import logging
from copy import deepcopy
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


def _get_prompt_templates(system_prompt: str) -> dict:
    """
    Build a complete prompt_templates dict for smolagents 1.26.0.

    Strategy:
        1. Deep-copy the default templates from CodeAgent class attribute
        2. Replace only the 'system_prompt' key with our custom prompt
        3. Return the complete dict — all required keys present

    This avoids the AssertionError about missing template keys while
    still injecting our domain-specific system prompt.
    """
    # CodeAgent.prompt_templates is a class-level dict of all defaults
    if not hasattr(CodeAgent, "prompt_templates"):
        logger.warning(
            "CodeAgent has no 'prompt_templates' class attribute. "
            "Returning minimal template dict."
        )
        return {"system_prompt": system_prompt}

    # Deep copy so we never mutate the class-level default
    templates = deepcopy(CodeAgent.prompt_templates)

    logger.info(
        "Default prompt_templates keys found: %s", list(templates.keys())
    )

    # Replace only the system_prompt slot
    if "system_prompt" in templates:
        templates["system_prompt"] = system_prompt
        logger.info("Replaced 'system_prompt' in prompt_templates.")
    else:
        # Key name may differ in some versions — log all keys for diagnosis
        logger.warning(
            "Key 'system_prompt' not found in default templates. "
            "Available keys: %s. Inserting anyway.",
            list(templates.keys()),
        )
        templates["system_prompt"] = system_prompt

    return templates


def _inspect_code_agent() -> dict:
    """Introspect CodeAgent.__init__ parameters at runtime."""
    init_params = set(inspect.signature(CodeAgent.__init__).parameters.keys())
    return {
        "init_params":           init_params,
        "has_system_prompt":     "system_prompt"     in init_params,
        "has_prompt_templates":  "prompt_templates"  in init_params,
        "has_planning_interval": "planning_interval" in init_params,
    }


def build_agent(config: dict) -> tuple[CodeAgent, bool]:
    """
    Instantiate CodeAgent for smolagents 1.26.0.

    Returns:
        (agent, needs_prompt_prepend)
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
    logger.info("CodeAgent init params: %s", findings["init_params"])

    # ------------------------------------------------------------------
    # Base constructor kwargs
    # ------------------------------------------------------------------
    agent_kwargs: dict = {
        "tools":   tools,
        "model":   model,
        "max_steps": 60,
        "additional_authorized_imports": [
            "pathlib", "json", "re", "statistics",
            "agent.utils.mdp_utils",
            "agent.recovery",
        ],
    }

    if findings["has_planning_interval"]:
        agent_kwargs["planning_interval"] = 5

    # ------------------------------------------------------------------
    # System prompt injection strategy
    # ------------------------------------------------------------------
    needs_prepend = False

    if findings["has_system_prompt"]:
        # Older API — direct kwarg
        agent_kwargs["system_prompt"] = system_prompt
        logger.info("Strategy: system_prompt constructor kwarg.")

    elif findings["has_prompt_templates"]:
        # 1.26.0 API — must pass complete templates dict
        agent_kwargs["prompt_templates"] = _get_prompt_templates(system_prompt)
        logger.info("Strategy: prompt_templates constructor kwarg (complete dict).")

    else:
        # No constructor-level injection — will try post-construction
        logger.info("Strategy: post-construction attribute injection.")

    # ------------------------------------------------------------------
    # Construct agent
    # ------------------------------------------------------------------
    agent = CodeAgent(**agent_kwargs)

    # ------------------------------------------------------------------
    # Post-construction injection (fallback for unknown API shapes)
    # ------------------------------------------------------------------
    if not findings["has_system_prompt"] and not findings["has_prompt_templates"]:
        injected = False

        for attr in ("system_prompt", "prompt_template", "prompt_templates"):
            if hasattr(agent, attr):
                current = getattr(agent, attr)
                if isinstance(current, dict):
                    current["system_prompt"] = system_prompt
                else:
                    setattr(agent, attr, system_prompt)
                logger.info(
                    "Post-construction injection via agent.%s", attr
                )
                injected = True
                break

        if not injected:
            prompt_attrs = [a for a in dir(agent) if "prompt" in a.lower()]
            logger.warning(
                "No injection point found. Prompt-related attrs: %s. "
                "Falling back to task-level prepend.",
                prompt_attrs,
            )
            needs_prepend = True

    logger.info("CodeAgent built. needs_prompt_prepend=%s", needs_prepend)
    return agent, needs_prepend