"""
Builds a CodeAgent configured for integration testing.
Uses real tools with mocked subprocess calls.
"""

from __future__ import annotations

from pathlib import Path

from smolagents import CodeAgent

from agent.tools import get_all_tools


def build_test_agent(
    llm,
    work_dir: Path,
    max_steps: int = 30,
) -> CodeAgent:
    """
    Build a CodeAgent with real tools but a mock LLM.

    Args:
        llm:      Any callable that accepts (messages) and returns
                  an object with a .content attribute.
        work_dir: Temporary working directory for this test.
        max_steps: Maximum agent steps before forced termination.

    Returns:
        Configured CodeAgent ready for agent.run(task).
    """
    tools = get_all_tools(work_dir=str(work_dir))

    agent = CodeAgent(
        tools=tools,
        model=llm,
        max_steps=max_steps,
        additional_authorized_imports=[
            "pathlib", "json", "re", "statistics",
            "agent.utils.mdp_utils",
            "agent.recovery",
        ],
    )
    return agent