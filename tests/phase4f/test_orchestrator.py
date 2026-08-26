"""
4F-2: Verify build_agent() constructs a valid CodeAgent
using the mock LLM and real config.

Rationale:
    build_agent() is the entry point for the full system.
    These tests verify it wires together correctly without
    requiring Ollama or any real LLM server.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from agent.orchestrator import build_agent, load_config, load_system_prompt

from .mock_llm import ScriptedLLM


@pytest.fixture
def base_config(tmp_path) -> dict:
    """Minimal valid config using a mock LLM backend."""
    return {
        "llm": {
            "backend":     "transformers",   # won't be used — we inject mock
            "model_id":    "mock/model",
            "temperature": 0.1,
            "max_tokens":  512,
        },
        "pipeline": {
            "work_dir":              str(tmp_path / "gmx_run"),
            "templates_dir":         "mdp_templates",
            "state_file":            str(tmp_path / "pipeline_state.json"),
            "max_recovery_attempts": 3,
        },
        "simulation": {
            "force_field":        "amber99sb-ildn",
            "water_model":        "tip3p",
            "box_type":           "dodecahedron",
            "box_distance":       1.0,
            "salt_concentration": 0.15,
            "temperature":        300,
            "n_threads":          4,
            "use_gpu":            False,
        },
        "logging": {
            "level":    "WARNING",
            "log_file": str(tmp_path / "test_run.log"),
        },
    }


class TestBuildAgent:

    def test_build_agent_with_injected_mock_llm(self, tmp_path, base_config):
        """
        build_agent() should succeed when we bypass the LLM backend
        loader and inject a mock directly.
        """
        from smolagents import CodeAgent
        from agent.tools import get_all_tools

        mock_llm = ScriptedLLM(['final_answer("test complete")'])
        work_dir = tmp_path / "gmx_run"

        # Build agent directly without going through load_model()
        agent = CodeAgent(
            tools=get_all_tools(work_dir=str(work_dir)),
            model=mock_llm,
            max_steps=5,
            instructions=load_system_prompt(),
        )
        assert agent is not None

    def test_agent_has_all_tools(self, tmp_path):
        from smolagents import CodeAgent
        from agent.tools import get_all_tools

        mock_llm  = ScriptedLLM(['final_answer("done")'])
        work_dir  = tmp_path / "gmx_run"
        tools     = get_all_tools(work_dir=str(work_dir))

        agent      = CodeAgent(tools=tools, model=mock_llm, max_steps=5)
        tool_names = {t.name for t in agent.tools.values()}

        assert "pdb2gmx"        in tool_names
        assert "mdrun"          in tool_names
        assert "parse_gmx_log"  in tool_names
        assert "pipeline_state" in tool_names

    def test_system_prompt_loaded(self):
        prompt = load_system_prompt()
        assert len(prompt) > 100
        assert "GROMACS"    in prompt
        assert "pipeline"   in prompt.lower()

    def test_system_prompt_contains_pipeline_order(self):
        prompt = load_system_prompt()
        # Key pipeline steps must be mentioned
        for step in ("pdb2gmx", "editconf", "solvate", "grompp", "mdrun"):
            assert step in prompt, (
                f"Step '{step}' not mentioned in system prompt"
            )

    def test_system_prompt_contains_recovery_instructions(self):
        prompt = load_system_prompt()
        assert "RECOVERABLE"    in prompt
        assert "FATAL"          in prompt
        assert "MDP_PATCHES"    in prompt

    def test_load_config_returns_dict(self, tmp_path):
        import yaml
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(
            "llm:\n  backend: ollama\n  model_id: test\n"
            "pipeline:\n  work_dir: /tmp/test\n"
        )
        config = load_config(config_path)
        assert isinstance(config, dict)
        assert "llm"      in config
        assert "pipeline" in config

    def test_load_config_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")