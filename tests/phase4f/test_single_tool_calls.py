"""
4F-3: Verify the agent can execute individual tool calls correctly
when given explicit scripted LLM responses.

Rationale:
    Tests that the CodeAgent correctly:
        - Parses tool call code from LLM output
        - Passes arguments to tools
        - Receives and processes tool output strings
    Each test scripts exactly one tool call + final_answer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from smolagents import CodeAgent

from agent.tools import get_all_tools
from agent.tools.state_tools import PipelineStateTool

from .mock_llm import ScriptedLLM
from tests.phase4d.mock_helpers import GMX_TOOLS_MODULE, make_gmx_mock


def build_agent(tmp_path: Path, responses: list[str]) -> CodeAgent:
    llm   = ScriptedLLM(responses)
    tools = get_all_tools(work_dir=str(tmp_path))
    return CodeAgent(
        tools=tools,
        model=llm,
        max_steps=10,
        additional_authorized_imports=["pathlib", "json"],
    )


class TestStateToolViaAgent:
    """
    State tool calls don't need subprocess mocking —
    they are pure file I/O.
    """

    def test_agent_can_reset_state(self, tmp_path):
        agent = build_agent(tmp_path, [
            'result = pipeline_state(action="reset")\nprint(result)',
            'final_answer("State reset complete")',
        ])
        result = agent.run("Reset the pipeline state.")
        state_file = tmp_path / "pipeline_state.json"
        assert state_file.exists()

    def test_agent_can_read_state(self, tmp_path):
        agent = build_agent(tmp_path, [
            'result = pipeline_state(action="read")\nprint(result)',
            'final_answer(result)',
        ])
        result = agent.run("Read the pipeline state.")
        assert result is not None

    def test_agent_can_update_state(self, tmp_path):
        agent = build_agent(tmp_path, [
            'result = pipeline_state(action="update", '
            'updates={"input_pdb": "eiwit.pdb", "current_step": "pdb2gmx"})\n'
            'print(result)',
            'final_answer("State updated")',
        ])
        agent.run("Update the pipeline state.")

        # Verify state was written correctly
        state_file = tmp_path / "pipeline_state.json"
        assert state_file.exists()
        state = json.loads(state_file.read_text())
        assert state["input_pdb"]    == "eiwit.pdb"
        assert state["current_step"] == "pdb2gmx"

    def test_agent_can_update_files_in_state(self, tmp_path):
        agent = build_agent(tmp_path, [
            'result = pipeline_state(action="update", '
            'updates={"files": {"gro": "conf.gro", "top": "topol.top"}})\n'
            'print(result)',
            'final_answer("Files updated")',
        ])
        agent.run("Update file paths in state.")

        state_file = tmp_path / "pipeline_state.json"
        state      = json.loads(state_file.read_text())
        assert state["files"]["gro"] == "conf.gro"
        assert state["files"]["top"] == "topol.top"

    def test_agent_can_read_then_update_state(self, tmp_path):
        """Simulate the agent reading state then updating it."""
        agent = build_agent(tmp_path, [
            # Step 1: read current state
            'state_str = pipeline_state(action="read")\n'
            'import json\n'
            'state = json.loads(state_str)\n'
            'print("Current step:", state.get("current_step"))',
            # Step 2: update with new step
            'result = pipeline_state(action="update", '
            'updates={"current_step": "editconf", '
            '"completed_steps": ["pdb2gmx"]})\n'
            'print(result)',
            'final_answer("Read and update complete")',
        ])
        agent.run("Read state then update it.")

        state_file = tmp_path / "pipeline_state.json"
        state      = json.loads(state_file.read_text())
        assert state["current_step"]      == "editconf"
        assert "pdb2gmx" in state["completed_steps"]


class TestReadWriteToolsViaAgent:
    """
    ReadFileTool and WriteFileTool — pure file I/O, no mocking needed.
    """

    def test_agent_can_write_mdp_file(self, tmp_path):
        mdp_content = (
            "integrator = steep\n"
            "nsteps = 5000\n"
            "emtol = 1000.0\n"
        )
        mdp_path = str(tmp_path / "em.mdp")
        agent    = build_agent(tmp_path, [
            f'result = write_file(file_path="{mdp_path}", '
            f'content="""{mdp_content}""")\nprint(result)',
            'final_answer("MDP written")',
        ])
        agent.run("Write an MDP file.")
        assert Path(mdp_path).exists()
        assert "integrator" in Path(mdp_path).read_text()

    def test_agent_can_read_written_file(self, tmp_path):
        # Pre-create a file
        test_file = tmp_path / "test.gro"
        test_file.write_text("GROMACS GRO file content\n5000 atoms\n")

        agent = build_agent(tmp_path, [
            f'content = read_file(file_path="{test_file}")\nprint(content)',
            'final_answer(content)',
        ])
        result = agent.run("Read the GRO file.")
        assert result is not None

    def test_agent_can_write_then_read_file(self, tmp_path):
        file_path = str(tmp_path / "roundtrip.mdp")
        agent     = build_agent(tmp_path, [
            f'write_file(file_path="{file_path}", content="integrator = md\\n")',
            f'content = read_file(file_path="{file_path}")\nprint(content)',
            'final_answer(content)',
        ])
        result = agent.run("Write then read a file.")
        assert result is not None


class TestGMXToolViaAgent:
    """
    GMX tool calls with mocked subprocess.
    Tests that the agent correctly calls tools and processes output.
    """

    def test_agent_calls_editconf_and_checks_success(
        self, tmp_path, monkeypatch
    ):
        # Create input file
        input_gro = tmp_path / "conf.gro"
        input_gro.write_text("mock GRO")
        out_gro   = tmp_path / "conf_box.gro"

        monkeypatch.setattr(
            GMX_TOOLS_MODULE,
            make_gmx_mock(create_files=[out_gro]),
        )

        agent = build_agent(tmp_path, [
            f'result = editconf('
            f'input_gro="{input_gro}", '
            f'output_gro="{out_gro}", '
            f'box_type="dodecahedron", '
            f'distance=1.0)\n'
            f'print(result)',
            'final_answer(result)',
        ])
        result = agent.run("Run editconf.")
        assert result is not None

    def test_agent_processes_tool_success_output(
        self, tmp_path, monkeypatch
    ):
        """Agent should receive SUCCESS: True and be able to act on it."""
        input_gro = tmp_path / "conf.gro"
        input_gro.write_text("mock GRO")
        out_gro   = tmp_path / "conf_box.gro"

        monkeypatch.setattr(
            GMX_TOOLS_MODULE,
            make_gmx_mock(create_files=[out_gro]),
        )

        agent = build_agent(tmp_path, [
            f'result = editconf(input_gro="{input_gro}", '
            f'output_gro="{out_gro}")\n'
            f'success = "SUCCESS: True" in result\n'
            f'print("Tool succeeded:", success)',
            'final_answer("editconf succeeded")',
        ])
        result = agent.run("Run editconf and check success.")
        assert result is not None

    def test_agent_processes_tool_failure_output(
        self, tmp_path, monkeypatch
    ):
        """Agent should receive SUCCESS: False and be able to act on it."""
        from tests.phase4d.mock_helpers import make_failing_mock
        input_gro = tmp_path / "conf.gro"
        input_gro.write_text("mock GRO")

        monkeypatch.setattr(
            GMX_TOOLS_MODULE,
            make_failing_mock(stderr="Fatal error: bad box"),
        )

        agent = build_agent(tmp_path, [
            f'result = editconf(input_gro="{input_gro}")\n'
            f'failed = "SUCCESS: False" in result\n'
            f'print("Tool failed as expected:", failed)',
            'final_answer("failure detected")',
        ])
        result = agent.run("Run editconf expecting failure.")
        assert result is not None