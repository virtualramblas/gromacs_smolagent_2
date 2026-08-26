"""
4F-4: Verify the agent correctly sequences multiple tool calls
and maintains state across steps.

Rationale:
    The core value of the agentic approach is correct multi-step
    orchestration. These tests verify:
        - State is updated after each step
        - File paths flow correctly between steps
        - The agent can complete a partial pipeline sequence
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from smolagents import CodeAgent

from agent.tools import get_all_tools
from tests.phase4d.mock_helpers import GMX_TOOLS_MODULE, make_gmx_mock

from .mock_llm import ScriptedLLM


def build_agent(tmp_path: Path, responses: list[str]) -> CodeAgent:
    llm   = ScriptedLLM(responses)
    tools = get_all_tools(work_dir=str(tmp_path))
    return CodeAgent(
        tools=tools,
        model=llm,
        max_steps=20,
        additional_authorized_imports=["pathlib", "json"],
    )


class TestPipelineSequence:

    def test_state_initialisation_sequence(self, tmp_path):
        """Agent correctly resets and initialises state."""
        agent = build_agent(tmp_path, [
            # Reset state
            'pipeline_state(action="reset")',
            # Set initial values
            'pipeline_state(action="update", updates={'
            '"input_pdb": "eiwit.pdb", '
            '"work_dir": "' + str(tmp_path) + '", '
            '"current_step": "pdb2gmx"})',
            'final_answer("Initialised")',
        ])
        agent.run("Initialise the pipeline.")

        state_file = tmp_path / "pipeline_state.json"
        state      = json.loads(state_file.read_text())
        assert state["input_pdb"]    == "eiwit.pdb"
        assert state["current_step"] == "pdb2gmx"

    def test_two_step_sequence_with_state_updates(self, tmp_path, monkeypatch):
        """
        Simulate pdb2gmx → editconf with state updates after each step.
        """
        pdb_file = tmp_path / "eiwit.pdb"
        pdb_file.write_text("ATOM mock")
        conf_gro     = tmp_path / "conf.gro"
        topol_top    = tmp_path / "topol.top"
        conf_box_gro = tmp_path / "conf_box.gro"

        # Mock both GMX calls to succeed
        call_count = {"n": 0}
        def _mock_gmx(args, work_dir, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # pdb2gmx outputs
                conf_gro.write_text("mock GRO")
                topol_top.write_text("mock TOP")
            else:
                # editconf output
                conf_box_gro.write_text("mock BOX GRO")
            return 0, "", ""

        monkeypatch.setattr(GMX_TOOLS_MODULE, _mock_gmx)

        agent = build_agent(tmp_path, [
            # Step 1: pdb2gmx
            f'r1 = pdb2gmx(pdb_file="{pdb_file}", '
            f'force_field="amber99sb-ildn", water_model="tip3p")\n'
            f'print(r1)',
            # Update state after pdb2gmx
            f'pipeline_state(action="update", updates={{'
            f'"completed_steps": ["pdb2gmx"], '
            f'"files": {{"gro": "{conf_gro}", "top": "{topol_top}"}}, '
            f'"current_step": "editconf"}})',
            # Step 2: editconf
            f'r2 = editconf(input_gro="{conf_gro}", '
            f'output_gro="{conf_box_gro}", '
            f'box_type="dodecahedron", distance=1.0)\n'
            f'print(r2)',
            # Update state after editconf
            f'pipeline_state(action="update", updates={{'
            f'"completed_steps": ["pdb2gmx", "editconf"], '
            f'"files": {{"gro_box": "{conf_box_gro}"}}, '
            f'"current_step": "solvate"}})',
            'final_answer("Two steps complete")',
        ])
        agent.run("Run pdb2gmx then editconf.")

        state_file = tmp_path / "pipeline_state.json"
        state      = json.loads(state_file.read_text())

        assert "pdb2gmx"  in state["completed_steps"]
        assert "editconf" in state["completed_steps"]
        assert state["current_step"]      == "solvate"
        assert state["files"]["gro"]      == str(conf_gro)
        assert state["files"]["gro_box"]  == str(conf_box_gro)

    def test_mdp_write_then_grompp_sequence(self, tmp_path, monkeypatch):
        """
        Simulate writing an MDP file then running grompp.
        Tests the write_file → grompp tool chain.
        """
        conf_gro  = tmp_path / "conf.gro"
        topol_top = tmp_path / "topol.top"
        em_mdp    = tmp_path / "em.mdp"
        em_tpr    = tmp_path / "em.tpr"

        conf_gro.write_text("mock GRO")
        topol_top.write_text("mock TOP")

        monkeypatch.setattr(
            GMX_TOOLS_MODULE,
            make_gmx_mock(create_files=[em_tpr]),
        )

        agent = build_agent(tmp_path, [
            # Write MDP file
            f'write_file(file_path="{em_mdp}", '
            f'content="integrator = steep\\nnsteps = 5000\\n")',
            # Run grompp
            f'r = grompp(mdp_file="{em_mdp}", '
            f'input_gro="{conf_gro}", '
            f'topology_top="{topol_top}", '
            f'output_tpr="{em_tpr}")\n'
            f'print(r)',
            # Update state
            f'pipeline_state(action="update", updates={{'
            f'"files": {{"tpr_em": "{em_tpr}"}}, '
            f'"current_step": "mdrun_em"}})',
            'final_answer("grompp complete")',
        ])
        agent.run("Write MDP and run grompp.")

        assert em_mdp.exists()
        state_file = tmp_path / "pipeline_state.json"
        state      = json.loads(state_file.read_text())
        assert state["files"]["tpr_em"]  == str(em_tpr)
        assert state["current_step"]     == "mdrun_em"

    def test_parse_log_integrated_in_sequence(self, tmp_path):
        """
        Agent calls parse_gmx_log after mdrun and reads the diagnosis.
        """
        # Create a realistic EM log
        em_log = tmp_path / "em.log"
        em_log.write_text(
            "integrator              = steep\n"
            "nsteps                  = 5000\n"
            "Steepest Descents converged to Fmax < 1000 in 500 steps\n"
            "Potential Energy  = -4.56789e+05\n"
            "Maximum force     =  9.87654e+02\n"
        )

        agent = build_agent(tmp_path, [
            f'diagnosis = parse_gmx_log(log_file="{em_log}")\n'
            f'print(diagnosis)',
            # Agent reads diagnosis and updates state
            f'converged = "SUCCESS_CONVERGED" in diagnosis\n'
            f'pipeline_state(action="update", updates={{'
            f'"em_converged": converged, '
            f'"completed_steps": ["mdrun_em"]}})',
            'final_answer("EM complete")',
        ])
        agent.run("Parse EM log and update state.")

        state_file = tmp_path / "pipeline_state.json"
        state      = json.loads(state_file.read_text())
        assert state["em_converged"]         is True
        assert "mdrun_em" in state["completed_steps"]

    def test_failed_em_triggers_state_update(self, tmp_path):
        """
        Agent detects EM failure from parse_gmx_log and records it.
        """
        em_log = tmp_path / "em.log"
        em_log.write_text(
            "integrator              = steep\n"
            "nsteps                  = 1000\n"
            "Steepest Descents did not converge to Fmax < 1000 in 1000 steps\n"
            "Potential Energy  = -2.34567e+05\n"
            "Maximum force     =  1.52340e+04\n"
        )

        agent = build_agent(tmp_path, [
            f'diagnosis = parse_gmx_log(log_file="{em_log}")\n'
            f'print(diagnosis)',
            f'failed = "RECOVERABLE" in diagnosis\n'
            f'pipeline_state(action="update", updates={{'
            f'"em_converged": False, '
            f'"warnings": ["EM did not converge — recovery needed"]}})',
            'final_answer("EM failed, recovery needed")',
        ])
        agent.run("Parse failed EM log.")

        state_file = tmp_path / "pipeline_state.json"
        state      = json.loads(state_file.read_text())
        assert state["em_converged"] is False
        assert len(state["warnings"]) >= 1