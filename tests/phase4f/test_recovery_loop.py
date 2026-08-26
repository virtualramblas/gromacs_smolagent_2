"""
4F-5: Verify the agent correctly executes the recovery loop —
reading MDP patches from parse_gmx_log output and applying them.

Rationale:
    The recovery loop is the highest-value agentic behaviour.
    These tests verify the agent can:
        - Read RECOVERABLE diagnosis
        - Extract MDP_PATCHES from diagnosis string
        - Apply patches via write_file
        - Re-run the failed step
        - Update state with recovery attempt count
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from smolagents import CodeAgent

from agent.tools import get_all_tools
from agent.recovery.log_parser import LogParser
from agent.recovery.diagnosis_engine import DiagnosisEngine
from agent.recovery.recovery_planner import RecoveryPlanner
from tests.phase4d.mock_helpers import GMX_TOOLS_MODULE, make_gmx_mock

from .mock_llm import ScriptedLLM


def build_agent(tmp_path: Path, responses: list[str]) -> CodeAgent:
    llm   = ScriptedLLM(responses)
    tools = get_all_tools(work_dir=str(tmp_path))
    return CodeAgent(
        tools=tools,
        model=llm,
        max_steps=20,
        additional_authorized_imports=["pathlib", "json", "re"],
    )


class TestRecoveryLoop:

    def test_agent_reads_mdp_patches_from_diagnosis(self, tmp_path):
        """
        Agent receives a RECOVERABLE diagnosis and can extract
        MDP_PATCHES from the structured output string.
        """
        em_log = tmp_path / "em.log"
        em_log.write_text(
            "integrator              = steep\n"
            "nsteps                  = 1000\n"
            "Steepest Descents did not converge to Fmax < 1000 in 500 steps\n"
            "Potential Energy  = -2.34567e+05\n"
            "Maximum force     =  1.52340e+04\n"
        )

        agent = build_agent(tmp_path, [
            # Get diagnosis
            f'diagnosis = parse_gmx_log(log_file="{em_log}")\n'
            f'has_patches = "MDP_PATCHES" in diagnosis\n'
            f'is_recoverable = "RECOVERABLE" in diagnosis\n'
            f'print("Has patches:", has_patches)\n'
            f'print("Is recoverable:", is_recoverable)',
            'final_answer("Diagnosis read")',
        ])
        agent.run("Read EM diagnosis.")
        # If agent ran without error, it successfully processed the diagnosis

    def test_agent_applies_mdp_patch_via_write_file(self, tmp_path):
        """
        Agent reads diagnosis, extracts emstep patch, and applies it
        to the MDP file via write_file.
        """
        em_mdp = tmp_path / "em.mdp"
        em_mdp.write_text(
            "integrator = steep\n"
            "emstep     = 0.01\n"
            "nsteps     = 1000\n"
            "emtol      = 1000.0\n"
        )

        agent = build_agent(tmp_path, [
            # Read current MDP
            f'current_mdp = read_file(file_path="{em_mdp}")\n'
            f'print("Current MDP:", current_mdp)',
            # Apply patch — agent writes new emstep value
            f'new_content = current_mdp.replace("emstep     = 0.01", '
            f'"emstep     = 0.001")\n'
            f'new_content = new_content.replace("nsteps     = 1000", '
            f'"nsteps     = 5000")\n'
            f'write_file(file_path="{em_mdp}", content=new_content)',
            # Verify patch applied
            f'patched = read_file(file_path="{em_mdp}")\n'
            f'print("Patched MDP:", patched)',
            'final_answer("Patch applied")',
        ])
        agent.run("Apply MDP patch.")

        patched_content = em_mdp.read_text()
        assert "0.001" in patched_content
        assert "5000"  in patched_content

    def test_agent_reruns_grompp_after_patch(self, tmp_path, monkeypatch):
        """
        After patching MDP, agent re-runs grompp to generate new TPR.
        """
        em_mdp    = tmp_path / "em.mdp"
        conf_gro  = tmp_path / "conf.gro"
        topol_top = tmp_path / "topol.top"
        em_tpr    = tmp_path / "em.tpr"

        em_mdp.write_text("integrator = steep\nemstep = 0.01\nnsteps = 1000\n")
        conf_gro.write_text("mock GRO")
        topol_top.write_text("mock TOP")

        monkeypatch.setattr(
            GMX_TOOLS_MODULE,
            make_gmx_mock(create_files=[em_tpr]),
        )

        agent = build_agent(tmp_path, [
            # Patch MDP
            f'write_file(file_path="{em_mdp}", '
            f'content="integrator = steep\\nemstep = 0.001\\nnsteps = 5000\\n")',
            # Re-run grompp
            f'r = grompp(mdp_file="{em_mdp}", '
            f'input_gro="{conf_gro}", '
            f'topology_top="{topol_top}", '
            f'output_tpr="{em_tpr}")\n'
            f'print(r)',
            # Update state
            f'pipeline_state(action="update", updates={{'
            f'"files": {{"tpr_em": "{em_tpr}"}}, '
            f'"current_step": "mdrun_em_retry_1"}})',
            'final_answer("Retry grompp complete")',
        ])
        agent.run("Patch MDP and re-run grompp.")

        assert em_tpr.exists()
        patched = em_mdp.read_text()
        assert "0.001" in patched

    def test_recovery_diagnosis_to_patch_pipeline(self, tmp_path):
        """
        End-to-end test of the recovery pipeline:
            parse_gmx_log → diagnosis string → extract patches →
            write_file → verify patched MDP
        """
        # Create a failed EM log
        em_log = tmp_path / "em.log"
        em_log.write_text(
            "integrator              = steep\n"
            "nsteps                  = 1000\n"
            "Steepest Descents did not converge to Fmax < 1000 in 500 steps\n"
            "Potential Energy  = -2.34567e+05\n"
            "Maximum force     =  1.52340e+04\n"
        )

        # Create initial MDP
        em_mdp = tmp_path / "em.mdp"
        em_mdp.write_text(
            "integrator = steep\n"
            "emstep     = 0.01\n"
            "nsteps     = 1000\n"
            "emtol      = 1000.0\n"
        )

        agent = build_agent(tmp_path, [
            # Get diagnosis with patches
            f'diagnosis = parse_gmx_log(log_file="{em_log}")\n'
            f'print(diagnosis)',
            # Read current MDP
            f'mdp_content = read_file(file_path="{em_mdp}")\n'
            f'print(mdp_content)',
            # Apply patches from diagnosis
            # (agent parses MDP_PATCHES section and applies them)
            f'import re\n'
            f'patched = mdp_content\n'
            f'if "emstep" in diagnosis:\n'
            f'    patched = re.sub(r"emstep\\s*=\\s*[\\d.]+", '
            f'"emstep     = 0.001", patched)\n'
            f'if "nsteps" in diagnosis:\n'
            f'    patched = re.sub(r"nsteps\\s*=\\s*[\\d]+", '
            f'"nsteps     = 5000", patched)\n'
            f'write_file(file_path="{em_mdp}", content=patched)',
            'final_answer("Recovery complete")',
        ])
        agent.run("Execute recovery loop.")

        patched_content = em_mdp.read_text()
        assert "0.001" in patched_content or "5000" in patched_content

    def test_recovery_planner_output_is_agent_actionable(self, tmp_path):
        """
        Verify that the ActionRecommendation.to_agent_string() output
        contains all fields the agent needs to execute recovery —
        tested end-to-end through parse_gmx_log tool.
        """
        em_log = tmp_path / "em.log"
        em_log.write_text(
            "integrator              = steep\n"
            "nsteps                  = 1000\n"
            "Steepest Descents did not converge to Fmax < 1000 in 500 steps\n"
            "Maximum force     =  1.52340e+04\n"
        )

        # Call parse_gmx_log directly (not via agent) to inspect output
        from agent.tools.file_tools import ParseGmxLogTool
        tool   = ParseGmxLogTool()
        output = tool.forward(log_file=str(em_log))

        # All fields the agent needs must be present
        assert "DIAGNOSIS"          in output
        assert "SEVERITY"           in output
        assert "RECOVERABLE"        in output
        assert "PRIMARY_ACTION"     in output
        assert "MDP_PATCHES"        in output
        assert "RERUN_STEPS"        in output
        assert "AGENT_INSTRUCTION"  in output
        assert "FALLBACK_ACTION"    in output

        # MDP patches must contain actionable parameter names
        assert "emstep" in output or "nsteps" in output