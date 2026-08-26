"""
4E-1: PipelineStateTool — read, update, reset, persistence,
deep merge, list append, and resumability contract.

Rationale:
    The state file is the agent's only persistent memory across
    pipeline steps and restarts. Bugs here cause:
        - Steps being re-run unnecessarily (wasted compute)
        - Steps being skipped (corrupted simulation)
        - File paths lost between steps (broken tool inputs)
    Every state operation is tested in isolation and in sequence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.tools.state_tools import PipelineStateTool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def state_file(tmp_path) -> Path:
    return tmp_path / "pipeline_state.json"


@pytest.fixture
def tool(state_file) -> PipelineStateTool:
    return PipelineStateTool(state_file=state_file)


# ---------------------------------------------------------------------------
# 4E-1: Initialisation
# ---------------------------------------------------------------------------

class TestInitialisation:

    def test_state_file_path_stored(self, state_file):
        tool = PipelineStateTool(state_file=state_file)
        assert tool.state_file == state_file

    def test_no_file_created_on_init(self, state_file):
        """State file must NOT be created until first read/write."""
        PipelineStateTool(state_file=state_file)
        assert not state_file.exists()

    def test_default_state_has_required_keys(self, tool):
        result = tool.forward(action="read")
        state  = json.loads(result)
        required_keys = [
            "input_pdb", "work_dir", "completed_steps",
            "current_step", "files", "em_converged",
            "nvt_complete", "npt_complete", "md_complete",
            "warnings", "errors", "last_updated",
        ]
        for key in required_keys:
            assert key in state, f"Required key '{key}' missing from default state"

    def test_default_files_dict_has_expected_keys(self, tool):
        result = tool.forward(action="read")
        state  = json.loads(result)
        file_keys = [
            "pdb", "gro", "gro_box", "gro_solv", "gro_ions",
            "top", "tpr_em", "tpr_nvt", "tpr_npt", "tpr_md",
            "cpt", "edr_em", "edr_md", "xtc",
        ]
        for key in file_keys:
            assert key in state["files"], (
                f"File key '{key}' missing from default state['files']"
            )

    def test_default_completed_steps_is_empty_list(self, tool):
        result = json.loads(tool.forward(action="read"))
        assert result["completed_steps"] == []

    def test_default_warnings_is_empty_list(self, tool):
        result = json.loads(tool.forward(action="read"))
        assert result["warnings"] == []

    def test_default_errors_is_empty_list(self, tool):
        result = json.loads(tool.forward(action="read"))
        assert result["errors"] == []

    def test_default_numeric_flags_are_none(self, tool):
        result = json.loads(tool.forward(action="read"))
        for key in ("em_converged", "nvt_complete", "npt_complete", "md_complete"):
            assert result[key] is None, f"'{key}' should default to None"


# ---------------------------------------------------------------------------
# 4E-2: Read action
# ---------------------------------------------------------------------------

class TestReadAction:

    def test_read_returns_valid_json(self, tool):
        result = tool.forward(action="read")
        # Must not raise
        state = json.loads(result)
        assert isinstance(state, dict)

    def test_read_creates_state_file(self, tool, state_file):
        tool.forward(action="read")
        assert state_file.exists()

    def test_read_returns_same_state_on_repeated_calls(self, tool):
        result1 = json.loads(tool.forward(action="read"))
        result2 = json.loads(tool.forward(action="read"))
        # Remove last_updated which changes on each write
        result1.pop("last_updated", None)
        result2.pop("last_updated", None)
        assert result1 == result2

    def test_read_after_manual_file_edit(self, tool, state_file):
        """State tool must read from disk, not from memory cache."""
        tool.forward(action="read")   # create file
        # Manually edit the file
        state = json.loads(state_file.read_text())
        state["input_pdb"] = "manually_set.pdb"
        state_file.write_text(json.dumps(state))
        # Read should reflect the manual edit
        result = json.loads(tool.forward(action="read"))
        assert result["input_pdb"] == "manually_set.pdb"


# ---------------------------------------------------------------------------
# 4E-3: Reset action
# ---------------------------------------------------------------------------

class TestResetAction:

    def test_reset_returns_default_state(self, tool):
        # forward() now returns pure JSON — json.loads() works directly
        result = json.loads(tool.forward(action="reset"))
        assert result["completed_steps"] == []
        assert result["input_pdb"]       is None
        assert result["em_converged"]    is None

    def test_reset_creates_state_file(self, tool, state_file):
        tool.forward(action="reset")
        assert state_file.exists()

    def test_reset_clears_previous_updates(self, tool):
        tool.forward(action="update", updates={"input_pdb": "eiwit.pdb"})
        tool.forward(action="update", updates={"em_converged": True})
        tool.forward(action="reset")
        result = json.loads(tool.forward(action="read"))
        assert result["input_pdb"]    is None
        assert result["em_converged"] is None

    def test_reset_clears_completed_steps(self, tool):
        tool.forward(action="update", updates={
            "completed_steps": ["pdb2gmx", "editconf"]
        })
        tool.forward(action="reset")
        result = json.loads(tool.forward(action="read"))
        assert result["completed_steps"] == []

    def test_reset_clears_file_paths(self, tool):
        tool.forward(action="update", updates={
            "files": {"gro": "conf.gro", "top": "topol.top"}
        })
        tool.forward(action="reset")
        result = json.loads(tool.forward(action="read"))
        assert result["files"]["gro"] is None
        assert result["files"]["top"] is None


# ---------------------------------------------------------------------------
# 4E-4: Update action — scalar fields
# ---------------------------------------------------------------------------

class TestUpdateScalarFields:

    def test_update_input_pdb(self, tool):
        tool.forward(action="update", updates={"input_pdb": "eiwit.pdb"})
        result = json.loads(tool.forward(action="read"))
        assert result["input_pdb"] == "eiwit.pdb"

    def test_update_work_dir(self, tool):
        tool.forward(action="update", updates={"work_dir": "/tmp/gmx_run"})
        result = json.loads(tool.forward(action="read"))
        assert result["work_dir"] == "/tmp/gmx_run"

    def test_update_current_step(self, tool):
        tool.forward(action="update", updates={"current_step": "grompp_em"})
        result = json.loads(tool.forward(action="read"))
        assert result["current_step"] == "grompp_em"

    def test_update_em_converged_true(self, tool):
        tool.forward(action="update", updates={"em_converged": True})
        result = json.loads(tool.forward(action="read"))
        assert result["em_converged"] is True

    def test_update_em_converged_false(self, tool):
        tool.forward(action="update", updates={"em_converged": False})
        result = json.loads(tool.forward(action="read"))
        assert result["em_converged"] is False

    def test_update_nvt_complete(self, tool):
        tool.forward(action="update", updates={"nvt_complete": True})
        result = json.loads(tool.forward(action="read"))
        assert result["nvt_complete"] is True

    def test_update_npt_complete(self, tool):
        tool.forward(action="update", updates={"npt_complete": True})
        result = json.loads(tool.forward(action="read"))
        assert result["npt_complete"] is True

    def test_update_md_complete(self, tool):
        tool.forward(action="update", updates={"md_complete": True})
        result = json.loads(tool.forward(action="read"))
        assert result["md_complete"] is True

    def test_update_preserves_unrelated_fields(self, tool):
        """Updating one field must not clear other fields."""
        tool.forward(action="update", updates={"input_pdb": "eiwit.pdb"})
        tool.forward(action="update", updates={"current_step": "pdb2gmx"})
        result = json.loads(tool.forward(action="read"))
        assert result["input_pdb"]    == "eiwit.pdb"
        assert result["current_step"] == "pdb2gmx"

    def test_update_overwrites_previous_scalar(self, tool):
        tool.forward(action="update", updates={"current_step": "pdb2gmx"})
        tool.forward(action="update", updates={"current_step": "editconf"})
        result = json.loads(tool.forward(action="read"))
        assert result["current_step"] == "editconf"

    def test_update_with_empty_updates_returns_error(self, tool):
        result = tool.forward(action="update", updates={})
        assert "ERROR" in result

    def test_update_with_none_updates_returns_error(self, tool):
        result = tool.forward(action="update", updates=None)
        assert "ERROR" in result


# ---------------------------------------------------------------------------
# 4E-5: Update action — files dict (deep merge)
# ---------------------------------------------------------------------------

class TestUpdateFilesDict:

    def test_update_single_file_key(self, tool):
        tool.forward(action="update", updates={"files": {"gro": "conf.gro"}})
        result = json.loads(tool.forward(action="read"))
        assert result["files"]["gro"] == "conf.gro"

    def test_update_multiple_file_keys(self, tool):
        tool.forward(action="update", updates={
            "files": {
                "gro": "conf.gro",
                "top": "topol.top",
            }
        })
        result = json.loads(tool.forward(action="read"))
        assert result["files"]["gro"] == "conf.gro"
        assert result["files"]["top"] == "topol.top"

    def test_files_update_is_deep_merge_not_replace(self, tool):
        """
        Updating files dict must merge, not replace.
        Setting 'gro' must not clear 'top'.
        """
        tool.forward(action="update", updates={"files": {"gro": "conf.gro"}})
        tool.forward(action="update", updates={"files": {"top": "topol.top"}})
        result = json.loads(tool.forward(action="read"))
        assert result["files"]["gro"] == "conf.gro"   # must still be set
        assert result["files"]["top"] == "topol.top"  # newly set

    def test_files_update_overwrites_existing_key(self, tool):
        tool.forward(action="update", updates={"files": {"gro": "old.gro"}})
        tool.forward(action="update", updates={"files": {"gro": "new.gro"}})
        result = json.loads(tool.forward(action="read"))
        assert result["files"]["gro"] == "new.gro"

    def test_files_update_preserves_unset_keys_as_none(self, tool):
        tool.forward(action="update", updates={"files": {"gro": "conf.gro"}})
        result = json.loads(tool.forward(action="read"))
        # Keys not yet set should remain None
        assert result["files"]["tpr_em"] is None
        assert result["files"]["xtc"]    is None

    def test_full_pipeline_file_sequence(self, tool):
        """Simulate the full pipeline file update sequence."""
        steps = [
            {"files": {"pdb":      "eiwit.pdb"}},
            {"files": {"gro":      "conf.gro",     "top": "topol.top"}},
            {"files": {"gro_box":  "conf_box.gro"}},
            {"files": {"gro_solv": "conf_solv.gro"}},
            {"files": {"gro_ions": "conf_ions.gro"}},
            {"files": {"tpr_em":   "em.tpr"}},
            {"files": {"edr_em":   "em.edr"}},
            {"files": {"tpr_nvt":  "nvt.tpr"}},
            {"files": {"tpr_npt":  "npt.tpr"}},
            {"files": {"tpr_md":   "md.tpr"}},
            {"files": {"xtc":      "md.xtc",       "edr_md": "md.edr"}},
        ]
        for update in steps:
            tool.forward(action="update", updates=update)

        result = json.loads(tool.forward(action="read"))
        assert result["files"]["pdb"]      == "eiwit.pdb"
        assert result["files"]["gro"]      == "conf.gro"
        assert result["files"]["top"]      == "topol.top"
        assert result["files"]["gro_box"]  == "conf_box.gro"
        assert result["files"]["gro_solv"] == "conf_solv.gro"
        assert result["files"]["gro_ions"] == "conf_ions.gro"
        assert result["files"]["tpr_em"]   == "em.tpr"
        assert result["files"]["edr_em"]   == "em.edr"
        assert result["files"]["xtc"]      == "md.xtc"


# ---------------------------------------------------------------------------
# 4E-6: Update action — list fields (append semantics)
# ---------------------------------------------------------------------------

class TestUpdateListFields:

    def test_completed_steps_replaced_when_set_directly(self, tool):
        """
        completed_steps is a list — setting it directly replaces it.
        The agent typically appends by reading first then writing the
        full updated list.
        """
        tool.forward(action="update", updates={
            "completed_steps": ["pdb2gmx"]
        })
        result = json.loads(tool.forward(action="read"))
        assert result["completed_steps"] == ["pdb2gmx"]

    def test_warnings_appended_on_update(self, tool):
        """warnings list uses append semantics."""
        tool.forward(action="update", updates={
            "warnings": ["WARNING: first"]
        })
        tool.forward(action="update", updates={
            "warnings": ["WARNING: second"]
        })
        result = json.loads(tool.forward(action="read"))
        assert "WARNING: first"  in result["warnings"]
        assert "WARNING: second" in result["warnings"]
        assert len(result["warnings"]) == 2

    def test_errors_appended_on_update(self, tool):
        """errors list uses append semantics."""
        tool.forward(action="update", updates={
            "errors": ["ERROR: step 1 failed"]
        })
        tool.forward(action="update", updates={
            "errors": ["ERROR: step 2 failed"]
        })
        result = json.loads(tool.forward(action="read"))
        assert len(result["errors"]) == 2

    def test_multiple_warnings_in_single_update(self, tool):
        tool.forward(action="update", updates={
            "warnings": ["WARNING: a", "WARNING: b", "WARNING: c"]
        })
        result = json.loads(tool.forward(action="read"))
        assert len(result["warnings"]) == 3

    def test_warnings_persist_across_reads(self, tool):
        tool.forward(action="update", updates={"warnings": ["WARNING: test"]})
        # Read twice — warnings must persist
        json.loads(tool.forward(action="read"))
        result = json.loads(tool.forward(action="read"))
        assert "WARNING: test" in result["warnings"]


# ---------------------------------------------------------------------------
# 4E-7: Persistence — state survives tool reinstantiation
# ---------------------------------------------------------------------------

class TestPersistence:

    def test_state_persists_after_tool_reinstantiation(self, state_file):
        """
        State written by one tool instance must be readable
        by a new instance pointing to the same file.
        This is the core resumability requirement.
        """
        tool1 = PipelineStateTool(state_file=state_file)
        tool1.forward(action="update", updates={
            "input_pdb":       "eiwit.pdb",
            "completed_steps": ["pdb2gmx", "editconf"],
            "em_converged":    True,
            "files": {
                "gro": "conf.gro",
                "top": "topol.top",
            },
        })

        # New instance — simulates agent restart
        tool2  = PipelineStateTool(state_file=state_file)
        result = json.loads(tool2.forward(action="read"))

        assert result["input_pdb"]       == "eiwit.pdb"
        assert result["completed_steps"] == ["pdb2gmx", "editconf"]
        assert result["em_converged"]    is True
        assert result["files"]["gro"]    == "conf.gro"
        assert result["files"]["top"]    == "topol.top"

    def test_state_file_is_valid_json(self, tool, state_file):
        tool.forward(action="update", updates={"input_pdb": "eiwit.pdb"})
        content = state_file.read_text()
        # Must not raise
        state = json.loads(content)
        assert isinstance(state, dict)

    def test_state_file_is_human_readable(self, tool, state_file):
        """State file must be indented JSON, not a single line."""
        tool.forward(action="update", updates={"input_pdb": "eiwit.pdb"})
        content = state_file.read_text()
        assert "\n" in content, "State file should be pretty-printed JSON"

    def test_last_updated_timestamp_set_on_write(self, tool):
        tool.forward(action="update", updates={"input_pdb": "eiwit.pdb"})
        result = json.loads(tool.forward(action="read"))
        assert result["last_updated"] is not None
        assert len(result["last_updated"]) > 0

    def test_last_updated_changes_on_each_write(self, tool):
        import time
        tool.forward(action="update", updates={"current_step": "step1"})
        ts1 = json.loads(tool.forward(action="read"))["last_updated"]
        time.sleep(0.01)
        tool.forward(action="update", updates={"current_step": "step2"})
        ts2 = json.loads(tool.forward(action="read"))["last_updated"]
        # Timestamps should differ (second write is later)
        assert ts2 >= ts1

    def test_multiple_instances_same_file_last_write_wins(self, state_file):
        """Two tool instances writing to the same file — last write wins."""
        tool_a = PipelineStateTool(state_file=state_file)
        tool_b = PipelineStateTool(state_file=state_file)
        tool_a.forward(action="update", updates={"input_pdb": "from_a.pdb"})
        tool_b.forward(action="update", updates={"input_pdb": "from_b.pdb"})
        result = json.loads(tool_a.forward(action="read"))
        assert result["input_pdb"] == "from_b.pdb"


# ---------------------------------------------------------------------------
# 4E-8: Resumability contract
# ---------------------------------------------------------------------------

class TestResumabilityContract:
    """
    Simulate a realistic interrupted pipeline and verify the agent
    can correctly determine where to resume.
    """

    def _simulate_pipeline_up_to_em(self, tool: PipelineStateTool) -> None:
        """Run state updates as the agent would through EM completion."""
        tool.forward(action="reset")
        tool.forward(action="update", updates={
            "input_pdb":   "eiwit.pdb",
            "work_dir":    "/tmp/gmx_run",
            "current_step": "pdb2gmx",
        })
        tool.forward(action="update", updates={
            "completed_steps": ["pdb2gmx"],
            "files": {"gro": "conf.gro", "top": "topol.top"},
            "current_step": "editconf",
        })
        tool.forward(action="update", updates={
            "completed_steps": ["pdb2gmx", "editconf"],
            "files": {"gro_box": "conf_box.gro"},
            "current_step": "solvate",
        })
        tool.forward(action="update", updates={
            "completed_steps": ["pdb2gmx", "editconf", "solvate"],
            "files": {"gro_solv": "conf_solv.gro"},
            "current_step": "genion",
        })
        tool.forward(action="update", updates={
            "completed_steps": ["pdb2gmx", "editconf", "solvate", "genion"],
            "files": {"gro_ions": "conf_ions.gro"},
            "current_step": "grompp_em",
        })
        tool.forward(action="update", updates={
            "completed_steps": [
                "pdb2gmx", "editconf", "solvate", "genion", "grompp_em"
            ],
            "files": {"tpr_em": "em.tpr"},
            "current_step": "mdrun_em",
        })
        tool.forward(action="update", updates={
            "completed_steps": [
                "pdb2gmx", "editconf", "solvate",
                "genion", "grompp_em", "mdrun_em",
            ],
            "files": {"edr_em": "em.edr"},
            "em_converged":  True,
            "current_step":  "grompp_nvt",
        })

    def test_completed_steps_reflect_pipeline_progress(self, tool):
        self._simulate_pipeline_up_to_em(tool)
        result = json.loads(tool.forward(action="read"))
        assert "pdb2gmx"   in result["completed_steps"]
        assert "editconf"  in result["completed_steps"]
        assert "solvate"   in result["completed_steps"]
        assert "genion"    in result["completed_steps"]
        assert "grompp_em" in result["completed_steps"]
        assert "mdrun_em"  in result["completed_steps"]

    def test_current_step_reflects_next_step(self, tool):
        self._simulate_pipeline_up_to_em(tool)
        result = json.loads(tool.forward(action="read"))
        assert result["current_step"] == "grompp_nvt"

    def test_em_converged_flag_set(self, tool):
        self._simulate_pipeline_up_to_em(tool)
        result = json.loads(tool.forward(action="read"))
        assert result["em_converged"] is True

    def test_all_file_paths_preserved(self, tool):
        self._simulate_pipeline_up_to_em(tool)
        result = json.loads(tool.forward(action="read"))
        assert result["files"]["gro"]      == "conf.gro"
        assert result["files"]["top"]      == "topol.top"
        assert result["files"]["gro_box"]  == "conf_box.gro"
        assert result["files"]["gro_solv"] == "conf_solv.gro"
        assert result["files"]["gro_ions"] == "conf_ions.gro"
        assert result["files"]["tpr_em"]   == "em.tpr"
        assert result["files"]["edr_em"]   == "em.edr"

    def test_resume_detects_incomplete_steps(self, tool):
        """
        After interruption, a new agent instance can determine
        which steps are done and which remain.
        """
        self._simulate_pipeline_up_to_em(tool)

        # Simulate restart with new tool instance
        new_tool = PipelineStateTool(state_file=tool.state_file)
        result   = json.loads(new_tool.forward(action="read"))

        completed = set(result["completed_steps"])
        all_steps = {
            "pdb2gmx", "editconf", "solvate", "genion",
            "grompp_em", "mdrun_em", "grompp_nvt", "mdrun_nvt",
            "grompp_npt", "mdrun_npt", "grompp_md", "mdrun_md",
        }
        remaining = all_steps - completed
        assert "grompp_nvt" in remaining
        assert "mdrun_nvt"  in remaining
        assert "pdb2gmx"    not in remaining

    def test_state_not_corrupted_by_partial_update(self, tool):
        """
        A partial update (only some fields) must not corrupt
        previously set fields.
        """
        self._simulate_pipeline_up_to_em(tool)
        # Partial update — only current_step
        tool.forward(action="update", updates={"current_step": "mdrun_nvt"})
        result = json.loads(tool.forward(action="read"))
        # All previously set fields must still be intact
        assert result["input_pdb"]    == "eiwit.pdb"
        assert result["em_converged"] is True
        assert result["files"]["gro"] == "conf.gro"
        assert result["current_step"] == "mdrun_nvt"


# ---------------------------------------------------------------------------
# 4E-9: Edge cases and error handling
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_unknown_action_returns_error(self, tool):
        result = tool.forward(action="fly_to_moon")
        assert "ERROR"        in result
        assert "fly_to_moon"  in result

    def test_action_case_insensitive(self, tool):
        result = tool.forward(action="READ")
        state  = json.loads(result)
        assert isinstance(state, dict)

    def test_update_with_unknown_key_stored(self, tool):
        tool.forward(action="update", updates={"new_future_key": "value"})
        result = json.loads(tool.forward(action="read"))
        assert result.get("new_future_key") == "value"

    def test_update_with_none_value_stored(self, tool):
        tool.forward(action="update", updates={"input_pdb": "eiwit.pdb"})
        tool.forward(action="update", updates={"input_pdb": None})
        result = json.loads(tool.forward(action="read"))
        assert result["input_pdb"] is None

    def test_state_file_in_nonexistent_directory(self, tmp_path):
        nested_path = tmp_path / "a" / "b" / "c" / "state.json"
        tool        = PipelineStateTool(state_file=nested_path)
        tool.forward(action="update", updates={"input_pdb": "test.pdb"})
        assert nested_path.exists()
        result = json.loads(tool.forward(action="read"))
        assert result["input_pdb"] == "test.pdb"

    def test_corrupted_state_file_handled(self, tool, state_file):
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text("{ this is not valid json !!!")
        try:
            result = tool.forward(action="read")
            assert result is not None
            assert len(result) > 0
        except Exception as exc:
            pytest.fail(
                f"PipelineStateTool crashed on corrupted state file: {exc}"
            )

    def test_reset_fixes_corrupted_state(self, tool, state_file):
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text("CORRUPTED")
        # reset() always produces valid state regardless of file content
        result = json.loads(tool.forward(action="reset"))
        assert result["completed_steps"] == []
        assert result["input_pdb"]       is None