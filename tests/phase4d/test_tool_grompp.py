"""
4D-5: GromppTool — MDP/GRO/TOP inputs, optional flags, maxwarn.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.tools.gmx_tools import GromppTool

from .mock_helpers import make_failing_mock, make_gmx_mock, make_warning_mock

MODULE = "agent.tools.base.run_gmx_command"


@pytest.fixture
def grompp_inputs(tmp_path):
    mdp = tmp_path / "em.mdp"
    gro = tmp_path / "conf.gro"
    top = tmp_path / "topol.top"
    mdp.write_text("integrator = steep\n")
    gro.write_text("mock GRO")
    top.write_text("mock TOP")
    return mdp, gro, top


class TestGromppTool:

    def test_success_creates_tpr(self, tmp_path, grompp_inputs, monkeypatch):
        mdp, gro, top = grompp_inputs
        tool          = GromppTool(work_dir=tmp_path)
        out_tpr       = tmp_path / "topol.tpr"
        monkeypatch.setattr(MODULE, make_gmx_mock(create_files=[out_tpr]))
        result = tool.forward(
            mdp_file=str(mdp),
            input_gro=str(gro),
            topology_top=str(top),
        )
        assert "SUCCESS: True" in result
        assert "topol.tpr"     in result

    def test_all_required_flags_in_command(self, tmp_path, grompp_inputs, monkeypatch):
        mdp, gro, top = grompp_inputs
        tool          = GromppTool(work_dir=tmp_path)
        captured      = {}

        def _capture(args, work_dir, **kwargs):
            captured["args"] = args
            (tmp_path / "topol.tpr").write_text("mock")
            return 0, "", ""

        monkeypatch.setattr(MODULE, _capture)
        tool.forward(
            mdp_file=str(mdp),
            input_gro=str(gro),
            topology_top=str(top),
        )
        assert "-f" in captured["args"]
        assert "-c" in captured["args"]
        assert "-p" in captured["args"]
        assert "-o" in captured["args"]

    def test_maxwarn_passed_to_command(self, tmp_path, grompp_inputs, monkeypatch):
        mdp, gro, top = grompp_inputs
        tool          = GromppTool(work_dir=tmp_path)
        captured      = {}

        def _capture(args, work_dir, **kwargs):
            captured["args"] = args
            (tmp_path / "topol.tpr").write_text("mock")
            return 0, "", ""

        monkeypatch.setattr(MODULE, _capture)
        tool.forward(
            mdp_file=str(mdp),
            input_gro=str(gro),
            topology_top=str(top),
            max_warnings=2,
        )
        assert "-maxwarn" in captured["args"]
        idx = captured["args"].index("-maxwarn")
        assert captured["args"][idx + 1] == "2"

    def test_optional_index_file_added_when_provided(
        self, tmp_path, grompp_inputs, monkeypatch
    ):
        mdp, gro, top = grompp_inputs
        ndx           = tmp_path / "index.ndx"
        ndx.write_text("mock NDX")
        tool          = GromppTool(work_dir=tmp_path)
        captured      = {}

        def _capture(args, work_dir, **kwargs):
            captured["args"] = args
            (tmp_path / "topol.tpr").write_text("mock")
            return 0, "", ""

        monkeypatch.setattr(MODULE, _capture)
        tool.forward(
            mdp_file=str(mdp),
            input_gro=str(gro),
            topology_top=str(top),
            index_file=str(ndx),
        )
        assert "-n"      in captured["args"]
        assert str(ndx)  in captured["args"]

    def test_index_file_absent_when_not_provided(
        self, tmp_path, grompp_inputs, monkeypatch
    ):
        mdp, gro, top = grompp_inputs
        tool          = GromppTool(work_dir=tmp_path)
        captured      = {}

        def _capture(args, work_dir, **kwargs):
            captured["args"] = args
            (tmp_path / "topol.tpr").write_text("mock")
            return 0, "", ""

        monkeypatch.setattr(MODULE, _capture)
        tool.forward(
            mdp_file=str(mdp),
            input_gro=str(gro),
            topology_top=str(top),
        )
        assert "-n" not in captured["args"]

    def test_checkpoint_file_added_when_provided(
        self, tmp_path, grompp_inputs, monkeypatch
    ):
        mdp, gro, top = grompp_inputs
        cpt           = tmp_path / "state.cpt"
        cpt.write_text("mock CPT")
        tool          = GromppTool(work_dir=tmp_path)
        captured      = {}

        def _capture(args, work_dir, **kwargs):
            captured["args"] = args
            (tmp_path / "topol.tpr").write_text("mock")
            return 0, "", ""

        monkeypatch.setattr(MODULE, _capture)
        tool.forward(
            mdp_file=str(mdp),
            input_gro=str(gro),
            topology_top=str(top),
            checkpoint_file=str(cpt),
        )
        assert "-t"      in captured["args"]
        assert str(cpt)  in captured["args"]

    def test_missing_tpr_means_failure(self, tmp_path, grompp_inputs, monkeypatch):
        mdp, gro, top = grompp_inputs
        tool          = GromppTool(work_dir=tmp_path)
        monkeypatch.setattr(MODULE, make_gmx_mock(create_files=[]))
        result = tool.forward(
            mdp_file=str(mdp),
            input_gro=str(gro),
            topology_top=str(top),
        )
        assert "SUCCESS: False" in result

    def test_warnings_in_output(self, tmp_path, grompp_inputs, monkeypatch):
        mdp, gro, top = grompp_inputs
        tool          = GromppTool(work_dir=tmp_path)
        out_tpr       = tmp_path / "topol.tpr"
        monkeypatch.setattr(
            MODULE,
            make_warning_mock(
                create_files=[out_tpr],
                warning_text="WARNING: 1-4 interaction not set",
            ),
        )
        result = tool.forward(
            mdp_file=str(mdp),
            input_gro=str(gro),
            topology_top=str(top),
        )
        assert "WARNING" in result

    def test_custom_tpr_output_path(self, tmp_path, grompp_inputs, monkeypatch):
        mdp, gro, top = grompp_inputs
        custom_tpr    = tmp_path / "em.tpr"
        tool          = GromppTool(work_dir=tmp_path)
        monkeypatch.setattr(MODULE, make_gmx_mock(create_files=[custom_tpr]))
        result = tool.forward(
            mdp_file=str(mdp),
            input_gro=str(gro),
            topology_top=str(top),
            output_tpr=str(custom_tpr),
        )
        assert "em.tpr" in result