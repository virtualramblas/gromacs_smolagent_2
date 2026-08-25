from __future__ import annotations

from pathlib import Path

import pytest

from agent.tools.gmx_tools import SolvateTool

from .mock_helpers import (
    GMX_TOOLS_MODULE,
    make_failing_mock,
    make_gmx_mock,
)


@pytest.fixture
def solvate_inputs(tmp_path):
    gro = tmp_path / "conf_box.gro"
    top = tmp_path / "topol.top"
    gro.write_text("mock GRO")
    top.write_text("mock TOP")
    return gro, top


class TestSolvateTool:

    def test_success_creates_solvated_gro(self, tmp_path, solvate_inputs, monkeypatch):
        gro, top = solvate_inputs
        tool     = SolvateTool(work_dir=tmp_path)
        out_gro  = tmp_path / "conf_solv.gro"
        monkeypatch.setattr(GMX_TOOLS_MODULE, make_gmx_mock(create_files=[out_gro]))
        result = tool.forward(input_gro=str(gro), topology_top=str(top))
        assert "SUCCESS: True" in result
        assert "conf_solv.gro" in result

    def test_topology_path_in_command(self, tmp_path, solvate_inputs, monkeypatch):
        gro, top = solvate_inputs
        tool     = SolvateTool(work_dir=tmp_path)
        captured = {}

        def _capture(args, work_dir, **kwargs):
            captured["args"] = args
            (tmp_path / "conf_solv.gro").write_text("mock")
            return 0, "", ""

        monkeypatch.setattr(GMX_TOOLS_MODULE, _capture)
        tool.forward(input_gro=str(gro), topology_top=str(top))
        assert "-p"     in captured["args"]
        assert str(top) in captured["args"]

    def test_solvent_model_passed_to_command(self, tmp_path, solvate_inputs, monkeypatch):
        gro, top = solvate_inputs
        tool     = SolvateTool(work_dir=tmp_path)
        captured = {}

        def _capture(args, work_dir, **kwargs):
            captured["args"] = args
            (tmp_path / "conf_solv.gro").write_text("mock")
            return 0, "", ""

        monkeypatch.setattr(GMX_TOOLS_MODULE, _capture)
        tool.forward(
            input_gro=str(gro),
            topology_top=str(top),
            solvent_model="spc216.gro",
        )
        assert "-cs"        in captured["args"]
        assert "spc216.gro" in captured["args"]

    def test_missing_output_gro_means_failure(
        self, tmp_path, solvate_inputs, monkeypatch
    ):
        gro, top = solvate_inputs
        tool     = SolvateTool(work_dir=tmp_path)
        monkeypatch.setattr(GMX_TOOLS_MODULE, make_gmx_mock(create_files=[]))
        result = tool.forward(input_gro=str(gro), topology_top=str(top))
        assert "SUCCESS: False" in result

    def test_water_count_extracted_from_stdout(
        self, tmp_path, solvate_inputs, monkeypatch
    ):
        gro, top = solvate_inputs
        tool     = SolvateTool(work_dir=tmp_path)
        out_gro  = tmp_path / "conf_solv.gro"

        def _mock(args, work_dir, **kwargs):
            out_gro.write_text("mock")
            return 0, "Number of solvent molecules: 12345", ""

        monkeypatch.setattr(GMX_TOOLS_MODULE, _mock)
        result = tool.forward(input_gro=str(gro), topology_top=str(top))
        assert "12345" in result or "solvent" in result.lower()

    def test_gmx_failure_reflected(self, tmp_path, solvate_inputs, monkeypatch):
        gro, top = solvate_inputs
        tool     = SolvateTool(work_dir=tmp_path)
        monkeypatch.setattr(
            GMX_TOOLS_MODULE,
            make_failing_mock(stderr="Fatal error: box too small"),
        )
        result = tool.forward(input_gro=str(gro), topology_top=str(top))
        assert "SUCCESS: False" in result