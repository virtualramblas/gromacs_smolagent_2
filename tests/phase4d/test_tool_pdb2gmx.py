"""
4D-2: Pdb2GmxTool — input construction, output file handling,
force field / water model passthrough, flag handling.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.tools.gmx_tools import Pdb2GmxTool

from .mock_helpers import (
    make_failing_mock,
    make_gmx_mock,
    make_warning_mock,
)

MODULE = "agent.tools.base.run_gmx_command"


@pytest.fixture
def pdb_file(tmp_path) -> Path:
    f = tmp_path / "eiwit.pdb"
    f.write_text("ATOM      1  N   ALA A   1       1.000   1.000   1.000\n")
    return f


class TestPdb2GmxTool:

    def test_success_creates_expected_outputs(self, tmp_path, pdb_file, monkeypatch):
        tool    = Pdb2GmxTool(work_dir=tmp_path)
        out_gro = tmp_path / "conf.gro"
        out_top = tmp_path / "topol.top"
        monkeypatch.setattr(
            MODULE,
            make_gmx_mock(create_files=[out_gro, out_top]),
        )
        result = tool.forward(pdb_file=str(pdb_file))
        assert "SUCCESS: True"  in result
        assert "conf.gro"       in result
        assert "topol.top"      in result

    def test_failure_when_gro_missing(self, tmp_path, pdb_file, monkeypatch):
        """If GMX returns 0 but output files are absent → SUCCESS: False."""
        tool = Pdb2GmxTool(work_dir=tmp_path)
        # Mock returns rc=0 but creates NO files
        monkeypatch.setattr(MODULE, make_gmx_mock(create_files=[]))
        result = tool.forward(pdb_file=str(pdb_file))
        assert "SUCCESS: False" in result

    def test_force_field_passed_to_command(self, tmp_path, pdb_file, monkeypatch):
        tool     = Pdb2GmxTool(work_dir=tmp_path)
        captured = {}

        def _capture(args, work_dir, **kwargs):
            captured["args"] = args
            out_gro = tmp_path / "conf.gro"
            out_top = tmp_path / "topol.top"
            out_gro.write_text("mock")
            out_top.write_text("mock")
            return 0, "", ""

        monkeypatch.setattr(MODULE, _capture)
        tool.forward(pdb_file=str(pdb_file), force_field="charmm36m")
        assert "-ff"        in captured["args"]
        assert "charmm36m"  in captured["args"]

    def test_water_model_passed_to_command(self, tmp_path, pdb_file, monkeypatch):
        tool     = Pdb2GmxTool(work_dir=tmp_path)
        captured = {}

        def _capture(args, work_dir, **kwargs):
            captured["args"] = args
            (tmp_path / "conf.gro").write_text("mock")
            (tmp_path / "topol.top").write_text("mock")
            return 0, "", ""

        monkeypatch.setattr(MODULE, _capture)
        tool.forward(pdb_file=str(pdb_file), water_model="tip4p")
        assert "-water" in captured["args"]
        assert "tip4p"  in captured["args"]

    def test_ignh_flag_added_when_ignore_hydrogens_true(
        self, tmp_path, pdb_file, monkeypatch
    ):
        tool     = Pdb2GmxTool(work_dir=tmp_path)
        captured = {}

        def _capture(args, work_dir, **kwargs):
            captured["args"] = args
            (tmp_path / "conf.gro").write_text("mock")
            (tmp_path / "topol.top").write_text("mock")
            return 0, "", ""

        monkeypatch.setattr(MODULE, _capture)
        tool.forward(pdb_file=str(pdb_file), ignore_hydrogens=True)
        assert "-ignh" in captured["args"]

    def test_ignh_flag_absent_when_ignore_hydrogens_false(
        self, tmp_path, pdb_file, monkeypatch
    ):
        tool     = Pdb2GmxTool(work_dir=tmp_path)
        captured = {}

        def _capture(args, work_dir, **kwargs):
            captured["args"] = args
            (tmp_path / "conf.gro").write_text("mock")
            (tmp_path / "topol.top").write_text("mock")
            return 0, "", ""

        monkeypatch.setattr(MODULE, _capture)
        tool.forward(pdb_file=str(pdb_file), ignore_hydrogens=False)
        assert "-ignh" not in captured["args"]

    def test_custom_output_paths_used(self, tmp_path, pdb_file, monkeypatch):
        tool        = Pdb2GmxTool(work_dir=tmp_path)
        custom_gro  = tmp_path / "custom.gro"
        custom_top  = tmp_path / "custom.top"
        monkeypatch.setattr(
            MODULE,
            make_gmx_mock(create_files=[custom_gro, custom_top]),
        )
        result = tool.forward(
            pdb_file=str(pdb_file),
            output_gro=str(custom_gro),
            output_top=str(custom_top),
        )
        assert "custom.gro" in result
        assert "custom.top" in result

    def test_warnings_captured_in_output(self, tmp_path, pdb_file, monkeypatch):
        tool    = Pdb2GmxTool(work_dir=tmp_path)
        out_gro = tmp_path / "conf.gro"
        out_top = tmp_path / "topol.top"
        monkeypatch.setattr(
            MODULE,
            make_warning_mock(
                create_files=[out_gro, out_top],
                warning_text="WARNING: missing hydrogen in chain B",
            ),
        )
        result = tool.forward(pdb_file=str(pdb_file))
        assert "WARNING"  in result
        assert "hydrogen" in result

    def test_gmx_failure_reflected_in_output(self, tmp_path, pdb_file, monkeypatch):
        tool = Pdb2GmxTool(work_dir=tmp_path)
        monkeypatch.setattr(
            MODULE,
            make_failing_mock(stderr="Fatal error: unknown residue LIG"),
        )
        result = tool.forward(pdb_file=str(pdb_file))
        assert "SUCCESS: False" in result
        assert "RETURN_CODE: 1" in result

    def test_default_output_paths_in_work_dir(self, tmp_path, pdb_file, monkeypatch):
        """Default output files should be placed in work_dir."""
        tool = Pdb2GmxTool(work_dir=tmp_path)
        monkeypatch.setattr(
            MODULE,
            make_gmx_mock(create_files=[
                tmp_path / "conf.gro",
                tmp_path / "topol.top",
            ]),
        )
        result = tool.forward(pdb_file=str(pdb_file))
        assert str(tmp_path) in result