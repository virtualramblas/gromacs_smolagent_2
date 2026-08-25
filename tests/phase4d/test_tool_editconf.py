from __future__ import annotations

from pathlib import Path

import pytest

from agent.tools.gmx_tools import EditconfTool

from .mock_helpers import (
    GMX_TOOLS_MODULE,
    make_failing_mock,
    make_gmx_mock,
)


@pytest.fixture
def input_gro(tmp_path) -> Path:
    f = tmp_path / "conf.gro"
    f.write_text("mock GRO content")
    return f


class TestEditconfTool:

    def test_success_creates_output_gro(self, tmp_path, input_gro, monkeypatch):
        tool    = EditconfTool(work_dir=tmp_path)
        out_gro = tmp_path / "conf_box.gro"
        monkeypatch.setattr(GMX_TOOLS_MODULE, make_gmx_mock(create_files=[out_gro]))
        result = tool.forward(input_gro=str(input_gro))
        assert "SUCCESS: True" in result
        assert "conf_box.gro"  in result

    def test_box_type_passed_to_command(self, tmp_path, input_gro, monkeypatch):
        tool     = EditconfTool(work_dir=tmp_path)
        captured = {}

        def _capture(args, work_dir, **kwargs):
            captured["args"] = args
            (tmp_path / "conf_box.gro").write_text("mock")
            return 0, "", ""

        monkeypatch.setattr(GMX_TOOLS_MODULE, _capture)
        tool.forward(input_gro=str(input_gro), box_type="cubic")
        assert "-bt"   in captured["args"]
        assert "cubic" in captured["args"]

    def test_distance_passed_to_command(self, tmp_path, input_gro, monkeypatch):
        tool     = EditconfTool(work_dir=tmp_path)
        captured = {}

        def _capture(args, work_dir, **kwargs):
            captured["args"] = args
            (tmp_path / "conf_box.gro").write_text("mock")
            return 0, "", ""

        monkeypatch.setattr(GMX_TOOLS_MODULE, _capture)
        tool.forward(input_gro=str(input_gro), distance=1.2)
        assert "-d"   in captured["args"]
        assert "1.2"  in captured["args"]

    @pytest.mark.parametrize("box_type", ["cubic", "dodecahedron", "triclinic"])
    def test_all_box_types_accepted(self, tmp_path, input_gro, monkeypatch, box_type):
        tool = EditconfTool(work_dir=tmp_path)
        monkeypatch.setattr(
            GMX_TOOLS_MODULE,
            make_gmx_mock(create_files=[tmp_path / "conf_box.gro"]),
        )
        result = tool.forward(input_gro=str(input_gro), box_type=box_type)
        assert "SUCCESS: True" in result

    def test_missing_output_file_means_failure(self, tmp_path, input_gro, monkeypatch):
        tool = EditconfTool(work_dir=tmp_path)
        monkeypatch.setattr(GMX_TOOLS_MODULE, make_gmx_mock(create_files=[]))
        result = tool.forward(input_gro=str(input_gro))
        assert "SUCCESS: False" in result

    def test_failure_returncode_means_failure(self, tmp_path, input_gro, monkeypatch):
        tool = EditconfTool(work_dir=tmp_path)
        monkeypatch.setattr(
            GMX_TOOLS_MODULE,
            make_failing_mock(stderr="Fatal error: bad box"),
        )
        result = tool.forward(input_gro=str(input_gro))
        assert "SUCCESS: False" in result

    def test_summary_contains_box_type_and_distance(
        self, tmp_path, input_gro, monkeypatch
    ):
        tool = EditconfTool(work_dir=tmp_path)
        monkeypatch.setattr(
            GMX_TOOLS_MODULE,
            make_gmx_mock(create_files=[tmp_path / "conf_box.gro"]),
        )
        result = tool.forward(
            input_gro=str(input_gro),
            box_type="dodecahedron",
            distance=1.0,
        )
        assert "dodecahedron" in result
        assert "1.0"          in result