from __future__ import annotations

from pathlib import Path

import pytest

from agent.tools.gmx_tools import MdrunTool

from .mock_helpers import (
    GMX_TOOLS_MODULE,
    make_failing_mock,
    make_gmx_mock,
    make_timeout_mock,
)


@pytest.fixture
def tpr_file(tmp_path) -> Path:
    f = tmp_path / "topol.tpr"
    f.write_text("mock TPR")
    return f


def _make_mdrun_outputs(work_dir: Path, label: str) -> list[Path]:
    return [
        work_dir / f"{label}.edr",
        work_dir / f"{label}.log",
    ]


class TestMdrunTool:

    def test_success_with_default_label(self, tmp_path, tpr_file, monkeypatch):
        tool = MdrunTool(work_dir=tmp_path)
        monkeypatch.setattr(
            GMX_TOOLS_MODULE,
            make_gmx_mock(create_files=_make_mdrun_outputs(tmp_path, "md")),
        )
        result = tool.forward(input_tpr=str(tpr_file))
        assert "SUCCESS: True" in result

    def test_run_label_used_in_command(self, tmp_path, tpr_file, monkeypatch):
        tool     = MdrunTool(work_dir=tmp_path)
        captured = {}

        def _capture(args, work_dir, **kwargs):
            captured["args"] = args
            for f in _make_mdrun_outputs(tmp_path, "em"):
                f.write_text("mock")
            return 0, "", ""

        monkeypatch.setattr(GMX_TOOLS_MODULE, _capture)
        tool.forward(input_tpr=str(tpr_file), run_label="em")
        assert "-deffnm" in captured["args"]
        assert "em"      in captured["args"]

    def test_thread_count_passed_to_command(self, tmp_path, tpr_file, monkeypatch):
        tool     = MdrunTool(work_dir=tmp_path)
        captured = {}

        def _capture(args, work_dir, **kwargs):
            captured["args"] = args
            for f in _make_mdrun_outputs(tmp_path, "md"):
                f.write_text("mock")
            return 0, "", ""

        monkeypatch.setattr(GMX_TOOLS_MODULE, _capture)
        tool.forward(input_tpr=str(tpr_file), n_threads=8)
        assert "-ntomp" in captured["args"]
        idx = captured["args"].index("-ntomp")
        assert captured["args"][idx + 1] == "8"

    def test_gpu_flag_added_when_use_gpu_true(self, tmp_path, tpr_file, monkeypatch):
        tool     = MdrunTool(work_dir=tmp_path)
        captured = {}

        def _capture(args, work_dir, **kwargs):
            captured["args"] = args
            for f in _make_mdrun_outputs(tmp_path, "md"):
                f.write_text("mock")
            return 0, "", ""

        monkeypatch.setattr(GMX_TOOLS_MODULE, _capture)
        tool.forward(input_tpr=str(tpr_file), use_gpu=True)
        assert "-nb"  in captured["args"]
        assert "gpu"  in captured["args"]

    def test_gpu_flag_absent_when_use_gpu_false(self, tmp_path, tpr_file, monkeypatch):
        tool     = MdrunTool(work_dir=tmp_path)
        captured = {}

        def _capture(args, work_dir, **kwargs):
            captured["args"] = args
            for f in _make_mdrun_outputs(tmp_path, "md"):
                f.write_text("mock")
            return 0, "", ""

        monkeypatch.setattr(GMX_TOOLS_MODULE, _capture)
        tool.forward(input_tpr=str(tpr_file), use_gpu=False)
        assert "-nb" not in captured["args"]

    def test_checkpoint_file_added_when_provided(
        self, tmp_path, tpr_file, monkeypatch
    ):
        cpt      = tmp_path / "state.cpt"
        cpt.write_text("mock CPT")
        tool     = MdrunTool(work_dir=tmp_path)
        captured = {}

        def _capture(args, work_dir, **kwargs):
            captured["args"] = args
            for f in _make_mdrun_outputs(tmp_path, "md"):
                f.write_text("mock")
            return 0, "", ""

        monkeypatch.setattr(GMX_TOOLS_MODULE, _capture)
        tool.forward(input_tpr=str(tpr_file), checkpoint_file=str(cpt))
        assert "-cpi"    in captured["args"]
        assert str(cpt)  in captured["args"]

    def test_extra_flags_appended_to_command(self, tmp_path, tpr_file, monkeypatch):
        tool     = MdrunTool(work_dir=tmp_path)
        captured = {}

        def _capture(args, work_dir, **kwargs):
            captured["args"] = args
            for f in _make_mdrun_outputs(tmp_path, "md"):
                f.write_text("mock")
            return 0, "", ""

        monkeypatch.setattr(GMX_TOOLS_MODULE, _capture)
        tool.forward(
            input_tpr=str(tpr_file),
            extra_flags=["-v", "-pin", "on"],
        )
        assert "-v"   in captured["args"]
        assert "-pin" in captured["args"]
        assert "on"   in captured["args"]

    def test_missing_edr_means_failure(self, tmp_path, tpr_file, monkeypatch):
        tool = MdrunTool(work_dir=tmp_path)
        monkeypatch.setattr(
            GMX_TOOLS_MODULE,
            make_gmx_mock(create_files=[tmp_path / "md.log"]),
        )
        result = tool.forward(input_tpr=str(tpr_file))
        assert "SUCCESS: False" in result

    def test_missing_log_means_failure(self, tmp_path, tpr_file, monkeypatch):
        tool = MdrunTool(work_dir=tmp_path)
        monkeypatch.setattr(
            GMX_TOOLS_MODULE,
            make_gmx_mock(create_files=[tmp_path / "md.edr"]),
        )
        result = tool.forward(input_tpr=str(tpr_file))
        assert "SUCCESS: False" in result

    def test_all_output_files_reported(self, tmp_path, tpr_file, monkeypatch):
        tool  = MdrunTool(work_dir=tmp_path)
        label = "nvt"
        all_outputs = [
            tmp_path / f"{label}.edr",
            tmp_path / f"{label}.log",
            tmp_path / f"{label}.xtc",
            tmp_path / f"{label}.cpt",
            tmp_path / f"{label}.gro",
        ]
        monkeypatch.setattr(GMX_TOOLS_MODULE, make_gmx_mock(create_files=all_outputs))
        result = tool.forward(input_tpr=str(tpr_file), run_label=label)
        for ext in ("edr", "log", "xtc", "cpt", "gro"):
            assert f"{label}.{ext}" in result

    def test_timeout_handled_gracefully(self, tmp_path, tpr_file, monkeypatch):
        tool = MdrunTool(work_dir=tmp_path)
        monkeypatch.setattr(GMX_TOOLS_MODULE, make_timeout_mock())
        result = tool.forward(input_tpr=str(tpr_file))
        assert "SUCCESS: False" in result

    @pytest.mark.parametrize("label", ["em", "nvt", "npt", "md"])
    def test_standard_run_labels_accepted(
        self, tmp_path, tpr_file, monkeypatch, label
    ):
        tool = MdrunTool(work_dir=tmp_path)
        monkeypatch.setattr(
            GMX_TOOLS_MODULE,
            make_gmx_mock(create_files=_make_mdrun_outputs(tmp_path, label)),
        )
        result = tool.forward(input_tpr=str(tpr_file), run_label=label)
        assert "SUCCESS: True" in result