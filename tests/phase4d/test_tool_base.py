from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agent.tools.gmx_tools import EditconfTool
from agent.tools.base import GMXResult

from .mock_helpers import (
    GMX_TOOLS_MODULE,
    make_env_error_mock,
    make_failing_mock,
    make_gmx_mock,
    make_timeout_mock,
)


class TestSafeRun:

    def test_success_returns_agent_string(self, tmp_path, monkeypatch):
        tool    = EditconfTool(work_dir=tmp_path)
        out_gro = tmp_path / "conf_box.gro"
        monkeypatch.setattr(
            GMX_TOOLS_MODULE,                        # ← correct target
            make_gmx_mock(create_files=[out_gro]),
        )
        result = tool.forward(
            input_gro=str(tmp_path / "conf.gro"),
            output_gro=str(out_gro),
        )
        assert "SUCCESS: True" in result

    def test_environment_error_returns_error_string(self, tmp_path, monkeypatch):
        tool = EditconfTool(work_dir=tmp_path)
        monkeypatch.setattr(GMX_TOOLS_MODULE, make_env_error_mock())
        result = tool.forward(input_gro=str(tmp_path / "conf.gro"))
        assert "SUCCESS: False"  in result
        assert "GROMACS binary"  in result or "Environment error" in result

    def test_timeout_returns_error_string(self, tmp_path, monkeypatch):
        tool = EditconfTool(work_dir=tmp_path)
        monkeypatch.setattr(GMX_TOOLS_MODULE, make_timeout_mock())
        result = tool.forward(input_gro=str(tmp_path / "conf.gro"))
        assert "SUCCESS: False" in result
        assert "timeout" in result.lower() or "timed out" in result.lower()

    def test_unexpected_exception_returns_error_string(self, tmp_path, monkeypatch):
        tool = EditconfTool(work_dir=tmp_path)

        def _crash(*args, **kwargs):
            raise RuntimeError("Unexpected internal error")

        monkeypatch.setattr(GMX_TOOLS_MODULE, _crash)
        result = tool.forward(input_gro=str(tmp_path / "conf.gro"))
        assert "SUCCESS: False" in result
        assert "Unexpected" in result or "RuntimeError" in result

    def test_failure_returncode_reflected(self, tmp_path, monkeypatch):
        tool = EditconfTool(work_dir=tmp_path)
        monkeypatch.setattr(
            GMX_TOOLS_MODULE,
            make_failing_mock(returncode=1, stderr="Fatal error: bad input"),
        )
        result = tool.forward(input_gro=str(tmp_path / "conf.gro"))
        assert "SUCCESS: False" in result
        assert "RETURN_CODE: 1" in result


class TestGMXResultHelpers:

    def test_extract_warnings_finds_warning_lines(self):
        from agent.tools.base import extract_warnings
        text     = "Normal line\nWARNING: something bad\nAnother line"
        warnings = extract_warnings(text)
        assert len(warnings) == 1
        assert "WARNING" in warnings[0]

    def test_extract_errors_finds_error_lines(self):
        from agent.tools.base import extract_errors
        text   = "Normal line\nFatal error: crash\nERROR: bad\nOK line"
        errors = extract_errors(text)
        assert len(errors) == 2

    def test_assert_files_exist_returns_empty_for_existing(self, tmp_path):
        from agent.tools.base import assert_files_exist
        f = tmp_path / "exists.gro"
        f.write_text("content")
        missing = assert_files_exist({"gro": f})
        assert missing == []

    def test_assert_files_exist_returns_message_for_missing(self, tmp_path):
        from agent.tools.base import assert_files_exist
        missing = assert_files_exist({"gro": tmp_path / "missing.gro"})
        assert len(missing) == 1
        assert "missing.gro" in missing[0]

    def test_work_dir_created_on_init(self, tmp_path):
        new_dir = tmp_path / "new" / "nested" / "dir"
        tool    = EditconfTool(work_dir=new_dir)
        assert new_dir.exists()