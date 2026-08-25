"""
4D-7: ReadFileTool, WriteFileTool, ParseGmxLogTool.
ValidateStructureTool is covered by the base tool tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.tools.file_tools import ParseGmxLogTool, ReadFileTool, WriteFileTool


class TestReadFileTool:

    def test_reads_existing_file(self, tmp_path):
        f = tmp_path / "test.gro"
        f.write_text("line1\nline2\nline3\n")
        tool   = ReadFileTool()
        result = tool.forward(file_path=str(f))
        assert "line1" in result
        assert "line2" in result

    def test_returns_error_for_missing_file(self, tmp_path):
        tool   = ReadFileTool()
        result = tool.forward(file_path=str(tmp_path / "missing.gro"))
        assert "ERROR" in result
        assert "not found" in result.lower() or "missing" in result.lower()

    def test_truncates_to_max_lines(self, tmp_path):
        f = tmp_path / "big.log"
        f.write_text("\n".join(f"line {i}" for i in range(500)))
        tool   = ReadFileTool()
        result = tool.forward(file_path=str(f), max_lines=10)
        lines  = [l for l in result.splitlines() if l.startswith("line")]
        assert len(lines) <= 10

    def test_truncation_message_shown(self, tmp_path):
        f = tmp_path / "big.log"
        f.write_text("\n".join(f"line {i}" for i in range(500)))
        tool   = ReadFileTool()
        result = tool.forward(file_path=str(f), max_lines=10)
        assert "truncated" in result.lower()

    def test_no_truncation_message_for_short_file(self, tmp_path):
        f = tmp_path / "short.log"
        f.write_text("line1\nline2\n")
        tool   = ReadFileTool()
        result = tool.forward(file_path=str(f), max_lines=200)
        assert "truncated" not in result.lower()

    def test_reads_mdp_file(self, tmp_mdp_file):
        tool   = ReadFileTool()
        result = tool.forward(file_path=str(tmp_mdp_file))
        assert "integrator" in result
        assert "steep"      in result


class TestWriteFileTool:

    def test_writes_content_to_file(self, tmp_path):
        dest   = tmp_path / "output.mdp"
        tool   = WriteFileTool()
        result = tool.forward(
            file_path=str(dest),
            content="integrator = steep\nnsteps = 5000\n",
        )
        assert "SUCCESS" in result
        assert dest.exists()
        assert "integrator" in dest.read_text()

    def test_creates_parent_directories(self, tmp_path):
        dest   = tmp_path / "a" / "b" / "c" / "test.mdp"
        tool   = WriteFileTool()
        result = tool.forward(file_path=str(dest), content="test content")
        assert dest.exists()
        assert "SUCCESS" in result

    def test_overwrites_existing_file(self, tmp_path):
        dest = tmp_path / "test.mdp"
        dest.write_text("old content")
        tool = WriteFileTool()
        tool.forward(file_path=str(dest), content="new content")
        assert dest.read_text() == "new content"

    def test_reports_character_count(self, tmp_path):
        dest    = tmp_path / "test.mdp"
        content = "integrator = steep\n"
        tool    = WriteFileTool()
        result  = tool.forward(file_path=str(dest), content=content)
        assert str(len(content)) in result

    def test_empty_content_writes_empty_file(self, tmp_path):
        dest   = tmp_path / "empty.mdp"
        tool   = WriteFileTool()
        result = tool.forward(file_path=str(dest), content="")
        assert dest.exists()
        assert dest.read_text() == ""


class TestParseGmxLogTool:

    def test_returns_diagnosis_string_for_converged_em(self, tmp_path):
        log = tmp_path / "em.log"
        log.write_text(
            "integrator              = steep\n"
            "nsteps                  = 5000\n"
            "Steepest Descents converged to Fmax < 1000 in 500 steps\n"
            "Potential Energy  = -4.56789e+05\n"
            "Maximum force     =  9.87654e+02\n"
        )
        tool   = ParseGmxLogTool()
        result = tool.forward(log_file=str(log))
        assert "DIAGNOSIS"         in result
        assert "SUCCESS_CONVERGED" in result
        assert "SEVERITY"          in result
        assert "OK"                in result

    def test_returns_diagnosis_for_not_converged(self, tmp_path):
        log = tmp_path / "em.log"
        log.write_text(
            "integrator              = steep\n"
            "nsteps                  = 1000\n"
            "Steepest Descents did not converge to Fmax < 1000 in 1000 steps\n"
            "Potential Energy  = -2.34567e+05\n"
            "Maximum force     =  1.52340e+04\n"
        )
        tool   = ParseGmxLogTool()
        result = tool.forward(log_file=str(log))
        assert "DIAGNOSIS"    in result
        assert "RECOVERABLE"  in result
        assert "PRIMARY_ACTION" in result
        assert "MDP_PATCHES"  in result

    def test_returns_error_for_missing_log(self, tmp_path):
        tool   = ParseGmxLogTool()
        result = tool.forward(log_file=str(tmp_path / "missing.log"))
        assert "ERROR" in result

    def test_agent_instruction_always_present(self, tmp_path):
        log = tmp_path / "em.log"
        log.write_text(
            "integrator = steep\nnsteps = 5000\n"
            "Steepest Descents converged to Fmax < 1000 in 500 steps\n"
            "Maximum force     =  9.87654e+02\n"
        )
        tool   = ParseGmxLogTool()
        result = tool.forward(log_file=str(log))
        assert "AGENT_INSTRUCTION" in result

    def test_nan_log_returns_recoverable(self, tmp_path):
        log = tmp_path / "em.log"
        log.write_text(
            "integrator = steep\nnsteps = 5000\n"
            "NaN detected in the force on atom 42\n"
        )
        tool   = ParseGmxLogTool()
        result = tool.forward(log_file=str(log))
        assert "NAN_DETECTED"  in result
        assert "RECOVERABLE"   in result

    def test_fatal_error_returns_fatal_severity(self, tmp_path):
        log = tmp_path / "md.log"
        log.write_text(
            "integrator = md\nnsteps = 50000\n"
            "No space left on device\n"
        )
        tool   = ParseGmxLogTool()
        result = tool.forward(log_file=str(log))
        assert "FATAL" in result