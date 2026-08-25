"""
4A-3: Verify GMXResult construction and to_agent_string() serialisation.

Rationale:
    GMXResult is the return type of every tool call.
    The agent reads its serialised form to decide next steps.
    Malformed serialisation = silent agent reasoning failures.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from agent.tools.base import GMXResult


class TestGMXResult:

    def test_success_result_construction(self):
        r = GMXResult(
            success=True,
            command="gmx pdb2gmx -f eiwit.pdb",
            returncode=0,
            stdout="Normal termination",
            stderr="",
            output_files={"gro": Path("conf.gro"), "top": Path("topol.top")},
            summary="pdb2gmx succeeded.",
        )
        assert r.success     is True
        assert r.returncode  == 0
        assert r.warnings    == []
        assert r.errors      == []

    def test_failure_result_construction(self):
        r = GMXResult(
            success=False,
            command="gmx mdrun -s topol.tpr",
            returncode=1,
            stdout="",
            stderr="Fatal error: LINCS ERROR",
            errors=["Fatal error: LINCS ERROR"],
            summary="mdrun FAILED.",
        )
        assert r.success    is False
        assert r.returncode == 1
        assert len(r.errors) == 1

    def test_to_agent_string_success(self):
        r = GMXResult(
            success=True,
            command="gmx editconf -f conf.gro -o conf_box.gro",
            returncode=0,
            stdout="",
            stderr="",
            output_files={"gro": Path("conf_box.gro")},
            summary="editconf succeeded.",
        )
        s = r.to_agent_string()
        assert "SUCCESS: True"   in s
        assert "RETURN_CODE: 0"  in s
        assert "conf_box.gro"    in s
        assert "SUMMARY:"        in s
        assert "editconf succeeded" in s

    def test_to_agent_string_failure_with_errors(self):
        r = GMXResult(
            success=False,
            command="gmx grompp -f em.mdp",
            returncode=1,
            stdout="",
            stderr="Fatal error: missing parameter\nERROR: topology",
            errors=["Fatal error: missing parameter", "ERROR: topology"],
            summary="grompp FAILED.",
        )
        s = r.to_agent_string()
        assert "SUCCESS: False"  in s
        assert "RETURN_CODE: 1"  in s
        assert "ERRORS:"         in s
        assert "missing parameter" in s

    def test_to_agent_string_with_warnings(self):
        r = GMXResult(
            success=True,
            command="gmx grompp -f nvt.mdp",
            returncode=0,
            stdout="",
            stderr="WARNING: 1-4 interaction not set",
            warnings=["WARNING: 1-4 interaction not set"],
            summary="grompp succeeded with warnings.",
        )
        s = r.to_agent_string()
        assert "WARNINGS:"                    in s
        assert "1-4 interaction not set"      in s

    def test_to_agent_string_output_files_show_existence(self, tmp_path):
        """
        Files that exist should show exists=True,
        files that don't exist should show exists=False.
        """
        existing = tmp_path / "conf.gro"
        existing.write_text("GROMACS GRO file")
        missing  = tmp_path / "topol.top"   # not created

        r = GMXResult(
            success=True,
            command="gmx pdb2gmx",
            returncode=0,
            stdout="",
            stderr="",
            output_files={
                "gro": existing,
                "top": missing,
            },
            summary="test",
        )
        s = r.to_agent_string()
        assert "exists=True"  in s
        assert "exists=False" in s

    def test_stderr_tail_truncated_to_40_lines(self):
        """Long stderr should be truncated to last 40 lines."""
        long_stderr = "\n".join(f"line {i}" for i in range(100))
        r = GMXResult(
            success=False,
            command="gmx mdrun",
            returncode=1,
            stdout="",
            stderr=long_stderr,
            summary="failed",
        )
        s = r.to_agent_string()
        # Last line should be present
        assert "line 99" in s
        # First line should NOT be present (truncated)
        assert "line 0\n" not in s

    def test_list_fields_independent_between_instances(self):
        r1 = GMXResult(
            success=True, command="", returncode=0,
            stdout="", stderr="", summary=""
        )
        r2 = GMXResult(
            success=True, command="", returncode=0,
            stdout="", stderr="", summary=""
        )
        r1.warnings.append("w1")
        assert r2.warnings == []