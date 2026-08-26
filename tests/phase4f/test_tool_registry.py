"""
4F-1: Verify get_all_tools() returns a correctly configured
tool registry that passes SmolAgent's validation.

Rationale:
    If any tool fails validate_arguments(), build_agent() crashes
    before the agent can run. These tests catch registration bugs
    without needing a real LLM.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.tools import get_all_tools
from agent.tools.gmx_tools import (
    EditconfTool,
    GenionTool,
    GromppTool,
    MdrunTool,
    Pdb2GmxTool,
    SolvateTool,
)
from agent.tools.file_tools import (
    ParseGmxLogTool,
    ReadFileTool,
    ValidateStructureTool,
    WriteFileTool,
)
from agent.tools.state_tools import PipelineStateTool
from agent.tools.analysis_tools import EnergyAnalysisTool, RMSDAnalysisTool


class TestToolRegistry:

    def test_get_all_tools_returns_list(self, tmp_path):
        tools = get_all_tools(work_dir=str(tmp_path))
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_all_expected_tools_present(self, tmp_path):
        tools     = get_all_tools(work_dir=str(tmp_path))
        tool_names = {t.name for t in tools}
        expected  = {
            "pdb2gmx", "editconf", "solvate", "genion",
            "grompp", "mdrun",
            "read_file", "write_file", "validate_structure", "parse_gmx_log",
            "pipeline_state",
            "energy_analysis", "rmsd_analysis",
        }
        missing = expected - tool_names
        assert not missing, f"Missing tools: {missing}"

    def test_no_duplicate_tool_names(self, tmp_path):
        tools      = get_all_tools(work_dir=str(tmp_path))
        tool_names = [t.name for t in tools]
        assert len(tool_names) == len(set(tool_names)), (
            "Duplicate tool names found in registry"
        )

    def test_all_tools_have_non_empty_description(self, tmp_path):
        tools = get_all_tools(work_dir=str(tmp_path))
        for tool in tools:
            assert tool.description.strip() != "", (
                f"Tool '{tool.name}' has empty description"
            )

    def test_all_tools_have_string_output_type(self, tmp_path):
        tools = get_all_tools(work_dir=str(tmp_path))
        for tool in tools:
            assert tool.output_type == "string", (
                f"Tool '{tool.name}' output_type is '{tool.output_type}', "
                "expected 'string'"
            )

    def test_all_tools_pass_smolagent_validation(self, tmp_path):
        """
        SmolAgent validates tools at instantiation time.
        If this test passes, all tools are correctly configured.
        """
        # get_all_tools() calls Tool.__init__() which calls validate_arguments()
        # If any tool is misconfigured, this raises an Exception
        try:
            tools = get_all_tools(work_dir=str(tmp_path))
            assert len(tools) > 0
        except Exception as exc:
            pytest.fail(
                f"Tool registry validation failed: {exc}"
            )

    def test_gmx_tools_work_dir_set_correctly(self, tmp_path):
        tools    = get_all_tools(work_dir=str(tmp_path))
        gmx_tool = next(t for t in tools if t.name == "pdb2gmx")
        assert gmx_tool.work_dir == tmp_path.resolve()

    def test_state_tool_state_file_in_work_dir(self, tmp_path):
        tools      = get_all_tools(work_dir=str(tmp_path))
        state_tool = next(t for t in tools if t.name == "pipeline_state")
        assert str(tmp_path) in str(state_tool.state_file)

    @pytest.mark.parametrize("tool_class,expected_name", [
        (Pdb2GmxTool,          "pdb2gmx"),
        (EditconfTool,         "editconf"),
        (SolvateTool,          "solvate"),
        (GenionTool,           "genion"),
        (GromppTool,           "grompp"),
        (MdrunTool,            "mdrun"),
        (ReadFileTool,         "read_file"),
        (WriteFileTool,        "write_file"),
        (ValidateStructureTool,"validate_structure"),
        (ParseGmxLogTool,      "parse_gmx_log"),
        (PipelineStateTool,    "pipeline_state"),
        (EnergyAnalysisTool,   "energy_analysis"),
        (RMSDAnalysisTool,     "rmsd_analysis"),
    ])
    def test_tool_name_matches_class_attribute(
        self, tmp_path, tool_class, expected_name
    ):
        if hasattr(tool_class.__init__, "__code__") and \
           "work_dir" in tool_class.__init__.__code__.co_varnames:
            tool = tool_class(work_dir=tmp_path)
        else:
            tool = tool_class()
        assert tool.name == expected_name