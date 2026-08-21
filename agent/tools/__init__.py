"""
GROMACS Agent Tool Registry.
Import all tools from here for clean agent instantiation.
"""

from .gmx_tools import (
    Pdb2GmxTool,
    EditconfTool,
    SolvateTool,
    GenionTool,
    GromppTool,
    MdrunTool,
)
from .file_tools import (
    ReadFileTool,
    WriteFileTool,
    ValidateStructureTool,
    ParseGmxLogTool,
)
from .state_tools import PipelineStateTool
from .analysis_tools import (
    EnergyAnalysisTool,
    RMSDAnalysisTool,
)


def get_all_tools(work_dir: str = ".") -> list:
    """
    Instantiate and return all tools configured for a given work directory.
    Pass this list directly to CodeAgent(tools=...).
    """
    return [
        # Pipeline tools
        Pdb2GmxTool(work_dir=work_dir),
        EditconfTool(work_dir=work_dir),
        SolvateTool(work_dir=work_dir),
        GenionTool(work_dir=work_dir),
        GromppTool(work_dir=work_dir),
        MdrunTool(work_dir=work_dir),
        # File & validation tools
        ReadFileTool(),
        WriteFileTool(),
        ValidateStructureTool(work_dir=work_dir),
        ParseGmxLogTool(),
        # State management
        PipelineStateTool(state_file=f"{work_dir}/pipeline_state.json"),
        # Analysis
        EnergyAnalysisTool(work_dir=work_dir),
        RMSDAnalysisTool(work_dir=work_dir),
    ]