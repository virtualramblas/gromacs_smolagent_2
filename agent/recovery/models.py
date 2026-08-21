"""
Dataclasses for the diagnosis and recovery pipeline.
All objects are serialisable to plain strings for LLM consumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enumerations — finite, well-defined vocabularies for the LLM to reason over
# ---------------------------------------------------------------------------

class SimulationPhase(str, Enum):
    """Which stage of the MD pipeline produced this log."""
    UNKNOWN        = "UNKNOWN"
    ENERGY_MIN     = "ENERGY_MIN"
    NVT_EQUIL      = "NVT_EQUIL"
    NPT_EQUIL      = "NPT_EQUIL"
    PRODUCTION_MD  = "PRODUCTION_MD"
    GENION_PREP    = "GENION_PREP"


class DiagnosisCode(str, Enum):
    """
    Exhaustive set of diagnosable GROMACS failure / warning modes.
    Each code maps 1-to-1 with a recovery strategy.
    """
    # ---- Success states ----
    SUCCESS_CONVERGED        = "SUCCESS_CONVERGED"
    SUCCESS_COMPLETED        = "SUCCESS_COMPLETED"

    # ---- Energy minimisation failures ----
    EM_NOT_CONVERGED         = "EM_NOT_CONVERGED"
    EM_STEP_LIMIT_HIT        = "EM_STEP_LIMIT_HIT"
    EM_LINCS_ERROR           = "EM_LINCS_ERROR"
    EM_EXPLODED              = "EM_EXPLODED"          # Epot >> 0 or NaN

    # ---- Equilibration failures ----
    EQUIL_TEMP_UNSTABLE      = "EQUIL_TEMP_UNSTABLE"
    EQUIL_PRESSURE_UNSTABLE  = "EQUIL_PRESSURE_UNSTABLE"
    EQUIL_DRIFT_TOO_HIGH     = "EQUIL_DRIFT_TOO_HIGH"

    # ---- General simulation failures ----
    LINCS_WARNING            = "LINCS_WARNING"
    SETTLE_ERROR             = "SETTLE_ERROR"
    PARTICLE_ESCAPED_BOX     = "PARTICLE_ESCAPED_BOX"
    NAN_DETECTED             = "NAN_DETECTED"
    NEIGHBOUR_LIST_ERROR     = "NEIGHBOUR_LIST_ERROR"

    # ---- Setup / topology failures ----
    MISSING_PARAMETERS       = "MISSING_PARAMETERS"
    CHARGE_IMBALANCE         = "CHARGE_IMBALANCE"
    TOPOLOGY_MISMATCH        = "TOPOLOGY_MISMATCH"

    # ---- Infrastructure failures ----
    GPU_ERROR                = "GPU_ERROR"
    MPI_ERROR                = "MPI_ERROR"
    DISK_FULL                = "DISK_FULL"
    TIMEOUT                  = "TIMEOUT"

    # ---- Ambiguous / needs human ----
    UNKNOWN_ERROR            = "UNKNOWN_ERROR"
    NEEDS_HUMAN_REVIEW       = "NEEDS_HUMAN_REVIEW"


class Severity(str, Enum):
    OK            = "OK"
    RECOVERABLE   = "RECOVERABLE"    # agent can fix automatically
    ASSISTED      = "ASSISTED"       # agent proposes fix, user confirms
    FATAL         = "FATAL"          # cannot continue, escalate to user


class RecoveryAction(str, Enum):
    """
    Discrete recovery actions the LLM can select from.
    Keeping this as a closed enum prevents hallucinated action names.
    """
    NONE                     = "NONE"
    REDUCE_EMSTEP            = "REDUCE_EMSTEP"
    INCREASE_NSTEPS          = "INCREASE_NSTEPS"
    REDUCE_EMSTEP_AND_INCREASE_NSTEPS = "REDUCE_EMSTEP_AND_INCREASE_NSTEPS"
    SWITCH_INTEGRATOR_SD     = "SWITCH_INTEGRATOR_SD"   # steep → sd for EM
    REDUCE_DT                = "REDUCE_DT"
    INCREASE_LINCS_ORDER     = "INCREASE_LINCS_ORDER"
    DISABLE_LINCS            = "DISABLE_LINCS"
    INCREASE_NEIGHBOUR_FREQ  = "INCREASE_NEIGHBOUR_FREQ"
    REBUILD_TOPOLOGY         = "REBUILD_TOPOLOGY"
    RERUN_GENION             = "RERUN_GENION"
    SWITCH_TO_CPU            = "SWITCH_TO_CPU"
    RESUME_FROM_CHECKPOINT   = "RESUME_FROM_CHECKPOINT"
    ESCALATE_TO_USER         = "ESCALATE_TO_USER"


# ---------------------------------------------------------------------------
# Raw metrics extracted from log
# ---------------------------------------------------------------------------

@dataclass
class LogMetrics:
    """Raw numerical and textual data extracted from a GROMACS log file."""
    phase: SimulationPhase = SimulationPhase.UNKNOWN

    # Energy minimisation
    em_converged: Optional[bool]    = None
    em_fmax_final: Optional[float]  = None
    em_fmax_target: Optional[float] = None
    em_steps_taken: Optional[int]   = None
    em_steps_limit: Optional[int]   = None
    epot_final: Optional[float]     = None
    epot_values: list[float]        = field(default_factory=list)

    # Dynamics
    temperature_mean: Optional[float]    = None
    temperature_target: Optional[float]  = None
    temperature_values: list[float]      = field(default_factory=list)
    pressure_mean: Optional[float]       = None
    pressure_values: list[float]         = field(default_factory=list)
    drift_kj_per_ns: Optional[float]     = None

    # Performance
    performance_ns_per_day: Optional[float] = None
    last_step: Optional[int]               = None
    total_steps: Optional[int]             = None

    # Error flags (boolean presence)
    has_nan: bool              = False
    has_lincs_error: bool      = False
    has_lincs_warning: bool    = False
    has_settle_error: bool     = False
    has_particle_escaped: bool = False
    has_gpu_error: bool        = False
    has_mpi_error: bool        = False
    has_disk_full: bool        = False
    has_missing_params: bool   = False
    has_charge_warning: bool   = False
    has_topology_error: bool   = False

    # Raw text collections
    warnings: list[str] = field(default_factory=list)
    errors: list[str]   = field(default_factory=list)
    notes: list[str]    = field(default_factory=list)


# ---------------------------------------------------------------------------
# Diagnosis output
# ---------------------------------------------------------------------------

@dataclass
class Diagnosis:
    """
    Structured interpretation of LogMetrics.
    This is what the DiagnosisEngine produces.
    """
    code: DiagnosisCode
    severity: Severity
    phase: SimulationPhase
    evidence: list[str]          = field(default_factory=list)
    secondary_codes: list[DiagnosisCode] = field(default_factory=list)
    metrics: Optional[LogMetrics] = None


# ---------------------------------------------------------------------------
# Recovery recommendation
# ---------------------------------------------------------------------------

@dataclass
class MDPPatch:
    """A single key-value change to apply to an .mdp file."""
    parameter: str
    old_value: Optional[str]
    new_value: str
    reason: str


@dataclass
class ActionRecommendation:
    """
    Complete, actionable recovery plan produced by RecoveryPlanner.
    This is the final output serialised for the LLM agent.
    """
    diagnosis: Diagnosis
    primary_action: RecoveryAction
    mdp_patches: list[MDPPatch]          = field(default_factory=list)
    rerun_steps: list[str]               = field(default_factory=list)
    agent_instruction: str               = ""
    fallback_action: Optional[RecoveryAction] = None
    fallback_instruction: str            = ""

    def to_agent_string(self) -> str:
        """
        Serialise to a compact, LLM-readable string.
        Designed to be unambiguous and token-efficient.
        """
        d = self.diagnosis
        lines = [
            "=" * 60,
            f"DIAGNOSIS        : {d.code.value}",
            f"SEVERITY         : {d.severity.value}",
            f"PHASE            : {d.phase.value}",
            "-" * 60,
            "EVIDENCE:",
        ]
        for ev in d.evidence:
            lines.append(f"  - {ev}")

        if d.secondary_codes:
            lines.append("SECONDARY_ISSUES:")
            for sc in d.secondary_codes:
                lines.append(f"  - {sc.value}")

        lines += [
            "-" * 60,
            f"PRIMARY_ACTION   : {self.primary_action.value}",
        ]

        if self.mdp_patches:
            lines.append("MDP_PATCHES:")
            for p in self.mdp_patches:
                old = p.old_value if p.old_value else "?"
                lines.append(
                    f"  {p.parameter}: {old} -> {p.new_value}  "
                    f"[reason: {p.reason}]"
                )

        if self.rerun_steps:
            lines.append("RERUN_STEPS:")
            for step in self.rerun_steps:
                lines.append(f"  {step}")

        lines.append(f"AGENT_INSTRUCTION: {self.agent_instruction}")

        if self.fallback_action and self.fallback_action != RecoveryAction.NONE:
            lines += [
                "-" * 60,
                f"FALLBACK_ACTION  : {self.fallback_action.value}",
                f"FALLBACK_INSTR   : {self.fallback_instruction}",
            ]

        lines.append("=" * 60)
        return "\n".join(lines)