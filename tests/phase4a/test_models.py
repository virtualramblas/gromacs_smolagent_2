"""
4A-2: Verify dataclass construction, defaults, and field types
for LogMetrics, Diagnosis, MDPPatch, and ActionRecommendation.

Rationale:
    These objects flow through the entire recovery pipeline.
    Wrong defaults or missing fields cause silent data corruption
    that is hard to trace back to the model layer.
"""

from __future__ import annotations

import pytest
from agent.recovery.models import (
    ActionRecommendation,
    Diagnosis,
    DiagnosisCode,
    LogMetrics,
    MDPPatch,
    RecoveryAction,
    Severity,
    SimulationPhase,
)


# ---------------------------------------------------------------------------
# LogMetrics
# ---------------------------------------------------------------------------

class TestLogMetrics:

    def test_default_construction(self):
        m = LogMetrics()
        assert m.phase          == SimulationPhase.UNKNOWN
        assert m.em_converged   is None
        assert m.em_fmax_final  is None
        assert m.epot_final     is None
        assert m.has_nan        is False
        assert m.has_lincs_error is False
        assert m.warnings       == []
        assert m.errors         == []

    def test_list_fields_are_independent_instances(self):
        """
        Mutable default fields must not be shared between instances.
        A common Python dataclass pitfall with list defaults.
        """
        m1 = LogMetrics()
        m2 = LogMetrics()
        m1.warnings.append("test warning")
        assert m2.warnings == [], (
            "LogMetrics list fields share the same instance — "
            "use field(default_factory=list)"
        )

    def test_field_assignment(self):
        m = LogMetrics()
        m.phase          = SimulationPhase.ENERGY_MIN
        m.em_converged   = False
        m.em_fmax_final  = 15234.5
        m.em_fmax_target = 1000.0
        m.epot_final     = -234567.8
        m.has_nan        = True
        m.warnings       = ["WARNING: something"]

        assert m.phase          == SimulationPhase.ENERGY_MIN
        assert m.em_converged   is False
        assert m.em_fmax_final  == pytest.approx(15234.5)
        assert m.em_fmax_target == pytest.approx(1000.0)
        assert m.epot_final     == pytest.approx(-234567.8)
        assert m.has_nan        is True
        assert len(m.warnings)  == 1

    def test_boolean_flags_default_false(self):
        m = LogMetrics()
        bool_flags = [
            "has_nan",
            "has_lincs_error",
            "has_lincs_warning",
            "has_settle_error",
            "has_particle_escaped",
            "has_gpu_error",
            "has_mpi_error",
            "has_disk_full",
            "has_missing_params",
            "has_charge_warning",
            "has_topology_error",
        ]
        for flag in bool_flags:
            assert getattr(m, flag) is False, (
                f"Boolean flag '{flag}' should default to False"
            )

    def test_numeric_fields_default_none(self):
        m = LogMetrics()
        none_fields = [
            "em_converged", "em_fmax_final", "em_fmax_target",
            "em_steps_taken", "em_steps_limit", "epot_final",
            "temperature_mean", "temperature_target",
            "pressure_mean", "drift_kj_per_ns",
            "performance_ns_per_day", "last_step", "total_steps",
        ]
        for field_name in none_fields:
            assert getattr(m, field_name) is None, (
                f"Field '{field_name}' should default to None"
            )


# ---------------------------------------------------------------------------
# MDPPatch
# ---------------------------------------------------------------------------

class TestMDPPatch:

    def test_construction_with_old_value(self):
        patch = MDPPatch(
            parameter="emstep",
            old_value="0.01",
            new_value="0.001",
            reason="Reduce step size for convergence.",
        )
        assert patch.parameter == "emstep"
        assert patch.old_value == "0.01"
        assert patch.new_value == "0.001"
        assert "convergence" in patch.reason

    def test_construction_without_old_value(self):
        patch = MDPPatch(
            parameter="nsteps",
            old_value=None,
            new_value="10000",
            reason="Increase step budget.",
        )
        assert patch.old_value is None
        assert patch.new_value == "10000"

    def test_parameter_is_string(self):
        patch = MDPPatch(
            parameter="lincs-order",
            old_value="4",
            new_value="6",
            reason="test",
        )
        assert isinstance(patch.parameter, str)
        assert isinstance(patch.new_value, str)


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------

class TestDiagnosis:

    def test_minimal_construction(self):
        d = Diagnosis(
            code=DiagnosisCode.EM_NOT_CONVERGED,
            severity=Severity.RECOVERABLE,
            phase=SimulationPhase.ENERGY_MIN,
        )
        assert d.code     == DiagnosisCode.EM_NOT_CONVERGED
        assert d.severity == Severity.RECOVERABLE
        assert d.phase    == SimulationPhase.ENERGY_MIN
        assert d.evidence == []
        assert d.secondary_codes == []
        assert d.metrics  is None

    def test_with_evidence(self):
        d = Diagnosis(
            code=DiagnosisCode.EM_NOT_CONVERGED,
            severity=Severity.RECOVERABLE,
            phase=SimulationPhase.ENERGY_MIN,
            evidence=[
                "Fmax = 15234.5 kJ/mol/nm",
                "Target: 1000.0 kJ/mol/nm",
            ],
        )
        assert len(d.evidence) == 2
        assert "Fmax" in d.evidence[0]

    def test_evidence_lists_are_independent(self):
        d1 = Diagnosis(
            code=DiagnosisCode.SUCCESS_CONVERGED,
            severity=Severity.OK,
            phase=SimulationPhase.ENERGY_MIN,
        )
        d2 = Diagnosis(
            code=DiagnosisCode.SUCCESS_CONVERGED,
            severity=Severity.OK,
            phase=SimulationPhase.ENERGY_MIN,
        )
        d1.evidence.append("test")
        assert d2.evidence == []

    def test_with_metrics(self):
        m = LogMetrics()
        m.em_fmax_final = 500.0
        d = Diagnosis(
            code=DiagnosisCode.SUCCESS_CONVERGED,
            severity=Severity.OK,
            phase=SimulationPhase.ENERGY_MIN,
            metrics=m,
        )
        assert d.metrics is m
        assert d.metrics.em_fmax_final == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# ActionRecommendation
# ---------------------------------------------------------------------------

class TestActionRecommendation:

    def _make_diagnosis(
        self,
        code=DiagnosisCode.EM_NOT_CONVERGED,
        severity=Severity.RECOVERABLE,
        phase=SimulationPhase.ENERGY_MIN,
    ) -> Diagnosis:
        return Diagnosis(code=code, severity=severity, phase=phase)

    def test_minimal_construction(self):
        rec = ActionRecommendation(
            diagnosis=self._make_diagnosis(),
            primary_action=RecoveryAction.REDUCE_EMSTEP,
        )
        assert rec.primary_action  == RecoveryAction.REDUCE_EMSTEP
        assert rec.mdp_patches     == []
        assert rec.rerun_steps     == []
        assert rec.agent_instruction == ""
        assert rec.fallback_action is None

    def test_with_patches(self):
        patches = [
            MDPPatch("emstep", "0.01", "0.001", "reduce step"),
            MDPPatch("nsteps", "1000", "5000",  "more steps"),
        ]
        rec = ActionRecommendation(
            diagnosis=self._make_diagnosis(),
            primary_action=RecoveryAction.REDUCE_EMSTEP_AND_INCREASE_NSTEPS,
            mdp_patches=patches,
            rerun_steps=["grompp (EM)", "mdrun (EM)"],
            agent_instruction="Apply patches and re-run EM.",
        )
        assert len(rec.mdp_patches) == 2
        assert rec.mdp_patches[0].parameter == "emstep"
        assert len(rec.rerun_steps) == 2

    def test_to_agent_string_contains_required_fields(self):
        """
        The serialised string must contain all fields the LLM
        needs to make a decision — verified by keyword presence.
        """
        patches = [
            MDPPatch("emstep", "0.01", "0.001", "reduce step size"),
        ]
        rec = ActionRecommendation(
            diagnosis=Diagnosis(
                code=DiagnosisCode.EM_NOT_CONVERGED,
                severity=Severity.RECOVERABLE,
                phase=SimulationPhase.ENERGY_MIN,
                evidence=["Fmax = 15234.5 kJ/mol/nm"],
            ),
            primary_action=RecoveryAction.REDUCE_EMSTEP,
            mdp_patches=patches,
            rerun_steps=["grompp (EM)", "mdrun (EM)"],
            agent_instruction="Apply MDP patches and re-run EM.",
            fallback_action=RecoveryAction.SWITCH_INTEGRATOR_SD,
            fallback_instruction="Switch to sd integrator.",
        )
        s = rec.to_agent_string()

        # All required LLM-readable fields must be present
        assert "DIAGNOSIS"          in s
        assert "EM_NOT_CONVERGED"   in s
        assert "SEVERITY"           in s
        assert "RECOVERABLE"        in s
        assert "PHASE"              in s
        assert "ENERGY_MIN"         in s
        assert "EVIDENCE"           in s
        assert "Fmax"               in s
        assert "PRIMARY_ACTION"     in s
        assert "REDUCE_EMSTEP"      in s
        assert "MDP_PATCHES"        in s
        assert "emstep"             in s
        assert "0.01"               in s
        assert "0.001"              in s
        assert "RERUN_STEPS"        in s
        assert "grompp"             in s
        assert "AGENT_INSTRUCTION"  in s
        assert "FALLBACK_ACTION"    in s
        assert "SWITCH_INTEGRATOR_SD" in s

    def test_to_agent_string_success_has_no_patches(self):
        rec = ActionRecommendation(
            diagnosis=Diagnosis(
                code=DiagnosisCode.SUCCESS_CONVERGED,
                severity=Severity.OK,
                phase=SimulationPhase.ENERGY_MIN,
            ),
            primary_action=RecoveryAction.NONE,
            agent_instruction="Proceed to next step.",
        )
        s = rec.to_agent_string()
        assert "SUCCESS_CONVERGED" in s
        assert "NONE"              in s
        assert "MDP_PATCHES"       not in s   # no patches for success

    def test_to_agent_string_no_fallback_section_when_none(self):
        rec = ActionRecommendation(
            diagnosis=self._make_diagnosis(
                code=DiagnosisCode.SUCCESS_CONVERGED,
                severity=Severity.OK,
            ),
            primary_action=RecoveryAction.NONE,
        )
        s = rec.to_agent_string()
        assert "FALLBACK_ACTION" not in s