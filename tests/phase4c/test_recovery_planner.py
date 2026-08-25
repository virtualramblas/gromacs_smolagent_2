"""
4C-2: Verify RecoveryPlanner produces correct RecoveryAction,
MDPPatches, rerun_steps, and agent_instruction for every DiagnosisCode.

Rationale:
    The planner is the last Python layer before the LLM acts.
    Wrong actions or missing MDP patches cause the agent to
    apply incorrect fixes — potentially making the simulation worse.
"""

from __future__ import annotations

import pytest
from agent.recovery.diagnosis_engine import DiagnosisEngine
from agent.recovery.models import (
    ActionRecommendation,
    Diagnosis,
    DiagnosisCode,
    MDPPatch,
    RecoveryAction,
    Severity,
    SimulationPhase,
)
from agent.recovery.recovery_planner import RecoveryPlanner

from .metrics_factory import (
    em_converged,
    em_exploded,
    em_lincs_error,
    em_nan,
    em_not_converged,
    npt_pressure_unstable,
    nvt_temp_unstable,
    production_complete,
    production_high_drift,
    with_charge_imbalance,
    with_disk_full,
    with_gpu_error,
    with_missing_params,
    with_mpi_error,
    with_particle_escaped,
    with_settle_error,
    with_topology_error,
)


def plan(metrics) -> ActionRecommendation:
    """Convenience: diagnose then plan in one call."""
    diagnosis = DiagnosisEngine.diagnose(metrics)
    return RecoveryPlanner.plan(diagnosis)


def plan_from_code(
    code: DiagnosisCode,
    severity: Severity = Severity.RECOVERABLE,
    phase: SimulationPhase = SimulationPhase.ENERGY_MIN,
) -> ActionRecommendation:
    """Build a Diagnosis directly from a code and plan it."""
    d = Diagnosis(code=code, severity=severity, phase=phase)
    return RecoveryPlanner.plan(d)


# ---------------------------------------------------------------------------
# Success states → NONE action
# ---------------------------------------------------------------------------

class TestSuccessPlanning:

    def test_em_converged_action_is_none(self):
        rec = plan(em_converged())
        assert rec.primary_action == RecoveryAction.NONE

    def test_em_converged_no_mdp_patches(self):
        rec = plan(em_converged())
        assert rec.mdp_patches == []

    def test_em_converged_no_rerun_steps(self):
        rec = plan(em_converged())
        assert rec.rerun_steps == []

    def test_production_complete_action_is_none(self):
        rec = plan(production_complete())
        assert rec.primary_action == RecoveryAction.NONE

    def test_success_instruction_says_proceed(self):
        rec = plan(em_converged())
        assert any(
            word in rec.agent_instruction.lower()
            for word in ("proceed", "next", "success", "complete")
        )


# ---------------------------------------------------------------------------
# EM failures
# ---------------------------------------------------------------------------

class TestEMRecovery:

    def test_em_not_converged_action(self):
        # steps_taken < steps_limit → EM_NOT_CONVERGED, not EM_STEP_LIMIT_HIT
        rec = plan(em_not_converged(
            fmax=15234.5,
            steps_taken=500,    # ← not at limit
            steps_limit=1000,
        ))
        assert rec.primary_action == RecoveryAction.REDUCE_EMSTEP_AND_INCREASE_NSTEPS

    def test_em_not_converged_patches_emstep(self):
        rec = plan(em_not_converged(
            fmax=15234.5,
            steps_taken=500,    # ← not at limit
            steps_limit=1000,
        ))
        params = {p.parameter.lower() for p in rec.mdp_patches}
        assert "emstep" in params

    def test_em_not_converged_patches_nsteps(self):
        rec = plan(em_not_converged(
            fmax=15234.5,
            steps_taken=500,    # ← not at limit
            steps_limit=1000,
        ))
        params = {p.parameter.lower() for p in rec.mdp_patches}
        assert "nsteps" in params

    def test_em_not_converged_has_rerun_steps(self):
        rec = plan(em_not_converged(
            steps_taken=500,    # ← not at limit
            steps_limit=1000,
        ))
        assert len(rec.rerun_steps) >= 2
        rerun_text = " ".join(rec.rerun_steps).lower()
        assert "grompp" in rerun_text
        assert "mdrun"  in rerun_text

    def test_em_not_converged_has_fallback(self):
        rec = plan(em_not_converged(
            steps_taken=500,    # ← not at limit
            steps_limit=1000,
        ))
        assert rec.fallback_action is not None
        assert rec.fallback_action != RecoveryAction.NONE

    @pytest.mark.parametrize("fmax,expected_emstep", [
        (150000.0, 0.0001),
        (15000.0,  0.001),
        (1500.0,   0.005),
    ])
    def test_emstep_scales_with_fmax_severity(self, fmax, expected_emstep):
        m = em_not_converged(
            fmax=fmax,
            fmax_target=1000.0,
            steps_taken=500,    # ← not at limit → EM_NOT_CONVERGED path
            steps_limit=1000,
        )
        rec = plan(m)
        emstep_patch = next(
            (p for p in rec.mdp_patches if p.parameter.lower() == "emstep"),
            None,
        )
        assert emstep_patch is not None, (
            f"No emstep patch found for fmax={fmax}. "
            f"Diagnosis was: {rec.diagnosis.code}. "
            f"Patches: {[p.parameter for p in rec.mdp_patches]}"
        )
        assert float(emstep_patch.new_value) == pytest.approx(
            expected_emstep, rel=1e-3
        )


# ---------------------------------------------------------------------------
# Equilibration failures
# ---------------------------------------------------------------------------

class TestEquilibrationRecovery:

    def test_temp_unstable_has_tau_t_patch(self):
        rec = plan(nvt_temp_unstable())
        params = {p.parameter.lower() for p in rec.mdp_patches}
        assert "tau-t" in params

    def test_temp_unstable_new_tau_t_larger(self):
        """Larger tau-t damps temperature oscillations."""
        rec = plan(nvt_temp_unstable())
        patch = next(
            (p for p in rec.mdp_patches if p.parameter.lower() == "tau-t"),
            None,
        )
        assert patch is not None
        if patch.old_value:
            assert float(patch.new_value) > float(patch.old_value)

    def test_temp_unstable_rerun_includes_nvt(self):
        rec = plan(nvt_temp_unstable())
        rerun_text = " ".join(rec.rerun_steps).lower()
        assert "nvt" in rerun_text

    def test_pressure_unstable_has_tau_p_patch(self):
        rec = plan(npt_pressure_unstable())
        params = {p.parameter.lower() for p in rec.mdp_patches}
        assert "tau-p" in params

    def test_pressure_unstable_new_tau_p_larger(self):
        """Larger tau-p stabilises pressure fluctuations."""
        rec = plan(npt_pressure_unstable())
        patch = next(
            (p for p in rec.mdp_patches if p.parameter.lower() == "tau-p"),
            None,
        )
        assert patch is not None
        if patch.old_value:
            assert float(patch.new_value) > float(patch.old_value)

    def test_pressure_unstable_rerun_includes_npt(self):
        rec = plan(npt_pressure_unstable())
        rerun_text = " ".join(rec.rerun_steps).lower()
        assert "npt" in rerun_text

    def test_high_drift_action_is_increase_neighbour_freq(self):
        rec = plan(production_high_drift())
        assert rec.primary_action == RecoveryAction.INCREASE_NEIGHBOUR_FREQ

    def test_high_drift_patches_nstlist(self):
        rec = plan(production_high_drift())
        params = {p.parameter.lower() for p in rec.mdp_patches}
        assert "nstlist" in params


# ---------------------------------------------------------------------------
# Runtime errors
# ---------------------------------------------------------------------------

class TestRuntimeErrorRecovery:

    def test_settle_error_action_is_reduce_dt(self):
        rec = plan(with_settle_error())
        assert rec.primary_action == RecoveryAction.REDUCE_DT

    def test_settle_error_patches_dt(self):
        rec = plan(with_settle_error())
        params = {p.parameter.lower() for p in rec.mdp_patches}
        assert "dt" in params

    def test_settle_error_fallback_is_escalate(self):
        rec = plan(with_settle_error())
        assert rec.fallback_action == RecoveryAction.ESCALATE_TO_USER

    def test_particle_escaped_action_is_reduce_dt(self):
        rec = plan(with_particle_escaped())
        assert rec.primary_action == RecoveryAction.REDUCE_DT

    def test_particle_escaped_patches_dt_and_nstcomm(self):
        rec = plan(with_particle_escaped())
        params = {p.parameter.lower() for p in rec.mdp_patches}
        assert "dt"      in params
        assert "nstcomm" in params

    def test_particle_escaped_fallback_is_resume_checkpoint(self):
        rec = plan(with_particle_escaped())
        assert rec.fallback_action == RecoveryAction.RESUME_FROM_CHECKPOINT


# ---------------------------------------------------------------------------
# Infrastructure failures
# ---------------------------------------------------------------------------

class TestInfrastructureRecovery:

    def test_gpu_error_action_is_switch_to_cpu(self):
        rec = plan(with_gpu_error())
        assert rec.primary_action == RecoveryAction.SWITCH_TO_CPU

    def test_gpu_error_rerun_mentions_cpu(self):
        rec = plan(with_gpu_error())
        rerun_text = " ".join(rec.rerun_steps).lower()
        assert "cpu" in rerun_text

    def test_disk_full_action_is_escalate(self):
        rec = plan(with_disk_full())
        assert rec.primary_action == RecoveryAction.ESCALATE_TO_USER

    def test_mpi_error_action_is_escalate(self):
        rec = plan(with_mpi_error())
        assert rec.primary_action == RecoveryAction.ESCALATE_TO_USER


# ---------------------------------------------------------------------------
# Setup / topology failures
# ---------------------------------------------------------------------------

class TestSetupRecovery:

    def test_missing_params_action_is_rebuild_topology(self):
        rec = plan(with_missing_params())
        assert rec.primary_action == RecoveryAction.REBUILD_TOPOLOGY

    def test_missing_params_fallback_is_escalate(self):
        rec = plan(with_missing_params())
        assert rec.fallback_action == RecoveryAction.ESCALATE_TO_USER

    def test_charge_imbalance_action_is_rerun_genion(self):
        rec = plan(with_charge_imbalance())
        assert rec.primary_action == RecoveryAction.RERUN_GENION

    def test_charge_imbalance_rerun_includes_genion(self):
        rec = plan(with_charge_imbalance())
        rerun_text = " ".join(rec.rerun_steps).lower()
        assert "genion" in rerun_text

    def test_topology_error_action_is_escalate(self):
        rec = plan(with_topology_error())
        assert rec.primary_action == RecoveryAction.ESCALATE_TO_USER


# ---------------------------------------------------------------------------
# ActionRecommendation contract tests
# ---------------------------------------------------------------------------

class TestRecommendationContract:

    def test_every_recommendation_has_agent_instruction(self):
        """
        Every possible DiagnosisCode must produce a non-empty
        agent_instruction — the LLM must always know what to do.
        """
        all_metrics = [
            em_converged(), em_not_converged(), em_exploded(),
            em_nan(), em_lincs_error(), nvt_temp_unstable(),
            npt_pressure_unstable(), production_high_drift(),
            with_settle_error(), with_particle_escaped(),
            with_gpu_error(), with_disk_full(), with_mpi_error(),
            with_missing_params(), with_charge_imbalance(),
            with_topology_error(),
        ]
        for m in all_metrics:
            rec = plan(m)
            assert rec.agent_instruction.strip() != "", (
                f"Empty agent_instruction for {rec.diagnosis.code}"
            )

    def test_every_patch_has_reason(self):
        """Every MDPPatch must have a non-empty reason string."""
        all_metrics = [
            em_not_converged(), em_exploded(), em_nan(),
            em_lincs_error(), nvt_temp_unstable(),
            npt_pressure_unstable(), production_high_drift(),
            with_settle_error(), with_particle_escaped(),
        ]
        for m in all_metrics:
            rec = plan(m)
            for patch in rec.mdp_patches:
                assert patch.reason.strip() != "", (
                    f"Empty reason for patch '{patch.parameter}' "
                    f"in {rec.diagnosis.code}"
                )

    def test_every_patch_new_value_is_valid_number_or_string(self):
        """new_value must be a non-empty string parseable by GROMACS."""
        all_metrics = [
            em_not_converged(), em_exploded(), em_nan(),
            em_lincs_error(), nvt_temp_unstable(),
            npt_pressure_unstable(), production_high_drift(),
        ]
        for m in all_metrics:
            rec = plan(m)
            for patch in rec.mdp_patches:
                assert patch.new_value.strip() != "", (
                    f"Empty new_value for patch '{patch.parameter}'"
                )

    def test_fatal_diagnoses_have_no_mdp_patches(self):
        """
        Fatal errors cannot be fixed by MDP changes.
        Patches for fatal diagnoses would mislead the agent.
        """
        fatal_metrics = [with_disk_full(), with_mpi_error(), with_topology_error()]
        for m in fatal_metrics:
            rec = plan(m)
            assert rec.mdp_patches == [], (
                f"Unexpected MDP patches for fatal diagnosis {rec.diagnosis.code}"
            )

    def test_to_agent_string_always_parseable(self):
        """
        to_agent_string() must never raise and must always contain
        the minimum fields the LLM needs.
        """
        all_metrics = [
            em_converged(), em_not_converged(), em_exploded(),
            em_nan(), em_lincs_error(), nvt_temp_unstable(),
            npt_pressure_unstable(), production_complete(),
            production_high_drift(), with_settle_error(),
            with_particle_escaped(), with_gpu_error(),
            with_disk_full(), with_mpi_error(), with_missing_params(),
            with_charge_imbalance(), with_topology_error(),
        ]
        required_fields = [
            "DIAGNOSIS", "SEVERITY", "PHASE",
            "PRIMARY_ACTION", "AGENT_INSTRUCTION",
        ]
        for m in all_metrics:
            rec  = plan(m)
            text = rec.to_agent_string()
            for field in required_fields:
                assert field in text, (
                    f"Field '{field}' missing from to_agent_string() "
                    f"for {rec.diagnosis.code}"
                )