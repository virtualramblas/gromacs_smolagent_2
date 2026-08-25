"""
4C-1: Verify DiagnosisEngine rule tree produces correct
DiagnosisCode and Severity for every diagnosable state.

Rationale:
    The rule tree has strict priority ordering.
    Tests verify:
        (a) correct code for each failure mode
        (b) correct severity for each code
        (c) priority ordering — higher-priority rules fire first
        (d) secondary codes are recorded correctly
        (e) edge cases at threshold boundaries
"""

from __future__ import annotations

import pytest
from agent.recovery.diagnosis_engine import (
    DiagnosisEngine,
    _EPOT_EXPLOSION_THRESHOLD,
    _FMAX_ACCEPTABLE,
    _TEMP_TOLERANCE_K,
    _PRES_TOLERANCE_BAR,
    _DRIFT_WARN_THRESHOLD,
)
from agent.recovery.models import (
    DiagnosisCode,
    LogMetrics,
    Severity,
    SimulationPhase,
)

from .metrics_factory import (
    em_converged,
    em_exploded,
    em_lincs_error,
    em_lincs_warning,
    em_nan,
    em_not_converged,
    npt_pressure_unstable,
    npt_stable,
    nvt_stable,
    nvt_temp_unstable,
    production_complete,
    production_high_drift,
    unknown_phase_no_errors,
    with_charge_imbalance,
    with_disk_full,
    with_gpu_error,
    with_missing_params,
    with_mpi_error,
    with_particle_escaped,
    with_settle_error,
    with_topology_error,
)


def diagnose(metrics: LogMetrics):
    return DiagnosisEngine.diagnose(metrics)


# ---------------------------------------------------------------------------
# Priority 1: Infrastructure failures
# ---------------------------------------------------------------------------

class TestInfrastructurePriority:

    def test_disk_full_is_fatal(self):
        d = diagnose(with_disk_full())
        assert d.code     == DiagnosisCode.DISK_FULL
        assert d.severity == Severity.FATAL

    def test_mpi_error_is_fatal(self):
        d = diagnose(with_mpi_error())
        assert d.code     == DiagnosisCode.MPI_ERROR
        assert d.severity == Severity.FATAL

    def test_gpu_error_is_recoverable(self):
        d = diagnose(with_gpu_error())
        assert d.code     == DiagnosisCode.GPU_ERROR
        assert d.severity == Severity.RECOVERABLE

    def test_disk_full_takes_priority_over_nan(self):
        """Disk full (P1) must fire before NaN (P2)."""
        m = with_disk_full()
        m.has_nan = True
        d = diagnose(m)
        assert d.code == DiagnosisCode.DISK_FULL

    def test_mpi_error_takes_priority_over_lincs(self):
        """MPI error (P1) must fire before LINCS error (P4)."""
        m = with_mpi_error()
        m.has_lincs_error = True
        d = diagnose(m)
        assert d.code == DiagnosisCode.MPI_ERROR

    def test_disk_full_phase_preserved(self):
        m = with_disk_full()
        m.phase = SimulationPhase.PRODUCTION_MD
        d = diagnose(m)
        assert d.phase == SimulationPhase.PRODUCTION_MD


# ---------------------------------------------------------------------------
# Priority 2: NaN / explosion
# ---------------------------------------------------------------------------

class TestNaNAndExplosion:

    def test_nan_detected_is_recoverable(self):
        d = diagnose(em_nan())
        assert d.code     == DiagnosisCode.NAN_DETECTED
        assert d.severity == Severity.RECOVERABLE

    def test_explosion_detected_is_recoverable(self):
        d = diagnose(em_exploded(epot=2.5e7))
        assert d.code     == DiagnosisCode.EM_EXPLODED
        assert d.severity == Severity.RECOVERABLE

    def test_nan_takes_priority_over_lincs(self):
        """NaN (P2) must fire before LINCS error (P4)."""
        m = em_nan()
        m.has_lincs_error = True
        d = diagnose(m)
        assert d.code == DiagnosisCode.NAN_DETECTED

    def test_explosion_threshold_boundary_above(self):
        """Epot just above threshold → EM_EXPLODED."""
        m = LogMetrics()
        m.phase      = SimulationPhase.ENERGY_MIN
        m.epot_final = _EPOT_EXPLOSION_THRESHOLD + 1.0
        d = diagnose(m)
        assert d.code == DiagnosisCode.EM_EXPLODED

    def test_explosion_threshold_boundary_below(self):
        """Epot just below threshold → NOT EM_EXPLODED."""
        m = LogMetrics()
        m.phase      = SimulationPhase.ENERGY_MIN
        m.epot_final = _EPOT_EXPLOSION_THRESHOLD - 1.0
        d = diagnose(m)
        assert d.code != DiagnosisCode.EM_EXPLODED

    def test_negative_epot_not_explosion(self):
        """Negative Epot (normal minimised system) must not trigger explosion."""
        m = em_converged(epot=-456789.0)
        d = diagnose(m)
        assert d.code != DiagnosisCode.EM_EXPLODED

    def test_nan_evidence_contains_epot_when_available(self):
        m = em_nan()
        m.epot_final = 1.23e6
        d = diagnose(m)
        assert any("Epot" in e or "1.23" in e for e in d.evidence)


# ---------------------------------------------------------------------------
# Priority 3: Topology / parameter errors
# ---------------------------------------------------------------------------

class TestTopologyPriority:

    def test_topology_error_is_fatal(self):
        d = diagnose(with_topology_error())
        assert d.code     == DiagnosisCode.TOPOLOGY_MISMATCH
        assert d.severity == Severity.FATAL

    def test_missing_params_is_assisted(self):
        d = diagnose(with_missing_params())
        assert d.code     == DiagnosisCode.MISSING_PARAMETERS
        assert d.severity == Severity.ASSISTED

    def test_charge_imbalance_is_assisted(self):
        d = diagnose(with_charge_imbalance())
        assert d.code     == DiagnosisCode.CHARGE_IMBALANCE
        assert d.severity == Severity.ASSISTED

    def test_topology_error_takes_priority_over_missing_params(self):
        m = with_topology_error()
        m.has_missing_params = True
        d = diagnose(m)
        assert d.code == DiagnosisCode.TOPOLOGY_MISMATCH

    def test_missing_params_takes_priority_over_charge(self):
        m = with_missing_params()
        m.has_charge_warning = True
        d = diagnose(m)
        assert d.code == DiagnosisCode.MISSING_PARAMETERS


# ---------------------------------------------------------------------------
# Priority 4: LINCS / SETTLE
# ---------------------------------------------------------------------------

class TestLincsSettlePriority:

    def test_lincs_error_is_recoverable(self):
        d = diagnose(em_lincs_error())
        assert d.code     == DiagnosisCode.EM_LINCS_ERROR
        assert d.severity == Severity.RECOVERABLE

    def test_lincs_error_evidence_contains_step(self):
        d = diagnose(em_lincs_error(last_step=250))
        assert any("250" in e for e in d.evidence)

    def test_settle_error_is_recoverable(self):
        d = diagnose(with_settle_error())
        assert d.code     == DiagnosisCode.SETTLE_ERROR
        assert d.severity == Severity.RECOVERABLE

    def test_lincs_error_takes_priority_over_settle(self):
        m = with_settle_error()
        m.has_lincs_error = True
        d = diagnose(m)
        assert d.code == DiagnosisCode.EM_LINCS_ERROR

    def test_lincs_warning_recorded_as_secondary(self):
        """
        LINCS warning alone should not be the primary code
        when a more severe issue is present.
        """
        m = em_not_converged()
        m.has_lincs_warning = True
        d = diagnose(m)
        # Primary should be EM failure, warning in secondary
        assert d.code in (
            DiagnosisCode.EM_NOT_CONVERGED,
            DiagnosisCode.EM_STEP_LIMIT_HIT,
        )
        assert DiagnosisCode.LINCS_WARNING in d.secondary_codes


# ---------------------------------------------------------------------------
# Priority 5: Particle escaped
# ---------------------------------------------------------------------------

class TestParticleEscaped:

    def test_particle_escaped_is_recoverable(self):
        d = diagnose(with_particle_escaped())
        assert d.code     == DiagnosisCode.PARTICLE_ESCAPED_BOX
        assert d.severity == Severity.RECOVERABLE

    def test_particle_escaped_takes_priority_over_em_convergence(self):
        m = em_not_converged()
        m.has_particle_escaped = True
        d = diagnose(m)
        assert d.code == DiagnosisCode.PARTICLE_ESCAPED_BOX


# ---------------------------------------------------------------------------
# Priority 6: EM convergence
# ---------------------------------------------------------------------------

class TestEMDiagnosis:

    def test_em_converged_is_ok(self):
        d = diagnose(em_converged())
        assert d.code     == DiagnosisCode.SUCCESS_CONVERGED
        assert d.severity == Severity.OK

    def test_em_not_converged_is_recoverable(self):
        d = diagnose(em_not_converged(fmax=15234.5))
        assert d.code     in (
            DiagnosisCode.EM_NOT_CONVERGED,
            DiagnosisCode.EM_STEP_LIMIT_HIT,
        )
        assert d.severity == Severity.RECOVERABLE

    def test_em_step_limit_hit_when_steps_exhausted(self):
        """When steps_taken >= steps_limit → EM_STEP_LIMIT_HIT."""
        d = diagnose(em_not_converged(
            steps_taken=1000, steps_limit=1000
        ))
        assert d.code == DiagnosisCode.EM_STEP_LIMIT_HIT

    def test_em_not_converged_when_steps_not_exhausted(self):
        """When steps_taken < steps_limit → EM_NOT_CONVERGED."""
        d = diagnose(em_not_converged(
            steps_taken=500, steps_limit=1000
        ))
        assert d.code == DiagnosisCode.EM_NOT_CONVERGED

    def test_em_converged_inferred_from_fmax_below_target(self):
        """No explicit convergence flag but Fmax < target → SUCCESS_CONVERGED."""
        m = LogMetrics()
        m.phase          = SimulationPhase.ENERGY_MIN
        m.em_converged   = None        # no explicit flag
        m.em_fmax_final  = 800.0
        m.em_fmax_target = 1000.0
        d = diagnose(m)
        assert d.code == DiagnosisCode.SUCCESS_CONVERGED

    def test_em_not_converged_inferred_from_fmax_above_target(self):
        """No explicit convergence flag but Fmax > target → EM_NOT_CONVERGED."""
        m = LogMetrics()
        m.phase          = SimulationPhase.ENERGY_MIN
        m.em_converged   = None
        m.em_fmax_final  = 5000.0
        m.em_fmax_target = 1000.0
        d = diagnose(m)
        assert d.code == DiagnosisCode.EM_NOT_CONVERGED

    def test_em_converged_evidence_contains_fmax(self):
        d = diagnose(em_converged(fmax=500.0))
        assert any("500" in e or "Fmax" in e for e in d.evidence)

    def test_em_not_converged_evidence_contains_fmax(self):
        d = diagnose(em_not_converged(fmax=15234.5))
        assert any("15234" in e or "Fmax" in e for e in d.evidence)

    @pytest.mark.parametrize("fmax,expected_code", [
        (999.9,  DiagnosisCode.SUCCESS_CONVERGED),
        (1000.0, DiagnosisCode.SUCCESS_CONVERGED),   # exactly at target
        (1000.1, DiagnosisCode.EM_NOT_CONVERGED),
        (50000.0,DiagnosisCode.EM_NOT_CONVERGED),
    ])
    def test_fmax_boundary_conditions(self, fmax, expected_code):
        m = LogMetrics()
        m.phase          = SimulationPhase.ENERGY_MIN
        m.em_converged   = None
        m.em_fmax_final  = fmax
        m.em_fmax_target = 1000.0
        d = diagnose(m)
        assert d.code == expected_code


# ---------------------------------------------------------------------------
# Priority 7: Equilibration / dynamics
# ---------------------------------------------------------------------------

class TestDynamicsDiagnosis:

    def test_nvt_stable_is_success(self):
        d = diagnose(nvt_stable())
        assert d.code     == DiagnosisCode.SUCCESS_COMPLETED
        assert d.severity == Severity.OK

    def test_nvt_temp_unstable_is_recoverable(self):
        d = diagnose(nvt_temp_unstable(temp_mean=340.0, temp_target=300.0))
        assert d.code     == DiagnosisCode.EQUIL_TEMP_UNSTABLE
        assert d.severity == Severity.RECOVERABLE

    def test_npt_stable_is_success(self):
        d = diagnose(npt_stable())
        assert d.code     == DiagnosisCode.SUCCESS_COMPLETED
        assert d.severity == Severity.OK

    def test_npt_pressure_unstable_is_recoverable(self):
        d = diagnose(npt_pressure_unstable(pres_mean=350.0))
        assert d.code     == DiagnosisCode.EQUIL_PRESSURE_UNSTABLE
        assert d.severity == Severity.RECOVERABLE

    def test_production_complete_is_success(self):
        d = diagnose(production_complete())
        assert d.code     == DiagnosisCode.SUCCESS_COMPLETED
        assert d.severity == Severity.OK

    def test_high_drift_is_recoverable(self):
        d = diagnose(production_high_drift(drift=50.0))
        assert d.code     == DiagnosisCode.EQUIL_DRIFT_TOO_HIGH
        assert d.severity == Severity.RECOVERABLE

    def test_drift_below_threshold_is_success(self):
        d = diagnose(production_complete(drift=_DRIFT_WARN_THRESHOLD - 0.1))
        assert d.code == DiagnosisCode.SUCCESS_COMPLETED

    def test_drift_above_threshold_is_recoverable(self):
        d = diagnose(production_high_drift(drift=_DRIFT_WARN_THRESHOLD + 0.1))
        assert d.code == DiagnosisCode.EQUIL_DRIFT_TOO_HIGH

    @pytest.mark.parametrize("temp_mean,expected_code", [
        (300.0, DiagnosisCode.SUCCESS_COMPLETED),
        (319.9, DiagnosisCode.SUCCESS_COMPLETED),   # just within tolerance
        (320.1, DiagnosisCode.EQUIL_TEMP_UNSTABLE), # just outside tolerance
        (350.0, DiagnosisCode.EQUIL_TEMP_UNSTABLE),
    ])
    def test_temperature_tolerance_boundary(self, temp_mean, expected_code):
        m = nvt_stable(
            temp_mean=temp_mean,
            temp_target=300.0,
            temp_values=[temp_mean] * 6,
        )
        d = diagnose(m)
        assert d.code == expected_code

    @pytest.mark.parametrize("pres_mean,expected_code", [
        (1.0,   DiagnosisCode.SUCCESS_COMPLETED),
        (199.9, DiagnosisCode.SUCCESS_COMPLETED),
        (200.1, DiagnosisCode.EQUIL_PRESSURE_UNSTABLE),
        (500.0, DiagnosisCode.EQUIL_PRESSURE_UNSTABLE),
    ])
    def test_pressure_tolerance_boundary(self, pres_mean, expected_code):
        m = npt_stable(
            pres_mean=pres_mean,
            pres_values=[pres_mean] * 6,
        )
        d = diagnose(m)
        assert d.code == expected_code

    def test_temp_instability_detected_via_std_dev(self):
        """High std dev in temperature values → EQUIL_TEMP_UNSTABLE."""
        m = nvt_stable(
            temp_mean=300.0,
            temp_target=300.0,
            # Mean is fine but std dev is very high
            temp_values=[250.0, 350.0, 260.0, 340.0, 270.0, 330.0],
        )
        d = diagnose(m)
        assert d.code == DiagnosisCode.EQUIL_TEMP_UNSTABLE


# ---------------------------------------------------------------------------
# Fallback / unknown
# ---------------------------------------------------------------------------

class TestFallbackDiagnosis:

    def test_unknown_phase_no_errors_is_success(self):
        d = diagnose(unknown_phase_no_errors())
        assert d.code     == DiagnosisCode.SUCCESS_COMPLETED
        assert d.severity == Severity.OK

    def test_unknown_phase_with_errors_needs_review(self):
        m = unknown_phase_no_errors()
        m.errors = ["Some unrecognised fatal error"]
        d = diagnose(m)
        assert d.code     == DiagnosisCode.UNKNOWN_ERROR
        assert d.severity == Severity.NEEDS_HUMAN_REVIEW

    def test_diagnosis_always_has_phase(self):
        """Every diagnosis must carry the phase from its metrics."""
        for factory in [
            em_converged, em_not_converged, nvt_stable,
            npt_stable, production_complete,
        ]:
            m = factory()
            d = diagnose(m)
            assert d.phase == m.phase

    def test_diagnosis_always_has_evidence(self):
        """Every non-success diagnosis must have at least one evidence item."""
        failure_metrics = [
            em_not_converged(), em_exploded(), em_nan(),
            em_lincs_error(), with_settle_error(),
            nvt_temp_unstable(), npt_pressure_unstable(),
            with_disk_full(), with_mpi_error(),
        ]
        for m in failure_metrics:
            d = diagnose(m)
            if d.severity != Severity.OK:
                assert len(d.evidence) >= 1, (
                    f"No evidence for {d.code} — agent cannot explain failure"
                )