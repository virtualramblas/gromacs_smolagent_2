"""
4A-1: Verify all enums are complete, consistent, and serialisable.

Rationale:
    DiagnosisCode, Severity, RecoveryAction, and SimulationPhase are the
    vocabulary the LLM reads. Any missing or misspelled member breaks the
    recovery loop silently. These tests act as a registry contract.
"""

from __future__ import annotations

import pytest
from agent.recovery.models import (
    DiagnosisCode,
    RecoveryAction,
    Severity,
    SimulationPhase,
)


# ---------------------------------------------------------------------------
# SimulationPhase
# ---------------------------------------------------------------------------

class TestSimulationPhase:

    def test_all_expected_phases_present(self):
        expected = {
            "UNKNOWN",
            "ENERGY_MIN",
            "NVT_EQUIL",
            "NPT_EQUIL",
            "PRODUCTION_MD",
            "GENION_PREP",
        }
        actual = {m.value for m in SimulationPhase}
        assert expected == actual, (
            f"Missing phases: {expected - actual}, "
            f"unexpected phases: {actual - expected}"
        )

    def test_members_are_strings(self):
        for member in SimulationPhase:
            assert isinstance(member.value, str)

    def test_members_serialise_to_string(self):
        """Enum values must be plain strings for LLM output serialisation."""
        for member in SimulationPhase:
            assert str(member.value) == member.value


# ---------------------------------------------------------------------------
# DiagnosisCode
# ---------------------------------------------------------------------------

class TestDiagnosisCode:

    # Every code must belong to exactly one category
    _SUCCESS_CODES = {
        "SUCCESS_CONVERGED",
        "SUCCESS_COMPLETED",
    }
    _EM_CODES = {
        "EM_NOT_CONVERGED",
        "EM_STEP_LIMIT_HIT",
        "EM_LINCS_ERROR",
        "EM_EXPLODED",
    }
    _EQUIL_CODES = {
        "EQUIL_TEMP_UNSTABLE",
        "EQUIL_PRESSURE_UNSTABLE",
        "EQUIL_DRIFT_TOO_HIGH",
    }
    _RUNTIME_CODES = {
        "LINCS_WARNING",
        "SETTLE_ERROR",
        "PARTICLE_ESCAPED_BOX",
        "NAN_DETECTED",
        "NEIGHBOUR_LIST_ERROR",
    }
    _SETUP_CODES = {
        "MISSING_PARAMETERS",
        "CHARGE_IMBALANCE",
        "TOPOLOGY_MISMATCH",
    }
    _INFRA_CODES = {
        "GPU_ERROR",
        "MPI_ERROR",
        "DISK_FULL",
        "TIMEOUT",
    }
    _FALLBACK_CODES = {
        "UNKNOWN_ERROR",
        "NEEDS_HUMAN_REVIEW",
    }

    @property
    def _all_expected(self) -> set[str]:
        return (
            self._SUCCESS_CODES
            | self._EM_CODES
            | self._EQUIL_CODES
            | self._RUNTIME_CODES
            | self._SETUP_CODES
            | self._INFRA_CODES
            | self._FALLBACK_CODES
        )

    def test_all_expected_codes_present(self):
        actual = {m.value for m in DiagnosisCode}
        missing = self._all_expected - actual
        assert not missing, f"Missing DiagnosisCode members: {missing}"

    def test_no_unexpected_codes(self):
        actual = {m.value for m in DiagnosisCode}
        unexpected = actual - self._all_expected
        assert not unexpected, f"Unexpected DiagnosisCode members: {unexpected}"

    def test_members_are_strings(self):
        for member in DiagnosisCode:
            assert isinstance(member.value, str)

    def test_no_duplicate_values(self):
        values = [m.value for m in DiagnosisCode]
        assert len(values) == len(set(values)), "Duplicate DiagnosisCode values found"

    def test_success_codes_are_distinct_from_failure_codes(self):
        failure_codes = (
            self._EM_CODES
            | self._EQUIL_CODES
            | self._RUNTIME_CODES
            | self._SETUP_CODES
            | self._INFRA_CODES
        )
        overlap = self._SUCCESS_CODES & failure_codes
        assert not overlap, f"Success/failure code overlap: {overlap}"


# ---------------------------------------------------------------------------
# Severity
# ---------------------------------------------------------------------------

class TestSeverity:

    _EXPECTED = {
        "OK",
        "RECOVERABLE",
        "ASSISTED",
        "FATAL",
        "NEEDS_HUMAN_REVIEW",    
    }

    def test_all_expected_severities_present(self):
        actual = {m.value for m in Severity}
        assert self._EXPECTED == actual

    def test_severity_ordering_concept(self):
        assert Severity.OK              != Severity.RECOVERABLE
        assert Severity.RECOVERABLE     != Severity.ASSISTED
        assert Severity.ASSISTED        != Severity.FATAL
        assert Severity.FATAL           != Severity.NEEDS_HUMAN_REVIEW  

    def test_members_are_strings(self):
        for member in Severity:
            assert isinstance(member.value, str)


# ---------------------------------------------------------------------------
# RecoveryAction
# ---------------------------------------------------------------------------

class TestRecoveryAction:

    _EXPECTED = {
        "NONE",
        "REDUCE_EMSTEP",
        "INCREASE_NSTEPS",
        "REDUCE_EMSTEP_AND_INCREASE_NSTEPS",
        "SWITCH_INTEGRATOR_SD",
        "REDUCE_DT",
        "INCREASE_LINCS_ORDER",
        "DISABLE_LINCS",
        "INCREASE_NEIGHBOUR_FREQ",
        "REBUILD_TOPOLOGY",
        "RERUN_GENION",
        "SWITCH_TO_CPU",
        "RESUME_FROM_CHECKPOINT",
        "ESCALATE_TO_USER",
    }

    def test_all_expected_actions_present(self):
        actual = {m.value for m in RecoveryAction}
        missing = self._EXPECTED - actual
        assert not missing, f"Missing RecoveryAction members: {missing}"

    def test_none_action_exists(self):
        """NONE must exist as the no-op action for success states."""
        assert RecoveryAction.NONE.value == "NONE"

    def test_escalate_action_exists(self):
        """ESCALATE_TO_USER must exist as the terminal fallback."""
        assert RecoveryAction.ESCALATE_TO_USER.value == "ESCALATE_TO_USER"

    def test_no_duplicate_values(self):
        values = [m.value for m in RecoveryAction]
        assert len(values) == len(set(values))

    def test_members_are_strings(self):
        for member in RecoveryAction:
            assert isinstance(member.value, str)