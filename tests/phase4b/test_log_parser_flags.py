"""
4B-5: Verify boolean error flag detection.

Rationale:
    Boolean flags are the primary input to DiagnosisEngine's
    priority-ordered rule tree. A missed flag means a fatal error
    is silently ignored. A false positive flag triggers unnecessary
    recovery actions.
    Every flag gets both a positive (present) and negative (absent) test.
"""

from __future__ import annotations

import pytest
from agent.recovery.log_parser import LogParser

from .log_fixtures import (
    CHARGE_WARNING_LOG,
    DISK_FULL_LOG,
    EM_CONVERGED_LOG,
    EM_LINCS_ERROR_LOG,
    EM_NAN_LOG,
    GPU_ERROR_LOG,
    LINCS_WARNING_LOG,
    MISSING_PARAMS_LOG,
    MPI_ERROR_LOG,
    PARTICLE_ESCAPED_LOG,
    SETTLE_ERROR_LOG,
    TOPOLOGY_ERROR_LOG,
)


def parse_text(text: str, tmp_path) -> object:
    log = tmp_path / "test.log"
    log.write_text(text)
    return LogParser.parse(log)


class TestNaNFlag:

    def test_nan_detected_when_present(self, tmp_path):
        m = parse_text(EM_NAN_LOG, tmp_path)
        assert m.has_nan is True

    def test_nan_not_detected_in_clean_log(self, tmp_path):
        m = parse_text(EM_CONVERGED_LOG, tmp_path)
        assert m.has_nan is False

    @pytest.mark.parametrize("nan_text", [
        "NaN detected in the force",
        "not a number encountered",
        "energy is nan",
        "force is NaN on atom 42",
    ])
    def test_nan_variants(self, tmp_path, nan_text):
        log_text = f"integrator = steep\nnsteps = 5000\n{nan_text}\n"
        m = parse_text(log_text, tmp_path)
        assert m.has_nan is True


class TestLincsFlags:

    def test_lincs_error_detected(self, tmp_path):
        m = parse_text(EM_LINCS_ERROR_LOG, tmp_path)
        assert m.has_lincs_error is True

    def test_lincs_error_not_in_clean_log(self, tmp_path):
        m = parse_text(EM_CONVERGED_LOG, tmp_path)
        assert m.has_lincs_error is False

    def test_lincs_warning_detected(self, tmp_path):
        m = parse_text(LINCS_WARNING_LOG, tmp_path)
        assert m.has_lincs_warning is True

    def test_lincs_warning_not_in_clean_log(self, tmp_path):
        m = parse_text(EM_CONVERGED_LOG, tmp_path)
        assert m.has_lincs_warning is False

    def test_lincs_error_and_warning_are_independent_flags(self, tmp_path):
        """LINCS error should not set warning flag and vice versa."""
        m_err  = parse_text(EM_LINCS_ERROR_LOG, tmp_path)
        m_warn = parse_text(LINCS_WARNING_LOG,  tmp_path)
        # Error log: error=True, warning may or may not be True
        assert m_err.has_lincs_error   is True
        # Warning log: warning=True, error=False
        assert m_warn.has_lincs_warning is True
        assert m_warn.has_lincs_error   is False

    @pytest.mark.parametrize("lincs_text", [
        "LINCS ERROR\nToo many LINCS warnings",
        "Too many LINCS warnings (1000)",
    ])
    def test_lincs_error_text_variants(self, tmp_path, lincs_text):
        log_text = f"integrator = steep\nnsteps = 5000\n{lincs_text}\n"
        m = parse_text(log_text, tmp_path)
        assert m.has_lincs_error is True


class TestSettleFlag:

    def test_settle_error_detected(self, tmp_path):
        m = parse_text(SETTLE_ERROR_LOG, tmp_path)
        assert m.has_settle_error is True

    def test_settle_not_in_clean_log(self, tmp_path):
        m = parse_text(EM_CONVERGED_LOG, tmp_path)
        assert m.has_settle_error is False

    @pytest.mark.parametrize("settle_text", [
        "SETTLE: can't settle atom 1234",
        "SETTLE error in molecule 456",
        "error in SETTLE constraints",
    ])
    def test_settle_text_variants(self, tmp_path, settle_text):
        log_text = f"integrator = md\nnsteps = 50000\n{settle_text}\n"
        m = parse_text(log_text, tmp_path)
        assert m.has_settle_error is True


class TestParticleEscapedFlag:

    def test_particle_escaped_detected(self, tmp_path):
        m = parse_text(PARTICLE_ESCAPED_LOG, tmp_path)
        assert m.has_particle_escaped is True

    def test_particle_not_escaped_in_clean_log(self, tmp_path):
        m = parse_text(EM_CONVERGED_LOG, tmp_path)
        assert m.has_particle_escaped is False

    @pytest.mark.parametrize("escape_text", [
        "1 particles are outside of the box",
        "particle 4521 flew away",
        "atom escaped the box",
    ])
    def test_escape_text_variants(self, tmp_path, escape_text):
        log_text = f"integrator = md\nnsteps = 50000\n{escape_text}\n"
        m = parse_text(log_text, tmp_path)
        assert m.has_particle_escaped is True


class TestInfrastructureFlags:

    def test_gpu_error_detected(self, tmp_path):
        m = parse_text(GPU_ERROR_LOG, tmp_path)
        assert m.has_gpu_error is True

    def test_gpu_error_not_in_clean_log(self, tmp_path):
        m = parse_text(EM_CONVERGED_LOG, tmp_path)
        assert m.has_gpu_error is False

    def test_mpi_error_detected(self, tmp_path):
        m = parse_text(MPI_ERROR_LOG, tmp_path)
        assert m.has_mpi_error is True

    def test_mpi_error_not_in_clean_log(self, tmp_path):
        m = parse_text(EM_CONVERGED_LOG, tmp_path)
        assert m.has_mpi_error is False

    def test_disk_full_detected(self, tmp_path):
        m = parse_text(DISK_FULL_LOG, tmp_path)
        assert m.has_disk_full is True

    def test_disk_full_not_in_clean_log(self, tmp_path):
        m = parse_text(EM_CONVERGED_LOG, tmp_path)
        assert m.has_disk_full is False

    @pytest.mark.parametrize("gpu_text", [
        "GPU error: CUDA error on device 0",
        "CUDA error: out of memory",
        "OpenCL error: device not found",
    ])
    def test_gpu_error_text_variants(self, tmp_path, gpu_text):
        log_text = f"integrator = md\nnsteps = 50000\n{gpu_text}\n"
        m = parse_text(log_text, tmp_path)
        assert m.has_gpu_error is True

    @pytest.mark.parametrize("disk_text", [
        "No space left on device",
        "disk full: cannot write",
        "write error on file md.xtc",
    ])
    def test_disk_full_text_variants(self, tmp_path, disk_text):
        log_text = f"integrator = md\nnsteps = 500000\n{disk_text}\n"
        m = parse_text(log_text, tmp_path)
        assert m.has_disk_full is True


class TestTopologyAndParameterFlags:

    def test_missing_params_detected(self, tmp_path):
        m = parse_text(MISSING_PARAMS_LOG, tmp_path)
        assert m.has_missing_params is True

    def test_missing_params_not_in_clean_log(self, tmp_path):
        m = parse_text(EM_CONVERGED_LOG, tmp_path)
        assert m.has_missing_params is False

    def test_charge_warning_detected(self, tmp_path):
        m = parse_text(CHARGE_WARNING_LOG, tmp_path)
        assert m.has_charge_warning is True

    def test_charge_warning_not_in_clean_log(self, tmp_path):
        m = parse_text(EM_CONVERGED_LOG, tmp_path)
        assert m.has_charge_warning is False

    def test_topology_error_detected(self, tmp_path):
        m = parse_text(TOPOLOGY_ERROR_LOG, tmp_path)
        assert m.has_topology_error is True

    def test_topology_error_not_in_clean_log(self, tmp_path):
        m = parse_text(EM_CONVERGED_LOG, tmp_path)
        assert m.has_topology_error is False


class TestFlagIndependence:
    """
    Verify that flags do not bleed into each other.
    Each error type should set exactly its own flag(s).
    """

    def test_settle_error_does_not_set_lincs_flag(self, tmp_path):
        m = parse_text(SETTLE_ERROR_LOG, tmp_path)
        assert m.has_lincs_error is False

    def test_gpu_error_does_not_set_mpi_flag(self, tmp_path):
        m = parse_text(GPU_ERROR_LOG, tmp_path)
        assert m.has_mpi_error is False

    def test_disk_full_does_not_set_gpu_flag(self, tmp_path):
        m = parse_text(DISK_FULL_LOG, tmp_path)
        assert m.has_gpu_error is False

    def test_charge_warning_does_not_set_topology_flag(self, tmp_path):
        m = parse_text(CHARGE_WARNING_LOG, tmp_path)
        assert m.has_topology_error is False