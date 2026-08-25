"""
4B-2: Verify EM convergence extraction — the most critical parsing task.

Rationale:
    em_converged, em_fmax_final, em_fmax_target, and em_steps_taken
    directly determine whether the agent retries EM or proceeds.
    False positives (reporting converged when not) skip recovery.
    False negatives (reporting failed when converged) cause infinite retries.
"""

from __future__ import annotations

import pytest
from agent.recovery.log_parser import LogParser

from .log_fixtures import (
    EM_CONVERGED_LOG,
    EM_EXPLODED_LOG,
    EM_LINCS_ERROR_LOG,
    EM_NAN_LOG,
    EM_NOT_CONVERGED_LOG,
)


def parse_text(text: str, tmp_path) -> object:
    log = tmp_path / "test.log"
    log.write_text(text)
    return LogParser.parse(log)


class TestEMConvergence:

    def test_converged_flag_true(self, tmp_path):
        m = parse_text(EM_CONVERGED_LOG, tmp_path)
        assert m.em_converged is True

    def test_converged_fmax_target_extracted(self, tmp_path):
        m = parse_text(EM_CONVERGED_LOG, tmp_path)
        assert m.em_fmax_target == pytest.approx(1000.0)

    def test_converged_steps_taken_extracted(self, tmp_path):
        m = parse_text(EM_CONVERGED_LOG, tmp_path)
        assert m.em_steps_taken == 847

    def test_converged_fmax_final_below_target(self, tmp_path):
        m = parse_text(EM_CONVERGED_LOG, tmp_path)
        assert m.em_fmax_final is not None
        assert m.em_fmax_final < 1000.0

    def test_not_converged_flag_false(self, tmp_path):
        m = parse_text(EM_NOT_CONVERGED_LOG, tmp_path)
        assert m.em_converged is False

    def test_not_converged_fmax_above_target(self, tmp_path):
        m = parse_text(EM_NOT_CONVERGED_LOG, tmp_path)
        assert m.em_fmax_final is not None
        assert m.em_fmax_final > 1000.0

    def test_not_converged_step_limit_extracted(self, tmp_path):
        m = parse_text(EM_NOT_CONVERGED_LOG, tmp_path)
        assert m.em_steps_limit == 1000

    def test_not_converged_steps_taken_at_limit(self, tmp_path):
        """When step limit is hit, last_step should equal nsteps."""
        m = parse_text(EM_NOT_CONVERGED_LOG, tmp_path)
        # last_step extracted from Step/Time table
        assert m.last_step == 1000

    def test_exploded_epot_positive(self, tmp_path):
        m = parse_text(EM_EXPLODED_LOG, tmp_path)
        assert m.epot_final is not None
        assert m.epot_final > 0

    def test_nan_flag_detected(self, tmp_path):
        m = parse_text(EM_NAN_LOG, tmp_path)
        assert m.has_nan is True

    def test_lincs_error_flag_detected(self, tmp_path):
        m = parse_text(EM_LINCS_ERROR_LOG, tmp_path)
        assert m.has_lincs_error is True

    def test_converged_epot_negative(self, tmp_path):
        """A properly minimised system should have negative Epot."""
        m = parse_text(EM_CONVERGED_LOG, tmp_path)
        assert m.epot_final is not None
        assert m.epot_final < 0

    @pytest.mark.parametrize("fmax_line,expected_fmax", [
        ("Maximum force     =  9.87654e+02", 987.654),
        ("Maximum force     =  1.52340e+04", 15234.0),
        ("Maximum force     =  1.00000e+03", 1000.0),
        ("Maximum force     =  1.23456e-01", 0.123456),
    ])
    def test_fmax_scientific_notation_variants(
        self, tmp_path, fmax_line, expected_fmax
    ):
        log_text = (
            "integrator = steep\nnsteps = 5000\n"
            f"Steepest Descents did not converge to Fmax < 1000 in 500 steps\n"
            f"{fmax_line}\n"
        )
        m = parse_text(log_text, tmp_path)
        assert m.em_fmax_final == pytest.approx(expected_fmax, rel=1e-3)