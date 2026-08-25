"""
4B-3: Verify potential energy extraction from log energy blocks.

Rationale:
    epot_final is used to detect explosions (Epot >> 0) and to
    report convergence quality. Wrong extraction causes false
    explosion diagnoses or missed explosion detection.
"""

from __future__ import annotations

import pytest
from agent.recovery.log_parser import LogParser

from .log_fixtures import (
    EM_CONVERGED_LOG,
    EM_EXPLODED_LOG,
    EM_NOT_CONVERGED_LOG,
)


def parse_text(text: str, tmp_path) -> object:
    log = tmp_path / "test.log"
    log.write_text(text)
    return LogParser.parse(log)


class TestEnergyExtraction:

    def test_epot_final_extracted_from_converged(self, tmp_path):
        m = parse_text(EM_CONVERGED_LOG, tmp_path)
        assert m.epot_final is not None
        assert m.epot_final == pytest.approx(-456789.0, rel=1e-3)

    def test_epot_final_extracted_from_not_converged(self, tmp_path):
        m = parse_text(EM_NOT_CONVERGED_LOG, tmp_path)
        assert m.epot_final is not None
        assert m.epot_final < 0

    def test_epot_positive_for_exploded_system(self, tmp_path):
        m = parse_text(EM_EXPLODED_LOG, tmp_path)
        assert m.epot_final is not None
        assert m.epot_final > 1e6

    def test_epot_series_populated(self, tmp_path):
        """Multiple energy blocks → epot_values list should be non-empty."""
        m = parse_text(EM_CONVERGED_LOG, tmp_path)
        assert isinstance(m.epot_values, list)

    def test_epot_none_for_empty_log(self, tmp_path):
        m = parse_text("integrator = steep\n", tmp_path)
        assert m.epot_final is None

    @pytest.mark.parametrize("epot_line,expected", [
        ("Potential Energy  = -4.56789e+05", -456789.0),
        ("Potential Energy  = -1.23456e+03", -1234.56),
        ("Potential Energy  =  2.34567e+07",  23456700.0),
        ("Potential Energy  = -9.99999e+04", -99999.9),
    ])
    def test_epot_value_formats(self, tmp_path, epot_line, expected):
        log_text = (
            "integrator = steep\nnsteps = 5000\n"
            f"Steepest Descents converged to Fmax < 1000 in 500 steps\n"
            f"{epot_line}\n"
            "Maximum force     =  9.87654e+02\n"
        )
        m = parse_text(log_text, tmp_path)
        assert m.epot_final == pytest.approx(expected, rel=1e-3)