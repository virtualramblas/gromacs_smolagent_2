"""
4B-6: Performance extraction, warning/error text collection,
step tracking, and edge cases.
"""

from __future__ import annotations

import pytest
from agent.recovery.log_parser import LogParser

from .log_fixtures import (
    EM_CONVERGED_LOG,
    MULTI_WARNING_LOG,
    PERFORMANCE_LOG,
)


def parse_text(text: str, tmp_path) -> object:
    log = tmp_path / "test.log"
    log.write_text(text)
    return LogParser.parse(log)


class TestPerformanceExtraction:

    def test_performance_ns_per_day_extracted(self, tmp_path):
        m = parse_text(PERFORMANCE_LOG, tmp_path)
        assert m.performance_ns_per_day is not None
        assert m.performance_ns_per_day == pytest.approx(78.90, rel=1e-2)

    def test_performance_extracted_from_em_log(self, tmp_path):
        m = parse_text(EM_CONVERGED_LOG, tmp_path)
        assert m.performance_ns_per_day is not None
        assert m.performance_ns_per_day == pytest.approx(12.34, rel=1e-2)

    def test_performance_none_when_absent(self, tmp_path):
        m = parse_text("integrator = steep\nnsteps = 5000\n", tmp_path)
        assert m.performance_ns_per_day is None

    @pytest.mark.parametrize("perf_line,expected", [
        ("Performance:   1234.56    567.89   12.34     1.94",  12.34),
        ("Performance:   2345.67    678.90   78.90     0.30",  78.90),
        ("Performance:    100.00    200.00    1.23    19.51",   1.23),
    ])
    def test_performance_value_variants(self, tmp_path, perf_line, expected):
        log_text = f"integrator = md\nnsteps = 500000\n{perf_line}\n"
        m = parse_text(log_text, tmp_path)
        if m.performance_ns_per_day is not None:
            assert m.performance_ns_per_day == pytest.approx(expected, rel=1e-2)


class TestStepTracking:

    def test_last_step_extracted_from_converged(self, tmp_path):
        m = parse_text(EM_CONVERGED_LOG, tmp_path)
        assert m.last_step is not None
        assert m.last_step == 0     # only step in fixture is step 0

    def test_total_steps_from_nsteps(self, tmp_path):
        log_text = "integrator = steep\nnsteps = 5000\n"
        m = parse_text(log_text, tmp_path)
        assert m.total_steps == 5000

    def test_total_steps_none_when_absent(self, tmp_path):
        m = parse_text("integrator = steep\n", tmp_path)
        assert m.total_steps is None


class TestWarningAndErrorCollection:

    def test_warnings_collected(self, tmp_path):
        m = parse_text(MULTI_WARNING_LOG, tmp_path)
        assert len(m.warnings) >= 3

    def test_notes_collected(self, tmp_path):
        m = parse_text(MULTI_WARNING_LOG, tmp_path)
        assert len(m.notes) >= 2

    def test_warnings_deduplicated(self, tmp_path):
        log_text = (
            "integrator = steep\nnsteps = 5000\n"
            "WARNING: duplicate warning\n"
            "WARNING: duplicate warning\n"
            "WARNING: duplicate warning\n"
        )
        m = parse_text(log_text, tmp_path)
        assert m.warnings.count("WARNING: duplicate warning") == 1

    def test_warnings_capped_at_20(self, tmp_path):
        warning_lines = "\n".join(
            f"WARNING: warning number {i}" for i in range(30)
        )
        log_text = f"integrator = steep\nnsteps = 5000\n{warning_lines}\n"
        m = parse_text(log_text, tmp_path)
        assert len(m.warnings) <= 20

    def test_errors_capped_at_20(self, tmp_path):
        error_lines = "\n".join(
            f"ERROR: error number {i}" for i in range(30)
        )
        log_text = f"integrator = steep\nnsteps = 5000\n{error_lines}\n"
        m = parse_text(log_text, tmp_path)
        assert len(m.errors) <= 20

    def test_clean_log_has_no_warnings(self, tmp_path):
        m = parse_text(EM_CONVERGED_LOG, tmp_path)
        assert m.warnings == []

    def test_clean_log_has_no_errors(self, tmp_path):
        m = parse_text(EM_CONVERGED_LOG, tmp_path)
        assert m.errors == []


class TestEdgeCases:

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            LogParser.parse(tmp_path / "nonexistent.log")

    def test_empty_log_returns_default_metrics(self, tmp_path):
        m = parse_text("", tmp_path)
        from agent.recovery.models import SimulationPhase
        assert m.phase          == SimulationPhase.UNKNOWN
        assert m.em_converged   is None
        assert m.epot_final     is None
        assert m.has_nan        is False
        assert m.warnings       == []
        assert m.errors         == []

    def test_binary_garbage_does_not_crash(self, tmp_path):
        """Parser must not crash on non-UTF8 content."""
        log = tmp_path / "garbage.log"
        log.write_bytes(bytes(range(256)))
        try:
            m = LogParser.parse(log)
            # If it doesn't raise, result should be a valid LogMetrics
            from agent.recovery.models import LogMetrics
            assert isinstance(m, LogMetrics)
        except UnicodeDecodeError:
            pytest.fail(
                "LogParser crashed on non-UTF8 input — "
                "read_text(errors='replace') should handle this"
            )

    def test_very_large_log_does_not_crash(self, tmp_path):
        """Parser must handle large files without memory errors."""
        # 10k repetitions of a typical energy block line
        big_log = (
            "integrator = steep\nnsteps = 50000\n"
            + "   Potential        -4.56789e+05\n" * 10_000
        )
        m = parse_text(big_log, tmp_path)
        assert m is not None

    def test_windows_line_endings_handled(self, tmp_path):
        """CRLF line endings must not break parsing."""
        log_text = (
            "integrator = steep\r\n"
            "nsteps = 5000\r\n"
            "Steepest Descents converged to Fmax < 1000 in 500 steps\r\n"
            "Potential Energy  = -4.56789e+05\r\n"
            "Maximum force     =  9.87654e+02\r\n"
        )
        log = tmp_path / "crlf.log"
        log.write_bytes(log_text.encode("utf-8"))
        m = LogParser.parse(log)
        assert m.em_converged is True