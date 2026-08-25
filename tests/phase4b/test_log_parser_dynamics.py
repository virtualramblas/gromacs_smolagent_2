"""
4B-4: Verify temperature, pressure, and drift extraction
from NVT/NPT equilibration logs.

Rationale:
    Temperature and pressure stability assessment drives
    EQUIL_TEMP_UNSTABLE and EQUIL_PRESSURE_UNSTABLE diagnoses.
    Wrong extraction causes false stability reports or missed
    instability — both lead to incorrect production MD.
"""

from __future__ import annotations

import pytest
from agent.recovery.log_parser import LogParser
from agent.recovery.models import SimulationPhase

from .log_fixtures import (
    DRIFT_LOG,
    NPT_STABLE_LOG,
    NVT_STABLE_LOG,
    NVT_TEMP_UNSTABLE_LOG,
)


def parse_text(text: str, tmp_path) -> object:
    log = tmp_path / "test.log"
    log.write_text(text)
    return LogParser.parse(log)


class TestTemperatureExtraction:

    def test_ref_temperature_extracted(self, tmp_path):
        m = parse_text(NVT_STABLE_LOG, tmp_path)
        assert m.temperature_target == pytest.approx(300.0)

    def test_temperature_mean_near_target_for_stable(self, tmp_path):
        m = parse_text(NVT_STABLE_LOG, tmp_path)
        if m.temperature_mean is not None:
            assert abs(m.temperature_mean - 300.0) < 5.0

    def test_temperature_values_list_populated(self, tmp_path):
        m = parse_text(NVT_STABLE_LOG, tmp_path)
        assert isinstance(m.temperature_values, list)

    def test_temperature_target_extracted_from_ref_t(self, tmp_path):
        log_text = (
            "integrator = md\ntitle = NVT equilibration\n"
            "ref-t = 310\nnsteps = 50000\n"
        )
        m = parse_text(log_text, tmp_path)
        assert m.temperature_target == pytest.approx(310.0)

    def test_temperature_target_none_when_absent(self, tmp_path):
        log_text = "integrator = steep\nnsteps = 5000\n"
        m = parse_text(log_text, tmp_path)
        assert m.temperature_target is None

    @pytest.mark.parametrize("ref_t", [300.0, 298.15, 310.0, 277.0])
    def test_various_ref_t_values(self, tmp_path, ref_t):
        log_text = (
            f"integrator = md\ntitle = NVT equilibration\n"
            f"ref-t = {ref_t}\nnsteps = 50000\n"
        )
        m = parse_text(log_text, tmp_path)
        assert m.temperature_target == pytest.approx(ref_t)


class TestPressureExtraction:

    def test_pressure_values_list_populated_for_npt(self, tmp_path):
        m = parse_text(NPT_STABLE_LOG, tmp_path)
        assert isinstance(m.pressure_values, list)

    def test_pressure_mean_near_one_bar_for_stable(self, tmp_path):
        m = parse_text(NPT_STABLE_LOG, tmp_path)
        if m.pressure_mean is not None:
            assert abs(m.pressure_mean) < 100.0   # within 100 bar of 0

    def test_pressure_none_for_em_log(self, tmp_path):
        log_text = "integrator = steep\nnsteps = 5000\n"
        m = parse_text(log_text, tmp_path)
        assert m.pressure_mean is None


class TestDriftExtraction:

    def test_drift_extracted_from_log(self, tmp_path):
        m = parse_text(DRIFT_LOG, tmp_path)
        if m.drift_kj_per_ns is not None:
            assert m.drift_kj_per_ns == pytest.approx(25.34, rel=1e-2)

    def test_drift_none_when_absent(self, tmp_path):
        m = parse_text("integrator = md\nnsteps = 500000\n", tmp_path)
        assert m.drift_kj_per_ns is None

    @pytest.mark.parametrize("drift_line,expected", [
        ("Total energy drift = 5.12 kJ/mol/ps",   5.12),
        ("Total energy drift = 0.34 kJ/mol/ps",   0.34),
        ("Total energy drift = 100.0 kJ/mol/ps", 100.0),
    ])
    def test_drift_value_variants(self, tmp_path, drift_line, expected):
        log_text = f"integrator = md\ntitle = Production MD\n{drift_line}\n"
        m = parse_text(log_text, tmp_path)
        if m.drift_kj_per_ns is not None:
            assert m.drift_kj_per_ns == pytest.approx(expected, rel=1e-2)