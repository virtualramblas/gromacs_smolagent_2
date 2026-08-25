"""
4B-1: Verify SimulationPhase detection from integrator and title lines.

Rationale:
    Phase detection drives the entire DiagnosisEngine decision tree.
    A misdetected phase routes EM failures through dynamics logic
    and vice versa — producing wrong recovery actions.
"""

from __future__ import annotations

import pytest
from agent.recovery.log_parser import LogParser
from agent.recovery.models import SimulationPhase

from .log_fixtures import (
    EM_CONVERGED_LOG,
    EM_NOT_CONVERGED_LOG,
    NVT_STABLE_LOG,
    NPT_STABLE_LOG,
    PERFORMANCE_LOG,
)


def parse_text(text: str, tmp_path) -> object:
    """Write text to a tmp log file and parse it."""
    log = tmp_path / "test.log"
    log.write_text(text)
    return LogParser.parse(log)


class TestPhaseDetection:

    def test_steep_integrator_is_energy_min(self, tmp_path):
        m = parse_text(EM_CONVERGED_LOG, tmp_path)
        assert m.phase == SimulationPhase.ENERGY_MIN

    def test_steep_not_converged_is_energy_min(self, tmp_path):
        m = parse_text(EM_NOT_CONVERGED_LOG, tmp_path)
        assert m.phase == SimulationPhase.ENERGY_MIN

    def test_md_with_nvt_title_is_nvt(self, tmp_path):
        m = parse_text(NVT_STABLE_LOG, tmp_path)
        assert m.phase == SimulationPhase.NVT_EQUIL

    def test_md_with_npt_title_is_npt(self, tmp_path):
        m = parse_text(NPT_STABLE_LOG, tmp_path)
        assert m.phase == SimulationPhase.NPT_EQUIL

    def test_md_without_title_defaults_to_production(self, tmp_path):
        log_text = "integrator              = md\nnsteps = 500000\n"
        m = parse_text(log_text, tmp_path)
        assert m.phase == SimulationPhase.PRODUCTION_MD

    def test_cg_integrator_is_energy_min(self, tmp_path):
        log_text = "integrator              = cg\nnsteps = 5000\n"
        m = parse_text(log_text, tmp_path)
        assert m.phase == SimulationPhase.ENERGY_MIN

    def test_lbfgs_integrator_is_energy_min(self, tmp_path):
        log_text = "integrator              = l-bfgs\nnsteps = 5000\n"
        m = parse_text(log_text, tmp_path)
        assert m.phase == SimulationPhase.ENERGY_MIN

    def test_sd_integrator_is_production(self, tmp_path):
        log_text = "integrator              = sd\nnsteps = 500000\n"
        m = parse_text(log_text, tmp_path)
        # sd without nvt/npt title → production
        assert m.phase == SimulationPhase.PRODUCTION_MD

    def test_no_integrator_is_unknown(self, tmp_path):
        log_text = "title = some simulation\nnsteps = 5000\n"
        m = parse_text(log_text, tmp_path)
        assert m.phase == SimulationPhase.UNKNOWN

    def test_empty_log_is_unknown(self, tmp_path):
        m = parse_text("", tmp_path)
        assert m.phase == SimulationPhase.UNKNOWN

    @pytest.mark.parametrize("title,expected_phase", [
        ("title = NVT equilibration run",  SimulationPhase.NVT_EQUIL),
        ("title = NPT equilibration run",  SimulationPhase.NPT_EQUIL),
        ("title = Production MD run",      SimulationPhase.PRODUCTION_MD),
        ("title = md production",          SimulationPhase.PRODUCTION_MD),
    ])
    def test_phase_from_title_variants(self, tmp_path, title, expected_phase):
        log_text = f"integrator = md\n{title}\nnsteps = 50000\n"
        m = parse_text(log_text, tmp_path)
        assert m.phase == expected_phase