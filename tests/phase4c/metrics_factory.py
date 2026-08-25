"""
Factory helpers for building LogMetrics objects in specific states.
Keeps test bodies concise and intent-revealing.
"""

from __future__ import annotations

from agent.recovery.models import LogMetrics, SimulationPhase


def em_converged(
    fmax: float = 500.0,
    fmax_target: float = 1000.0,
    steps_taken: int = 847,
    epot: float = -456789.0,
) -> LogMetrics:
    m = LogMetrics()
    m.phase          = SimulationPhase.ENERGY_MIN
    m.em_converged   = True
    m.em_fmax_final  = fmax
    m.em_fmax_target = fmax_target
    m.em_steps_taken = steps_taken
    m.epot_final     = epot
    return m


def em_not_converged(
    fmax: float = 15234.5,
    fmax_target: float = 1000.0,
    steps_taken: int = 1000,
    steps_limit: int = 1000,
    epot: float = -234567.0,
) -> LogMetrics:
    m = LogMetrics()
    m.phase          = SimulationPhase.ENERGY_MIN
    m.em_converged   = False
    m.em_fmax_final  = fmax
    m.em_fmax_target = fmax_target
    m.em_steps_taken = steps_taken
    m.em_steps_limit = steps_limit
    m.epot_final     = epot
    return m


def em_exploded(epot: float = 2.5e7) -> LogMetrics:
    m = LogMetrics()
    m.phase      = SimulationPhase.ENERGY_MIN
    m.epot_final = epot
    return m


def em_nan() -> LogMetrics:
    m = LogMetrics()
    m.phase   = SimulationPhase.ENERGY_MIN
    m.has_nan = True
    return m


def em_lincs_error(last_step: int = 100) -> LogMetrics:
    m = LogMetrics()
    m.phase           = SimulationPhase.ENERGY_MIN
    m.has_lincs_error = True
    m.last_step       = last_step
    return m


def em_lincs_warning() -> LogMetrics:
    m = LogMetrics()
    m.phase             = SimulationPhase.ENERGY_MIN
    m.has_lincs_warning = True
    return m


def nvt_stable(
    temp_mean: float = 300.1,
    temp_target: float = 300.0,
    temp_values: list[float] | None = None,
) -> LogMetrics:
    m = LogMetrics()
    m.phase              = SimulationPhase.NVT_EQUIL
    m.temperature_target = temp_target
    m.temperature_mean   = temp_mean
    m.temperature_values = temp_values or [299.8, 300.1, 300.2, 299.9, 300.1]
    return m


def nvt_temp_unstable(
    temp_mean: float = 340.0,
    temp_target: float = 300.0,
    temp_values: list[float] | None = None,
) -> LogMetrics:
    m = LogMetrics()
    m.phase              = SimulationPhase.NVT_EQUIL
    m.temperature_target = temp_target
    m.temperature_mean   = temp_mean
    m.temperature_values = temp_values or [250.0, 340.0, 280.0, 360.0, 310.0]
    return m


def npt_stable(
    temp_mean: float = 300.1,
    temp_target: float = 300.0,
    pres_mean: float = 1.5,
    pres_values: list[float] | None = None,
) -> LogMetrics:
    m = LogMetrics()
    m.phase              = SimulationPhase.NPT_EQUIL
    m.temperature_target = temp_target
    m.temperature_mean   = temp_mean
    m.temperature_values = [299.8, 300.1, 300.2, 299.9, 300.1]
    m.pressure_mean      = pres_mean
    m.pressure_values    = pres_values or [1.0, 1.5, 0.8, 1.2, 1.1]
    return m


def npt_pressure_unstable(
    pres_mean: float = 350.0,
    pres_values: list[float] | None = None,
) -> LogMetrics:
    m = LogMetrics()
    m.phase              = SimulationPhase.NPT_EQUIL
    m.temperature_target = 300.0
    m.temperature_mean   = 300.1
    m.temperature_values = [299.8, 300.1, 300.2, 299.9, 300.1]
    m.pressure_mean      = pres_mean
    m.pressure_values    = pres_values or [100.0, 350.0, 200.0, 400.0, 300.0]
    return m


def production_complete(
    performance: float = 45.6,
    drift: float = 0.5,
) -> LogMetrics:
    m = LogMetrics()
    m.phase                  = SimulationPhase.PRODUCTION_MD
    m.performance_ns_per_day = performance
    m.drift_kj_per_ns        = drift
    return m


def production_high_drift(drift: float = 50.0) -> LogMetrics:
    m = LogMetrics()
    m.phase           = SimulationPhase.PRODUCTION_MD
    m.drift_kj_per_ns = drift
    return m


def with_disk_full() -> LogMetrics:
    m = LogMetrics()
    m.phase         = SimulationPhase.PRODUCTION_MD
    m.has_disk_full = True
    return m


def with_mpi_error() -> LogMetrics:
    m = LogMetrics()
    m.phase         = SimulationPhase.PRODUCTION_MD
    m.has_mpi_error = True
    return m


def with_gpu_error() -> LogMetrics:
    m = LogMetrics()
    m.phase         = SimulationPhase.PRODUCTION_MD
    m.has_gpu_error = True
    return m


def with_settle_error() -> LogMetrics:
    m = LogMetrics()
    m.phase            = SimulationPhase.NVT_EQUIL
    m.has_settle_error = True
    return m


def with_particle_escaped() -> LogMetrics:
    m = LogMetrics()
    m.phase               = SimulationPhase.PRODUCTION_MD
    m.has_particle_escaped = True
    return m


def with_missing_params() -> LogMetrics:
    m = LogMetrics()
    m.phase              = SimulationPhase.ENERGY_MIN
    m.has_missing_params = True
    return m


def with_charge_imbalance() -> LogMetrics:
    m = LogMetrics()
    m.phase              = SimulationPhase.ENERGY_MIN
    m.has_charge_warning = True
    return m


def with_topology_error() -> LogMetrics:
    m = LogMetrics()
    m.phase              = SimulationPhase.ENERGY_MIN
    m.has_topology_error = True
    return m


def unknown_phase_no_errors() -> LogMetrics:
    """Clean metrics with unknown phase — no errors, no flags."""
    m = LogMetrics()
    m.phase = SimulationPhase.UNKNOWN
    return m