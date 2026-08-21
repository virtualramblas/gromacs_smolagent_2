"""
Pure regex-based extractor of raw metrics from GROMACS log files.
No LLM involvement. Produces a LogMetrics object.
"""

from __future__ import annotations

import re
import statistics
from pathlib import Path

from .models import LogMetrics, SimulationPhase


class LogParser:
    """
    Stateless parser — call LogParser.parse(path) to get LogMetrics.
    All patterns are compiled once at class level for performance.
    """

    # ------------------------------------------------------------------
    # Compiled regex patterns
    # ------------------------------------------------------------------

    # Phase detection (from mdp integrator / run title lines)
    _P_INTEGRATOR   = re.compile(r"integrator\s*=\s*(\S+)", re.I)
    _P_TITLE        = re.compile(r"^title\s*=\s*(.+)$", re.I | re.M)

    # EM convergence
    _P_EM_CONV      = re.compile(
        r"converged to Fmax\s*<\s*([\d.eE+\-]+)\s*in\s*(\d+)\s*steps",
        re.I
    )
    _P_EM_NOT_CONV  = re.compile(r"did not converge to Fmax", re.I)
    _P_FMAX         = re.compile(r"Maximum force\s*[=:]\s*([\d.eE+\-]+)")
    _P_FMAX_TARGET  = re.compile(r"Fmax\s*<\s*([\d.eE+\-]+)")
    _P_NSTEPS_LIMIT = re.compile(r"nsteps\s*=\s*(\d+)", re.I)

    # Energies — matches table rows in GROMACS log energy blocks
    _P_EPOT_LABEL   = re.compile(r"Potential Energy\s*$", re.M)
    _P_EPOT_VALUE   = re.compile(
        r"Potential Energy\s*[=:\s]+([-\d.eE+]+)"
    )
    _P_EPOT_BLOCK   = re.compile(
        r"Potential\s+([-\d.eE+]+)", re.M
    )

    # Temperature
    _P_TEMP_REF     = re.compile(r"ref[_-]?t\s*=\s*([\d.]+)", re.I)
    _P_TEMP_VAL     = re.compile(
        r"Temperature\s+([-\d.eE+]+)\s+([-\d.eE+]+)"
    )

    # Pressure
    _P_PRES_VAL     = re.compile(
        r"Pressure\s+\(bar\)\s+([-\d.eE+]+)"
    )

    # Performance
    _P_PERF         = re.compile(r"Performance:\s*([\d.]+)\s*ns/day")
    _P_STEP_TIME    = re.compile(r"^\s+Step\s+Time\s*$", re.M)
    _P_STEP_VAL     = re.compile(r"^\s+(\d+)\s+([\d.]+)\s*$", re.M)

    # Drift
    _P_DRIFT        = re.compile(
        r"Total energy drift.*?([\d.eE+\-]+)\s*kJ/mol/ps", re.I
    )

    # Error / warning flags
    _P_NAN          = re.compile(r"\bnan\b|\bnot a number\b", re.I)
    _P_LINCS_ERR    = re.compile(r"LINCS ERROR|Too many LINCS", re.I)
    _P_LINCS_WARN   = re.compile(r"LINCS WARNING|lincs_warnangle", re.I)
    _P_SETTLE       = re.compile(r"SETTLE.*error|error.*SETTLE", re.I)
    _P_ESCAPED      = re.compile(
        r"particle.*outside.*box|escaped.*box|flew away", re.I
    )
    _P_GPU_ERR      = re.compile(r"GPU.*error|CUDA.*error|OpenCL.*error", re.I)
    _P_MPI_ERR      = re.compile(r"MPI.*error|mpirun.*error", re.I)
    _P_DISK         = re.compile(r"no space left|disk full|write error", re.I)
    _P_MISSING_PAR  = re.compile(r"missing.*parameter|no.*parameters.*for", re.I)
    _P_CHARGE       = re.compile(r"net charge|total charge.*not.*zero", re.I)
    _P_TOPO_ERR     = re.compile(r"topology.*error|inconsistent.*topology", re.I)

    # Generic warning / error / note lines
    _P_WARNING      = re.compile(r"^.*\bWARNING\b.*$", re.M)
    _P_ERROR        = re.compile(r"^.*\b(ERROR|Fatal error)\b.*$", re.M)
    _P_NOTE         = re.compile(r"^.*\bNOTE\b.*$", re.M)

    # ------------------------------------------------------------------
    # Phase detection helper
    # ------------------------------------------------------------------

    @classmethod
    def _detect_phase(cls, text: str) -> SimulationPhase:
        m = cls._P_INTEGRATOR.search(text)
        if not m:
            return SimulationPhase.UNKNOWN
        integrator = m.group(1).lower()
        title_m = cls._P_TITLE.search(text)
        title = title_m.group(1).lower() if title_m else ""

        if integrator in ("steep", "cg", "l-bfgs"):
            return SimulationPhase.ENERGY_MIN
        if integrator in ("md", "md-vv", "sd", "bd"):
            if "nvt" in title or "npt" not in title and "equil" in title:
                return SimulationPhase.NVT_EQUIL
            if "npt" in title:
                return SimulationPhase.NPT_EQUIL
            if "prod" in title or "md" in title:
                return SimulationPhase.PRODUCTION_MD
            return SimulationPhase.PRODUCTION_MD  # default for dynamics
        return SimulationPhase.UNKNOWN

    # ------------------------------------------------------------------
    # Value extraction helpers
    # ------------------------------------------------------------------

    @classmethod
    def _extract_epot_series(cls, text: str) -> list[float]:
        """
        Extract all Potential Energy values from energy table blocks.
        GROMACS prints these in columnar format — we grab the first
        numeric token after a 'Potential' label in energy blocks.
        """
        values = []
        for m in cls._P_EPOT_BLOCK.finditer(text):
            try:
                values.append(float(m.group(1)))
            except ValueError:
                pass
        return values

    @classmethod
    def _extract_temperature_series(cls, text: str) -> list[float]:
        values = []
        for m in cls._P_TEMP_VAL.finditer(text):
            try:
                values.append(float(m.group(1)))
            except ValueError:
                pass
        return values

    @classmethod
    def _extract_pressure_series(cls, text: str) -> list[float]:
        values = []
        for m in cls._P_PRES_VAL.finditer(text):
            try:
                values.append(float(m.group(1)))
            except ValueError:
                pass
        return values

    @classmethod
    def _safe_float(cls, pattern: re.Pattern, text: str,
                    group: int = 1) -> float | None:
        m = pattern.search(text)
        if m:
            try:
                return float(m.group(group))
            except ValueError:
                pass
        return None

    @classmethod
    def _safe_int(cls, pattern: re.Pattern, text: str,
                  group: int = 1) -> int | None:
        m = pattern.search(text)
        if m:
            try:
                return int(m.group(group))
            except ValueError:
                pass
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def parse(cls, log_path: str | Path) -> LogMetrics:
        """
        Parse a GROMACS .log file and return a populated LogMetrics object.

        Args:
            log_path: Path to the .log file.

        Returns:
            LogMetrics with all extractable fields populated.
        """
        path = Path(log_path)
        if not path.exists():
            raise FileNotFoundError(f"Log file not found: {log_path}")

        text = path.read_text(errors="replace")
        m = LogMetrics()

        # Phase
        m.phase = cls._detect_phase(text)

        # EM convergence
        conv_match = cls._P_EM_CONV.search(text)
        if conv_match:
            m.em_converged    = True
            m.em_fmax_target  = float(conv_match.group(1))
            m.em_steps_taken  = int(conv_match.group(2))
        elif cls._P_EM_NOT_CONV.search(text):
            m.em_converged = False

        m.em_fmax_final  = cls._safe_float(cls._P_FMAX, text)
        m.em_steps_limit = cls._safe_int(cls._P_NSTEPS_LIMIT, text)

        if not m.em_fmax_target:
            m.em_fmax_target = cls._safe_float(cls._P_FMAX_TARGET, text)

        # Energies
        epot_series = cls._extract_epot_series(text)
        if epot_series:
            m.epot_values = epot_series
            m.epot_final  = epot_series[-1]

        # Also try direct Potential Energy = X pattern
        direct_epot = cls._safe_float(cls._P_EPOT_VALUE, text)
        if direct_epot is not None:
            m.epot_final = direct_epot

        # Temperature
        m.temperature_target = cls._safe_float(cls._P_TEMP_REF, text)
        temp_series = cls._extract_temperature_series(text)
        if temp_series:
            m.temperature_values = temp_series
            m.temperature_mean   = statistics.mean(temp_series)

        # Pressure
        pres_series = cls._extract_pressure_series(text)
        if pres_series:
            m.pressure_values = pres_series
            m.pressure_mean   = statistics.mean(pres_series)

        # Drift
        m.drift_kj_per_ns = cls._safe_float(cls._P_DRIFT, text)

        # Performance
        m.performance_ns_per_day = cls._safe_float(cls._P_PERF, text)

        # Last step
        step_vals = cls._P_STEP_VAL.findall(text)
        if step_vals:
            try:
                m.last_step = int(step_vals[-1][0])
            except (ValueError, IndexError):
                pass
        m.total_steps = cls._safe_int(cls._P_NSTEPS_LIMIT, text)

        # Boolean error flags
        m.has_nan             = bool(cls._P_NAN.search(text))
        m.has_lincs_error     = bool(cls._P_LINCS_ERR.search(text))
        m.has_lincs_warning   = bool(cls._P_LINCS_WARN.search(text))
        m.has_settle_error    = bool(cls._P_SETTLE.search(text))
        m.has_particle_escaped= bool(cls._P_ESCAPED.search(text))
        m.has_gpu_error       = bool(cls._P_GPU_ERR.search(text))
        m.has_mpi_error       = bool(cls._P_MPI_ERR.search(text))
        m.has_disk_full       = bool(cls._P_DISK.search(text))
        m.has_missing_params  = bool(cls._P_MISSING_PAR.search(text))
        m.has_charge_warning  = bool(cls._P_CHARGE.search(text))
        m.has_topology_error  = bool(cls._P_TOPO_ERR.search(text))

        # Text collections (deduplicated, capped at 20 each)
        m.warnings = list(dict.fromkeys(
            l.strip() for l in cls._P_WARNING.findall(text)
        ))[:20]
        m.errors = list(dict.fromkeys(
            l.strip() for l in cls._P_ERROR.findall(text)
        ))[:20]
        m.notes = list(dict.fromkeys(
            l.strip() for l in cls._P_NOTE.findall(text)
        ))[:10]

        return m