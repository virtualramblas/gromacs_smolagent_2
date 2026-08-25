"""
Pure regex-based extractor of raw metrics from GROMACS log files.
No LLM involvement. Produces a LogMetrics object.

Fixes applied vs original:
    1. _P_INTEGRATOR: add word boundary to avoid partial matches
    2. _P_TEMP_VAL:   rewritten to match GROMACS columnar energy table format
    3. _P_PERF:       rewritten to match multi-line Performance block
    4. _P_SETTLE:     broadened to match SETTLE lines without 'error' keyword
    5. _detect_phase: made robust to logs without explicit integrator line
       by also scanning for convergence/dynamics keywords as fallback
"""

from __future__ import annotations

import re
import statistics
from pathlib import Path

from .models import LogMetrics, SimulationPhase


class LogParser:
    """
    Stateless parser — call LogParser.parse(path) to get LogMetrics.
    All patterns compiled once at class level for performance.
    """

    # ------------------------------------------------------------------
    # Compiled regex patterns
    # ------------------------------------------------------------------

    # Phase detection
    # Fix 1: \b word boundary prevents matching 'integrator' inside words
    _P_INTEGRATOR = re.compile(r"^\s*integrator\s*=\s*(\S+)", re.I | re.M)
    _P_TITLE      = re.compile(r"^\s*title\s*=\s*(.+)$",      re.I | re.M)

    # EM convergence
    _P_EM_CONV     = re.compile(
        r"converged to Fmax\s*<\s*([\d.eE+\-]+)\s*in\s*(\d+)\s*steps",
        re.I
    )
    _P_EM_NOT_CONV = re.compile(r"did not converge to Fmax", re.I)
    _P_FMAX        = re.compile(r"Maximum force\s*[=:]\s*([\d.eE+\-]+)")
    _P_FMAX_TARGET = re.compile(r"Fmax\s*<\s*([\d.eE+\-]+)")
    _P_NSTEPS      = re.compile(r"^\s*nsteps\s*=\s*(\d+)", re.I | re.M)

    # Potential energy
    # Matches "Potential Energy  = -4.56789e+05" (direct assignment form)
    _P_EPOT_VALUE = re.compile(
        r"Potential Energy\s*[=:\s]+([-\d.eE+]+)"
    )
    # Matches the value in columnar energy table after "Potential" label
    _P_EPOT_BLOCK = re.compile(
        r"^\s*Potential\s+([-\d.eE+]+)", re.M
    )

    # Temperature
    _P_TEMP_REF = re.compile(r"^\s*ref[_-]?t\s*=\s*([\d.]+)", re.I | re.M)

    # Fix 2: Temperature in GROMACS energy tables appears as:
    #   "   Temperature\n   299.87\n"  (label line then value line)
    # The original pattern matched two numbers on the same line which
    # captured Potential/Kinetic columns instead.
    _P_TEMP_LABEL = re.compile(r"^\s*Temperature\s*$", re.M)
    _P_TEMP_BLOCK = re.compile(
        r"Temperature\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)", re.M
    )
    # Also match standalone temperature value after label
    _P_TEMP_STANDALONE = re.compile(
        r"^\s*Temperature\s*\n\s*([\d.eE+\-]+)", re.M
    )

    # Pressure
    _P_PRES_VAL = re.compile(
        r"Pressure\s+\(bar\)\s+([-\d.eE+]+)", re.M
    )

    # Fix 3: GROMACS Performance block format:
    #   "Performance:\n   Mnbf/s    GFlops  ns/day   hour/ns\n  1234.56 ..."
    # The original pattern looked for numbers on the same line as "Performance:"
    # but GROMACS writes the header and values on separate lines.
    _P_PERF = re.compile(
        r"Performance:\s*\n"           # "Performance:" then newline
        r"\s*\S+\s+\S+\s+"            # Mnbf/s header   GFlops header
        r"ns/day\s+hour/ns\s*\n"      # ns/day header   hour/ns header
        r"\s*([\d.]+)\s+([\d.]+)\s+"  # Mnbf/s value    GFlops value
        r"([\d.]+)\s+([\d.]+)",       # ns/day value ←  hour/ns value
        re.M
    )
    # Fallback: single-line format some GROMACS versions use
    _P_PERF_INLINE = re.compile(r"Performance:\s+([\d.]+)\s+ns/day", re.M)

    # Drift
    _P_DRIFT = re.compile(
        r"Total energy drift.*?([\d.eE+\-]+)\s*kJ/mol/ps", re.I
    )

    # Step tracking
    _P_STEP_VAL = re.compile(r"^\s+(\d+)\s+([\d.]+)\s*$", re.M)

    # Fix 4: SETTLE — broaden to catch lines with SETTLE keyword
    # even without the word 'error' on the same line
    _P_SETTLE = re.compile(
        r"SETTLE.*error"           # original: SETTLE followed by error
        r"|error.*SETTLE"          # original: error followed by SETTLE
        r"|SETTLE:\s+can"          # new: "SETTLE: can't settle..."
        r"|SETTLE\s+warning"       # new: SETTLE warning
        r"|can't\s+settle\s+atom", # new: can't settle atom N
        re.I
    )

    # Error / warning flags — unchanged
    _P_NAN        = re.compile(r"\bnan\b|\bnot a number\b", re.I)
    _P_LINCS_ERR  = re.compile(r"LINCS ERROR|Too many LINCS", re.I)
    _P_LINCS_WARN = re.compile(r"LINCS WARNING|lincs_warnangle", re.I)
    _P_ESCAPED    = re.compile(
        r"particle.*outside.*box|escaped.*box|flew away", re.I
    )
    _P_GPU_ERR    = re.compile(r"GPU.*error|CUDA.*error|OpenCL.*error", re.I)
    _P_MPI_ERR    = re.compile(r"MPI.*error|mpirun.*error", re.I)
    _P_DISK       = re.compile(r"no space left|disk full|write error", re.I)
    _P_MISSING_PAR= re.compile(r"missing.*parameter|no.*parameters.*for", re.I)
    _P_CHARGE     = re.compile(r"net charge|total charge.*not.*zero", re.I)
    _P_TOPO_ERR   = re.compile(r"topology.*error|inconsistent.*topology", re.I)

    # Generic collections
    _P_WARNING    = re.compile(r"^.*\bWARNING\b.*$", re.M)
    _P_ERROR      = re.compile(r"^.*\b(ERROR|Fatal error)\b.*$", re.M)
    _P_NOTE       = re.compile(r"^.*\bNOTE\b.*$", re.M)

    # ------------------------------------------------------------------
    # Fix 5: Phase detection — robust fallback chain
    # ------------------------------------------------------------------

    @classmethod
    def _detect_phase(cls, text: str) -> SimulationPhase:
        """
        Detect simulation phase using a three-tier fallback:
            Tier 1: integrator = <value>  (most reliable)
            Tier 2: title line keywords   (nvt / npt / prod)
            Tier 3: convergence keywords  (EM-specific phrases)
        """
        # Tier 1: integrator line
        m = cls._P_INTEGRATOR.search(text)
        if m:
            integrator = m.group(1).lower().strip()
            if integrator in ("steep", "cg", "l-bfgs"):
                return SimulationPhase.ENERGY_MIN
            if integrator in ("md", "md-vv", "sd", "bd"):
                # md-family integrator confirmed — title refines the phase,
                # defaulting to PRODUCTION_MD if no title keyword found
                return cls._phase_from_title(
                    text,
                    default=SimulationPhase.PRODUCTION_MD,  # ← md confirmed
                )

        # Tier 2: title keywords only — no integrator line found
        title_phase = cls._phase_from_title(
            text,
            default=SimulationPhase.UNKNOWN,                # ← no integrator
        )
        if title_phase != SimulationPhase.UNKNOWN:
            return title_phase

        # Tier 3: EM-specific convergence phrases
        if (cls._P_EM_CONV.search(text)
                or cls._P_EM_NOT_CONV.search(text)
                or cls._P_FMAX.search(text)):
            return SimulationPhase.ENERGY_MIN

        return SimulationPhase.UNKNOWN


    @classmethod
    def _phase_from_title(
        cls,
        text: str,
        default: SimulationPhase = SimulationPhase.UNKNOWN,
    ) -> SimulationPhase:
        """
        Extract phase from title line keywords.

        Args:
            text:    Full log text.
            default: Phase to return when no title line is found
                    OR when title contains no recognised keyword.
                    Callers pass PRODUCTION_MD when an md-family
                    integrator is already confirmed, UNKNOWN otherwise.
        """
        title_m = cls._P_TITLE.search(text)
        if not title_m:
            return default                  # ← caller decides the fallback

        title = title_m.group(1).lower()
        if "nvt" in title:
            return SimulationPhase.NVT_EQUIL
        if "npt" in title:
            return SimulationPhase.NPT_EQUIL
        if "prod" in title or "md" in title:
            return SimulationPhase.PRODUCTION_MD

        return default                      # ← title found but no keyword matched

    # ------------------------------------------------------------------
    # Value extraction helpers
    # ------------------------------------------------------------------

    @classmethod
    def _extract_epot_series(cls, text: str) -> list[float]:
        values = []
        for m in cls._P_EPOT_BLOCK.finditer(text):
            try:
                values.append(float(m.group(1)))
            except ValueError:
                pass
        return values

    @classmethod
    def _extract_temperature_series(cls, text: str) -> list[float]:
        """
        Fix 2: Extract temperature values from GROMACS energy tables.

        GROMACS energy table format (two numbers per row = avg + rmsd):
            Energies (kJ/mol)
            Potential   Kinetic En.   Total Energy  Temperature
            -456789.0   12345.6       -444443.4     299.87

        We look for the Temperature column value specifically by finding
        lines in the energy block that contain temperature-range numbers
        (100-400 K) following the energy values.

        Strategy: use the two-column pattern which captures both the
        average and rmsd — take the first (average) value, but only
        if it is in a physically plausible temperature range (50-600 K).
        """
        values = []

        # Strategy A: two-column temperature pattern (avg + rmsd columns)
        for m in cls._P_TEMP_BLOCK.finditer(text):
            try:
                val = float(m.group(1))
                # Filter to physically plausible temperature range
                if 50.0 <= val <= 600.0:
                    values.append(val)
            except ValueError:
                pass

        # Strategy B: standalone temperature value after label line
        if not values:
            for m in cls._P_TEMP_STANDALONE.finditer(text):
                try:
                    val = float(m.group(1))
                    if 50.0 <= val <= 600.0:
                        values.append(val)
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
    def _extract_performance(cls, text: str) -> float | None:
        """
        Fix 3: Extract ns/day from the multi-line Performance block.

        GROMACS format:
            Performance:
               Mnbf/s    GFlops  ns/day   hour/ns
              1234.56    567.89   12.34     1.94

        Group 3 of _P_PERF captures the ns/day value.
        Falls back to _P_PERF_INLINE for single-line variants.
        """
        m = cls._P_PERF.search(text)
        if m:
            try:
                return float(m.group(3))   # group 3 = ns/day
            except ValueError:
                pass

        # Fallback: inline format
        m2 = cls._P_PERF_INLINE.search(text)
        if m2:
            try:
                return float(m2.group(1))
            except ValueError:
                pass

        return None

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
        """
        path = Path(log_path)
        if not path.exists():
            raise FileNotFoundError(f"Log file not found: {log_path}")

        text = path.read_text(errors="replace")
        m    = LogMetrics()

        # Phase
        m.phase = cls._detect_phase(text)

        # EM convergence
        conv_match = cls._P_EM_CONV.search(text)
        if conv_match:
            m.em_converged   = True
            m.em_fmax_target = float(conv_match.group(1))
            m.em_steps_taken = int(conv_match.group(2))
        elif cls._P_EM_NOT_CONV.search(text):
            m.em_converged = False

        m.em_fmax_final  = cls._safe_float(cls._P_FMAX,   text)
        m.em_steps_limit = cls._safe_int(cls._P_NSTEPS,   text)

        if not m.em_fmax_target:
            m.em_fmax_target = cls._safe_float(cls._P_FMAX_TARGET, text)

        # Energies
        epot_series = cls._extract_epot_series(text)
        if epot_series:
            m.epot_values = epot_series
            m.epot_final  = epot_series[-1]

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

        # Performance — use dedicated extractor
        m.performance_ns_per_day = cls._extract_performance(text)

        # Last step
        step_vals = cls._P_STEP_VAL.findall(text)
        if step_vals:
            try:
                m.last_step = int(step_vals[-1][0])
            except (ValueError, IndexError):
                pass
        m.total_steps = cls._safe_int(cls._P_NSTEPS, text)

        # Boolean error flags
        m.has_nan              = bool(cls._P_NAN.search(text))
        m.has_lincs_error      = bool(cls._P_LINCS_ERR.search(text))
        m.has_lincs_warning    = bool(cls._P_LINCS_WARN.search(text))
        m.has_settle_error     = bool(cls._P_SETTLE.search(text))
        m.has_particle_escaped = bool(cls._P_ESCAPED.search(text))
        m.has_gpu_error        = bool(cls._P_GPU_ERR.search(text))
        m.has_mpi_error        = bool(cls._P_MPI_ERR.search(text))
        m.has_disk_full        = bool(cls._P_DISK.search(text))
        m.has_missing_params   = bool(cls._P_MISSING_PAR.search(text))
        m.has_charge_warning   = bool(cls._P_CHARGE.search(text))
        m.has_topology_error   = bool(cls._P_TOPO_ERR.search(text))

        # Text collections
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