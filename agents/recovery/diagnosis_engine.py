"""
Rule-based diagnosis engine.
Maps LogMetrics → Diagnosis via a priority-ordered decision tree.
No LLM involvement.
"""

from __future__ import annotations

import statistics

from .models import (
    Diagnosis,
    DiagnosisCode,
    LogMetrics,
    Severity,
    SimulationPhase,
)

# Thresholds — centralised for easy tuning
_EPOT_EXPLOSION_THRESHOLD   =  1e6    # kJ/mol — clearly unphysical
_FMAX_ACCEPTABLE            =  1000.0 # kJ/mol/nm — default emtol
_TEMP_TOLERANCE_K           =  20.0   # K deviation from target
_TEMP_INSTABILITY_STD       =  15.0   # K std dev over trajectory
_PRES_TOLERANCE_BAR         =  200.0  # bar deviation (NPT)
_PRES_INSTABILITY_STD       =  150.0  # bar std dev
_DRIFT_WARN_THRESHOLD       =  10.0   # kJ/mol/ns — energy drift warning


class DiagnosisEngine:
    """
    Stateless engine — call DiagnosisEngine.diagnose(metrics) → Diagnosis.

    Rules are evaluated in priority order:
        1. Infrastructure failures (GPU, MPI, disk) — always fatal
        2. NaN / explosion — always fatal
        3. Topology / parameter errors — fatal
        4. LINCS / SETTLE errors — recoverable
        5. EM-specific convergence — recoverable
        6. Equilibration stability — recoverable
        7. Particle escape — recoverable
        8. Success states
        9. Unknown fallback
    """

    @classmethod
    def diagnose(cls, m: LogMetrics) -> Diagnosis:
        """
        Apply rule tree to LogMetrics and return a Diagnosis.
        """

        # ----------------------------------------------------------------
        # Priority 1: Infrastructure failures
        # ----------------------------------------------------------------
        if m.has_disk_full:
            return Diagnosis(
                code=DiagnosisCode.DISK_FULL,
                severity=Severity.FATAL,
                phase=m.phase,
                evidence=["Disk full / write error detected in log."],
                metrics=m,
            )

        if m.has_mpi_error:
            return Diagnosis(
                code=DiagnosisCode.MPI_ERROR,
                severity=Severity.FATAL,
                phase=m.phase,
                evidence=["MPI error detected — likely a cluster configuration issue."],
                metrics=m,
            )

        if m.has_gpu_error:
            return Diagnosis(
                code=DiagnosisCode.GPU_ERROR,
                severity=Severity.RECOVERABLE,
                phase=m.phase,
                evidence=["GPU/CUDA/OpenCL error detected."],
                metrics=m,
            )

        # ----------------------------------------------------------------
        # Priority 2: NaN / explosion
        # ----------------------------------------------------------------
        if m.has_nan:
            evidence = ["NaN detected in energies or forces."]
            if m.epot_final is not None:
                evidence.append(f"Final Epot = {m.epot_final:.4e} kJ/mol")
            return Diagnosis(
                code=DiagnosisCode.NAN_DETECTED,
                severity=Severity.RECOVERABLE,
                phase=m.phase,
                evidence=evidence,
                metrics=m,
            )

        if (m.epot_final is not None
                and m.epot_final > _EPOT_EXPLOSION_THRESHOLD):
            return Diagnosis(
                code=DiagnosisCode.EM_EXPLODED,
                severity=Severity.RECOVERABLE,
                phase=m.phase,
                evidence=[
                    f"Epot = {m.epot_final:.4e} kJ/mol "
                    f"(threshold: {_EPOT_EXPLOSION_THRESHOLD:.0e})",
                    "System likely has severe clashes — structure needs repair.",
                ],
                metrics=m,
            )

        # ----------------------------------------------------------------
        # Priority 3: Topology / parameter errors
        # ----------------------------------------------------------------
        if m.has_topology_error:
            return Diagnosis(
                code=DiagnosisCode.TOPOLOGY_MISMATCH,
                severity=Severity.FATAL,
                phase=m.phase,
                evidence=["Topology inconsistency detected in log."],
                metrics=m,
            )

        if m.has_missing_params:
            return Diagnosis(
                code=DiagnosisCode.MISSING_PARAMETERS,
                severity=Severity.ASSISTED,
                phase=m.phase,
                evidence=["Missing force field parameters for one or more atoms."],
                metrics=m,
            )

        if m.has_charge_warning:
            return Diagnosis(
                code=DiagnosisCode.CHARGE_IMBALANCE,
                severity=Severity.ASSISTED,
                phase=m.phase,
                evidence=["Non-zero net charge detected — genion step may be needed."],
                metrics=m,
            )

        # ----------------------------------------------------------------
        # Priority 4: LINCS / SETTLE
        # ----------------------------------------------------------------
        if m.has_lincs_error:
            evidence = ["LINCS ERROR: bond constraints could not be satisfied."]
            if m.last_step is not None:
                evidence.append(f"Failed at step {m.last_step}.")
            return Diagnosis(
                code=DiagnosisCode.EM_LINCS_ERROR,
                severity=Severity.RECOVERABLE,
                phase=m.phase,
                evidence=evidence,
                metrics=m,
            )

        if m.has_settle_error:
            return Diagnosis(
                code=DiagnosisCode.SETTLE_ERROR,
                severity=Severity.RECOVERABLE,
                phase=m.phase,
                evidence=[
                    "SETTLE error: water geometry constraints failed.",
                    "Often caused by too-large timestep or bad initial structure.",
                ],
                metrics=m,
            )

        if m.has_lincs_warning:
            # LINCS warning is secondary — continue checking but record it
            lincs_secondary = [DiagnosisCode.LINCS_WARNING]
        else:
            lincs_secondary = []

        # ----------------------------------------------------------------
        # Priority 5: Particle escaped box
        # ----------------------------------------------------------------
        if m.has_particle_escaped:
            return Diagnosis(
                code=DiagnosisCode.PARTICLE_ESCAPED_BOX,
                severity=Severity.RECOVERABLE,
                phase=m.phase,
                evidence=[
                    "Particle(s) escaped the simulation box.",
                    "Likely cause: too-large timestep or insufficient equilibration.",
                ],
                secondary_codes=lincs_secondary,
                metrics=m,
            )

        # ----------------------------------------------------------------
        # Priority 6: EM-specific convergence
        # ----------------------------------------------------------------
        if m.phase == SimulationPhase.ENERGY_MIN:
            return cls._diagnose_em(m, lincs_secondary)

        # ----------------------------------------------------------------
        # Priority 7: Equilibration / production stability
        # ----------------------------------------------------------------
        if m.phase in (
            SimulationPhase.NVT_EQUIL,
            SimulationPhase.NPT_EQUIL,
            SimulationPhase.PRODUCTION_MD,
        ):
            return cls._diagnose_dynamics(m, lincs_secondary)

        # ----------------------------------------------------------------
        # Priority 8: Generic success (phase unknown but no errors)
        # ----------------------------------------------------------------
        if not m.errors and not m.has_nan:
            return Diagnosis(
                code=DiagnosisCode.SUCCESS_COMPLETED,
                severity=Severity.OK,
                phase=m.phase,
                evidence=["No errors detected. Simulation appears complete."],
                secondary_codes=lincs_secondary,
                metrics=m,
            )

        # ----------------------------------------------------------------
        # Priority 9: Unknown fallback
        # ----------------------------------------------------------------
        return Diagnosis(
            code=DiagnosisCode.UNKNOWN_ERROR,
            severity=Severity.NEEDS_HUMAN_REVIEW,
            phase=m.phase,
            evidence=m.errors[:5] or ["No specific error pattern matched."],
            secondary_codes=lincs_secondary,
            metrics=m,
        )

    # ------------------------------------------------------------------
    # Sub-diagnosers
    # ------------------------------------------------------------------

    @classmethod
    def _diagnose_em(
        cls, m: LogMetrics, secondary: list
    ) -> Diagnosis:
        """EM-specific convergence diagnosis."""

        # Explicit convergence flag
        if m.em_converged is True:
            return Diagnosis(
                code=DiagnosisCode.SUCCESS_CONVERGED,
                severity=Severity.OK,
                phase=m.phase,
                evidence=[
                    f"EM converged: Fmax = {m.em_fmax_final} "
                    f"< {m.em_fmax_target} kJ/mol/nm "
                    f"in {m.em_steps_taken} steps.",
                    f"Final Epot = {m.epot_final:.4e} kJ/mol"
                    if m.epot_final else "Epot not extracted.",
                ],
                secondary_codes=secondary,
                metrics=m,
            )

        if m.em_converged is False:
            evidence = ["EM did not converge (explicit log statement)."]
            code = DiagnosisCode.EM_NOT_CONVERGED

            # Distinguish step-limit from genuine non-convergence
            if (m.em_steps_taken is not None
                    and m.em_steps_limit is not None
                    and m.em_steps_taken >= m.em_steps_limit):
                code = DiagnosisCode.EM_STEP_LIMIT_HIT
                evidence.append(
                    f"Step limit reached: {m.em_steps_taken} / {m.em_steps_limit}."
                )

            if m.em_fmax_final is not None:
                evidence.append(
                    f"Final Fmax = {m.em_fmax_final:.4e} kJ/mol/nm "
                    f"(target: {m.em_fmax_target or _FMAX_ACCEPTABLE})."
                )
            if m.epot_final is not None:
                evidence.append(f"Final Epot = {m.epot_final:.4e} kJ/mol.")

            return Diagnosis(
                code=code,
                severity=Severity.RECOVERABLE,
                phase=m.phase,
                evidence=evidence,
                secondary_codes=secondary,
                metrics=m,
            )

        # Fmax available but no explicit convergence statement
        if m.em_fmax_final is not None:
            target = m.em_fmax_target or _FMAX_ACCEPTABLE
            if m.em_fmax_final <= target:
                return Diagnosis(
                    code=DiagnosisCode.SUCCESS_CONVERGED,
                    severity=Severity.OK,
                    phase=m.phase,
                    evidence=[
                        f"Fmax = {m.em_fmax_final:.4e} <= {target} kJ/mol/nm "
                        "(inferred convergence)."
                    ],
                    secondary_codes=secondary,
                    metrics=m,
                )
            else:
                return Diagnosis(
                    code=DiagnosisCode.EM_NOT_CONVERGED,
                    severity=Severity.RECOVERABLE,
                    phase=m.phase,
                    evidence=[
                        f"Fmax = {m.em_fmax_final:.4e} > {target} kJ/mol/nm "
                        "(inferred non-convergence)."
                    ],
                    secondary_codes=secondary,
                    metrics=m,
                )

        # Cannot determine EM outcome
        return Diagnosis(
            code=DiagnosisCode.UNKNOWN_ERROR,
            severity=Severity.NEEDS_HUMAN_REVIEW,
            phase=m.phase,
            evidence=["EM phase detected but convergence status undetermined."],
            secondary_codes=secondary,
            metrics=m,
        )

    @classmethod
    def _diagnose_dynamics(
        cls, m: LogMetrics, secondary: list
    ) -> Diagnosis:
        """NVT / NPT / production MD stability diagnosis."""

        issues: list[str] = []
        codes: list[DiagnosisCode] = list(secondary)

        # Temperature check
        if (m.temperature_target is not None
                and m.temperature_mean is not None):
            delta_t = abs(m.temperature_mean - m.temperature_target)
            if delta_t > _TEMP_TOLERANCE_K:
                issues.append(
                    f"Temperature deviation: mean={m.temperature_mean:.1f} K, "
                    f"target={m.temperature_target:.1f} K, "
                    f"delta={delta_t:.1f} K > {_TEMP_TOLERANCE_K} K threshold."
                )
                codes.append(DiagnosisCode.EQUIL_TEMP_UNSTABLE)

        if len(m.temperature_values) > 5:
            t_std = statistics.stdev(m.temperature_values)
            if t_std > _TEMP_INSTABILITY_STD:
                issues.append(
                    f"Temperature std dev = {t_std:.1f} K "
                    f"> {_TEMP_INSTABILITY_STD} K (unstable thermostat)."
                )
                if DiagnosisCode.EQUIL_TEMP_UNSTABLE not in codes:
                    codes.append(DiagnosisCode.EQUIL_TEMP_UNSTABLE)

        # Pressure check (NPT only)
        if m.phase == SimulationPhase.NPT_EQUIL:
            if (m.pressure_mean is not None
                    and abs(m.pressure_mean) > _PRES_TOLERANCE_BAR):
                issues.append(
                    f"Pressure mean = {m.pressure_mean:.1f} bar "
                    f"(|deviation| > {_PRES_TOLERANCE_BAR} bar)."
                )
                codes.append(DiagnosisCode.EQUIL_PRESSURE_UNSTABLE)

            if len(m.pressure_values) > 5:
                p_std = statistics.stdev(m.pressure_values)
                if p_std > _PRES_INSTABILITY_STD:
                    issues.append(
                        f"Pressure std dev = {p_std:.1f} bar "
                        f"> {_PRES_INSTABILITY_STD} bar."
                    )
                    if DiagnosisCode.EQUIL_PRESSURE_UNSTABLE not in codes:
                        codes.append(DiagnosisCode.EQUIL_PRESSURE_UNSTABLE)

        # Energy drift
        if (m.drift_kj_per_ns is not None
                and m.drift_kj_per_ns > _DRIFT_WARN_THRESHOLD):
            issues.append(
                f"Energy drift = {m.drift_kj_per_ns:.2f} kJ/mol/ns "
                f"> {_DRIFT_WARN_THRESHOLD} kJ/mol/ns."
            )
            codes.append(DiagnosisCode.EQUIL_DRIFT_TOO_HIGH)

        # Determine primary code and severity
        if not issues and not m.errors:
            return Diagnosis(
                code=DiagnosisCode.SUCCESS_COMPLETED,
                severity=Severity.OK,
                phase=m.phase,
                evidence=[
                    "Dynamics simulation completed without detected issues.",
                    f"Performance: {m.performance_ns_per_day} ns/day"
                    if m.performance_ns_per_day else "",
                ],
                secondary_codes=[c for c in codes if c != DiagnosisCode.LINCS_WARNING],
                metrics=m,
            )

        # Pick the most severe primary code
        primary = codes[0] if codes else DiagnosisCode.UNKNOWN_ERROR
        return Diagnosis(
            code=primary,
            severity=Severity.RECOVERABLE,
            phase=m.phase,
            evidence=issues or m.errors[:3],
            secondary_codes=codes[1:],
            metrics=m,
        )