"""
Maps a Diagnosis → ActionRecommendation with concrete MDP patches.
No LLM involvement. Pure rule-based lookup table + logic.
"""

from __future__ import annotations

from .models import (
    ActionRecommendation,
    Diagnosis,
    DiagnosisCode,
    MDPPatch,
    RecoveryAction,
    Severity,
    SimulationPhase,
)


class RecoveryPlanner:
    """
    Stateless planner — call RecoveryPlanner.plan(diagnosis) → ActionRecommendation.
    """

    @classmethod
    def plan(cls, diagnosis: Diagnosis) -> ActionRecommendation:
        """
        Produce a fully specified ActionRecommendation from a Diagnosis.
        """
        m = diagnosis.metrics  # may be None

        # Dispatch table — ordered by DiagnosisCode
        dispatch = {
            DiagnosisCode.SUCCESS_CONVERGED:       cls._plan_success,
            DiagnosisCode.SUCCESS_COMPLETED:       cls._plan_success,
            DiagnosisCode.EM_NOT_CONVERGED:        cls._plan_em_not_converged,
            DiagnosisCode.EM_STEP_LIMIT_HIT:       cls._plan_em_step_limit,
            DiagnosisCode.EM_LINCS_ERROR:          cls._plan_lincs_error,
            DiagnosisCode.EM_EXPLODED:             cls._plan_explosion,
            DiagnosisCode.NAN_DETECTED:            cls._plan_nan,
            DiagnosisCode.EQUIL_TEMP_UNSTABLE:     cls._plan_temp_unstable,
            DiagnosisCode.EQUIL_PRESSURE_UNSTABLE: cls._plan_pressure_unstable,
            DiagnosisCode.EQUIL_DRIFT_TOO_HIGH:    cls._plan_drift,
            DiagnosisCode.LINCS_WARNING:           cls._plan_lincs_warning,
            DiagnosisCode.SETTLE_ERROR:            cls._plan_settle_error,
            DiagnosisCode.PARTICLE_ESCAPED_BOX:    cls._plan_particle_escaped,
            DiagnosisCode.MISSING_PARAMETERS:      cls._plan_missing_params,
            DiagnosisCode.CHARGE_IMBALANCE:        cls._plan_charge_imbalance,
            DiagnosisCode.TOPOLOGY_MISMATCH:       cls._plan_topology_error,
            DiagnosisCode.GPU_ERROR:               cls._plan_gpu_error,
            DiagnosisCode.MPI_ERROR:               cls._plan_escalate,
            DiagnosisCode.DISK_FULL:               cls._plan_escalate,
            DiagnosisCode.TIMEOUT:                 cls._plan_timeout,
            DiagnosisCode.UNKNOWN_ERROR:           cls._plan_escalate,
            DiagnosisCode.NEEDS_HUMAN_REVIEW:      cls._plan_escalate,
        }

        handler = dispatch.get(diagnosis.code, cls._plan_escalate)
        return handler(diagnosis)

    # ------------------------------------------------------------------
    # Individual planners
    # ------------------------------------------------------------------

    @classmethod
    def _plan_success(cls, d: Diagnosis) -> ActionRecommendation:
        return ActionRecommendation(
            diagnosis=d,
            primary_action=RecoveryAction.NONE,
            agent_instruction=(
                "Simulation completed successfully. "
                "Proceed to the next pipeline step."
            ),
        )

    @classmethod
    def _plan_em_not_converged(cls, d: Diagnosis) -> ActionRecommendation:
        m = d.metrics
        # Determine how far off Fmax is to scale the fix aggressiveness
        fmax = m.em_fmax_final if m else None
        target = (m.em_fmax_target if m else None) or 1000.0

        if fmax is not None and fmax > target * 100:
            # Very far from convergence — aggressive fix
            new_emstep = "0.0001"
            new_nsteps = "10000"
            reason_step = "Fmax >> target: aggressive step size reduction needed."
        elif fmax is not None and fmax > target * 10:
            new_emstep = "0.001"
            new_nsteps = "5000"
            reason_step = "Fmax > 10x target: moderate step size reduction."
        else:
            new_emstep = "0.005"
            new_nsteps = "2000"
            reason_step = "Fmax slightly above target: minor step size reduction."

        patches = [
            MDPPatch(
                parameter="emstep",
                old_value=None,
                new_value=new_emstep,
                reason=reason_step,
            ),
            MDPPatch(
                parameter="nsteps",
                old_value=str(m.em_steps_limit) if m and m.em_steps_limit else None,
                new_value=new_nsteps,
                reason="Increase step budget to allow convergence.",
            ),
        ]

        return ActionRecommendation(
            diagnosis=d,
            primary_action=RecoveryAction.REDUCE_EMSTEP_AND_INCREASE_NSTEPS,
            mdp_patches=patches,
            rerun_steps=["grompp (EM)", "mdrun (EM)"],
            agent_instruction=(
                f"Apply MDP patches to em.mdp, re-run grompp and mdrun for EM. "
                f"Target Fmax < {target} kJ/mol/nm."
            ),
            fallback_action=RecoveryAction.SWITCH_INTEGRATOR_SD,
            fallback_instruction=(
                "If steepest descent still fails after patch, "
                "switch integrator to 'sd' (stochastic dynamics) for EM."
            ),
        )

    @classmethod
    def _plan_em_step_limit(cls, d: Diagnosis) -> ActionRecommendation:
        m = d.metrics
        current_limit = m.em_steps_limit if m else 1000
        new_limit = str(current_limit * 5) if current_limit else "10000"

        patches = [
            MDPPatch(
                parameter="nsteps",
                old_value=str(current_limit),
                new_value=new_limit,
                reason="Step limit was hit before convergence — increase budget.",
            ),
        ]

        return ActionRecommendation(
            diagnosis=d,
            primary_action=RecoveryAction.INCREASE_NSTEPS,
            mdp_patches=patches,
            rerun_steps=["grompp (EM)", "mdrun (EM)"],
            agent_instruction=(
                f"Increase nsteps to {new_limit} in em.mdp and re-run EM. "
                "The minimiser hit the step limit before reaching Fmax target."
            ),
            fallback_action=RecoveryAction.REDUCE_EMSTEP_AND_INCREASE_NSTEPS,
            fallback_instruction=(
                "If increasing nsteps alone is insufficient, "
                "also reduce emstep to 0.001."
            ),
        )

    @classmethod
    def _plan_lincs_error(cls, d: Diagnosis) -> ActionRecommendation:
        patches = [
            MDPPatch(
                parameter="lincs-order",
                old_value="4",
                new_value="6",
                reason="Higher LINCS order improves constraint accuracy.",
            ),
            MDPPatch(
                parameter="lincs-iter",
                old_value="1",
                new_value="2",
                reason="More LINCS iterations for difficult constraint geometries.",
            ),
        ]

        return ActionRecommendation(
            diagnosis=d,
            primary_action=RecoveryAction.INCREASE_LINCS_ORDER,
            mdp_patches=patches,
            rerun_steps=["grompp", "mdrun"],
            agent_instruction=(
                "LINCS constraint error detected. "
                "Apply MDP patches to increase lincs-order and lincs-iter, "
                "then re-run grompp and mdrun."
            ),
            fallback_action=RecoveryAction.REDUCE_DT,
            fallback_instruction=(
                "If LINCS error persists, reduce dt from 0.002 to 0.001 ps "
                "to reduce per-step atomic displacement."
            ),
        )

    @classmethod
    def _plan_explosion(cls, d: Diagnosis) -> ActionRecommendation:
        patches = [
            MDPPatch(
                parameter="emstep",
                old_value=None,
                new_value="0.0001",
                reason="Explosion indicates severe clashes — very small step needed.",
            ),
            MDPPatch(
                parameter="nsteps",
                old_value=None,
                new_value="50000",
                reason="Large step budget needed to escape high-energy region.",
            ),
        ]

        return ActionRecommendation(
            diagnosis=d,
            primary_action=RecoveryAction.REDUCE_EMSTEP_AND_INCREASE_NSTEPS,
            mdp_patches=patches,
            rerun_steps=["grompp (EM)", "mdrun (EM)"],
            agent_instruction=(
                "System has exploded (Epot >> 0 or NaN). "
                "Apply aggressive emstep reduction. "
                "If this fails, the input structure likely has severe clashes "
                "requiring manual repair (missing atoms, overlapping residues)."
            ),
            fallback_action=RecoveryAction.ESCALATE_TO_USER,
            fallback_instruction=(
                "If explosion persists after emstep=0.0001, "
                "escalate to user: input PDB likely needs structural repair "
                "(e.g. Modeller, PDBFixer, or manual editing)."
            ),
        )

    @classmethod
    def _plan_nan(cls, d: Diagnosis) -> ActionRecommendation:
        patches = [
            MDPPatch(
                parameter="dt",
                old_value="0.002",
                new_value="0.001",
                reason="NaN often caused by too-large timestep.",
            ),
        ]

        return ActionRecommendation(
            diagnosis=d,
            primary_action=RecoveryAction.REDUCE_DT,
            mdp_patches=patches,
            rerun_steps=["grompp", "mdrun"],
            agent_instruction=(
                "NaN detected in forces/energies. "
                "Halve the timestep (dt) and re-run. "
                "If NaN appears in EM, switch to explosion recovery strategy."
            ),
            fallback_action=RecoveryAction.ESCALATE_TO_USER,
            fallback_instruction=(
                "If NaN persists after dt reduction, "
                "the structure likely needs repair before simulation."
            ),
        )

    @classmethod
    def _plan_temp_unstable(cls, d: Diagnosis) -> ActionRecommendation:
        patches = [
            MDPPatch(
                parameter="tau-t",
                old_value="0.1",
                new_value="0.5",
                reason=(
                    "Increase thermostat coupling time constant "
                    "to damp temperature oscillations."
                ),
            ),
        ]

        return ActionRecommendation(
            diagnosis=d,
            primary_action=RecoveryAction.REDUCE_DT,
            mdp_patches=patches,
            rerun_steps=["grompp (NVT)", "mdrun (NVT)"],
            agent_instruction=(
                "Temperature instability detected during equilibration. "
                "Increase tau-t to 0.5 ps and re-run NVT equilibration. "
                "Ensure prior EM converged properly before NVT."
            ),
            fallback_action=RecoveryAction.INCREASE_NSTEPS,
            fallback_instruction=(
                "If temperature is still drifting, extend NVT nsteps "
                "to allow longer equilibration time."
            ),
        )

    @classmethod
    def _plan_pressure_unstable(cls, d: Diagnosis) -> ActionRecommendation:
        patches = [
            MDPPatch(
                parameter="tau-p",
                old_value="2.0",
                new_value="5.0",
                reason=(
                    "Increase barostat coupling time constant "
                    "to stabilise pressure fluctuations."
                ),
            ),
            MDPPatch(
                parameter="compressibility",
                old_value="4.5e-5",
                new_value="4.5e-5",
                reason="Verify compressibility is set for water (4.5e-5 bar^-1).",
            ),
        ]

        return ActionRecommendation(
            diagnosis=d,
            primary_action=RecoveryAction.REDUCE_DT,
            mdp_patches=patches,
            rerun_steps=["grompp (NPT)", "mdrun (NPT)"],
            agent_instruction=(
                "Pressure instability detected during NPT equilibration. "
                "Increase tau-p to 5.0 ps. "
                "Ensure NVT equilibration was complete before NPT."
            ),
            fallback_action=RecoveryAction.INCREASE_NSTEPS,
            fallback_instruction=(
                "If pressure is still unstable, extend NPT nsteps "
                "and verify the barostat type (Parrinello-Rahman recommended)."
            ),
        )

    @classmethod
    def _plan_drift(cls, d: Diagnosis) -> ActionRecommendation:
        patches = [
            MDPPatch(
                parameter="nstlist",
                old_value="10",
                new_value="20",
                reason=(
                    "Increase neighbour list update frequency "
                    "to reduce energy drift."
                ),
            ),
        ]

        return ActionRecommendation(
            diagnosis=d,
            primary_action=RecoveryAction.INCREASE_NEIGHBOUR_FREQ,
            mdp_patches=patches,
            rerun_steps=["grompp", "mdrun"],
            agent_instruction=(
                "High energy drift detected. "
                "Increase nstlist to reduce drift. "
                "Also verify rlist, rcoulomb, rvdw are consistent."
            ),
        )

    @classmethod
    def _plan_lincs_warning(cls, d: Diagnosis) -> ActionRecommendation:
        patches = [
            MDPPatch(
                parameter="lincs-warnangle",
                old_value="30",
                new_value="45",
                reason="Suppress LINCS warnings for large-angle rotations.",
            ),
        ]

        return ActionRecommendation(
            diagnosis=d,
            primary_action=RecoveryAction.INCREASE_LINCS_ORDER,
            mdp_patches=patches,
            rerun_steps=["grompp", "mdrun"],
            agent_instruction=(
                "LINCS warning (not error) detected. "
                "Increase lincs-warnangle to suppress if warnings are expected "
                "for flexible regions. Monitor if warnings escalate to errors."
            ),
        )

    @classmethod
    def _plan_settle_error(cls, d: Diagnosis) -> ActionRecommendation:
        patches = [
            MDPPatch(
                parameter="dt",
                old_value="0.002",
                new_value="0.001",
                reason="SETTLE errors often caused by too-large timestep.",
            ),
        ]

        return ActionRecommendation(
            diagnosis=d,
            primary_action=RecoveryAction.REDUCE_DT,
            mdp_patches=patches,
            rerun_steps=["grompp", "mdrun"],
            agent_instruction=(
                "SETTLE error: water geometry constraints failed. "
                "Reduce dt to 0.001 ps and re-run. "
                "If error persists, verify water model matches force field."
            ),
            fallback_action=RecoveryAction.ESCALATE_TO_USER,
            fallback_instruction=(
                "If SETTLE error persists after dt reduction, "
                "check water model / force field compatibility."
            ),
        )

    @classmethod
    def _plan_particle_escaped(cls, d: Diagnosis) -> ActionRecommendation:
        patches = [
            MDPPatch(
                parameter="dt",
                old_value="0.002",
                new_value="0.001",
                reason="Reduce timestep to prevent large per-step displacements.",
            ),
            MDPPatch(
                parameter="nstcomm",
                old_value="100",
                new_value="10",
                reason=(
                    "More frequent centre-of-mass motion removal "
                    "to prevent drift."
                ),
            ),
        ]

        return ActionRecommendation(
            diagnosis=d,
            primary_action=RecoveryAction.REDUCE_DT,
            mdp_patches=patches,
            rerun_steps=["grompp", "mdrun"],
            agent_instruction=(
                "Particle(s) escaped the simulation box. "
                "Reduce dt and increase COM removal frequency. "
                "Ensure NPT equilibration was complete before production MD."
            ),
            fallback_action=RecoveryAction.RESUME_FROM_CHECKPOINT,
            fallback_instruction=(
                "If a checkpoint (.cpt) exists from before the escape event, "
                "resume from that checkpoint with reduced dt."
            ),
        )

    @classmethod
    def _plan_missing_params(cls, d: Diagnosis) -> ActionRecommendation:
        return ActionRecommendation(
            diagnosis=d,
            primary_action=RecoveryAction.REBUILD_TOPOLOGY,
            rerun_steps=["pdb2gmx", "grompp"],
            agent_instruction=(
                "Missing force field parameters detected. "
                "Options: (1) switch to a different force field in pdb2gmx, "
                "(2) add custom parameters to topol.top, "
                "(3) use a ligand parameterisation tool (e.g. ACPYPE). "
                "This requires user input to select the correct approach."
            ),
            fallback_action=RecoveryAction.ESCALATE_TO_USER,
            fallback_instruction=(
                "Escalate to user: missing parameters cannot be resolved "
                "automatically without knowing the molecule type."
            ),
        )

    @classmethod
    def _plan_charge_imbalance(cls, d: Diagnosis) -> ActionRecommendation:
        return ActionRecommendation(
            diagnosis=d,
            primary_action=RecoveryAction.RERUN_GENION,
            rerun_steps=["grompp (ions.mdp)", "genion", "grompp (EM)"],
            agent_instruction=(
                "Non-zero net charge detected. "
                "Re-run genion with -neutral flag to add counter-ions. "
                "Then re-run grompp for EM."
            ),
        )

    @classmethod
    def _plan_topology_error(cls, d: Diagnosis) -> ActionRecommendation:
        return ActionRecommendation(
            diagnosis=d,
            primary_action=RecoveryAction.ESCALATE_TO_USER,
            agent_instruction=(
                "Topology inconsistency detected — cannot recover automatically. "
                "User must inspect topol.top and .itp files for mismatches."
            ),
        )

    @classmethod
    def _plan_gpu_error(cls, d: Diagnosis) -> ActionRecommendation:
        return ActionRecommendation(
            diagnosis=d,
            primary_action=RecoveryAction.SWITCH_TO_CPU,
            rerun_steps=["mdrun (CPU only)"],
            agent_instruction=(
                "GPU error detected. "
                "Retry mdrun with use_gpu=False to fall back to CPU execution."
            ),
        )

    @classmethod
    def _plan_timeout(cls, d: Diagnosis) -> ActionRecommendation:
        return ActionRecommendation(
            diagnosis=d,
            primary_action=RecoveryAction.RESUME_FROM_CHECKPOINT,
            rerun_steps=["mdrun (from checkpoint)"],
            agent_instruction=(
                "Simulation timed out. "
                "Resume from the latest .cpt checkpoint file using "
                "mdrun with checkpoint_file parameter."
            ),
        )

    @classmethod
    def _plan_escalate(cls, d: Diagnosis) -> ActionRecommendation:
        return ActionRecommendation(
            diagnosis=d,
            primary_action=RecoveryAction.ESCALATE_TO_USER,
            agent_instruction=(
                f"Cannot automatically recover from {d.code.value}. "
                "Escalating to user for manual intervention. "
                f"Evidence: {'; '.join(d.evidence[:3])}"
            ),
        )