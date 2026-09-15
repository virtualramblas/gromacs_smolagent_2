"""
ui/results_writer.py

Provides ResultsWriter — called by the agent runner to push
simulation results into UIState so the Results tab updates live.

Usage (in ui/runner.py):
    writer = ResultsWriter(ui_state)

    # After EM completes:
    writer.record_em_results(fmax=987.6, epot=-456789.0)

    # After NVT completes:
    writer.record_nvt_results(temp_mean=300.1, temp_values=[...])

    # After NPT completes:
    writer.record_npt_results(pres_mean=1.2, pres_values=[...])

    # After energy_analysis tool returns:
    writer.record_energy_data({"Potential": [...], "Kinetic-En.": [...]})

    # After rmsd_analysis tool returns:
    writer.record_rmsd_data([(0.0, 0.01), (10.0, 0.12), ...])

    # After final_answer:
    writer.record_final_answer("PIPELINE COMPLETE ...")
"""

from __future__ import annotations

import re
from pathlib import Path

from ui.state import UIState


class ResultsWriter:
    """
    Thread-safe helper for pushing simulation results into UIState.
    """

    def __init__(self, ui_state: UIState):
        self.ui_state = ui_state

    # ── EM ────────────────────────────────────────────────────────────────

    def record_em_results(
        self,
        fmax:      float | None = None,
        epot:      float | None = None,
        converged: bool  | None = None,
    ) -> None:
        with self.ui_state.lock:
            if fmax      is not None:
                self.ui_state.em_fmax = fmax
            if epot      is not None:
                self.ui_state.em_epot = epot
            if converged is not None:
                self.ui_state.em_converged_flag = converged
        self.ui_state.append_log(
            f"RESULT  | EM: Fmax={fmax:.1f} kJ/mol/nm, "
            f"Epot={epot/1000:.1f} kJ/mol, "
            f"converged={converged}"
        )

    # ── NVT ───────────────────────────────────────────────────────────────

    def record_nvt_results(
        self,
        temp_mean:   float | None       = None,
        temp_values: list[float] | None = None,
    ) -> None:
        with self.ui_state.lock:
            if temp_mean   is not None:
                self.ui_state.nvt_temp_mean = temp_mean
            if temp_values is not None:
                self.ui_state.nvt_complete_flag = True
        self.ui_state.append_log(
            f"RESULT  | NVT: mean temperature = {temp_mean:.1f} K"
        )

    # ── NPT ───────────────────────────────────────────────────────────────

    def record_npt_results(
        self,
        pres_mean:   float | None       = None,
        pres_values: list[float] | None = None,
    ) -> None:
        with self.ui_state.lock:
            if pres_mean   is not None:
                self.ui_state.npt_pres_mean = pres_mean
            if pres_values is not None:
                self.ui_state.npt_complete_flag = True
        self.ui_state.append_log(
            f"RESULT  | NPT: mean pressure = {pres_mean:.1f} bar"
        )

    # ── Energy data ───────────────────────────────────────────────────────

    def record_energy_data(
        self,
        energy_dict: dict[str, list[float]],
    ) -> None:
        """
        Store energy term time series for plotting.

        Args:
            energy_dict: {term_name: [value, value, ...]}
                         e.g. {"Potential": [-456789, -456800, ...]}
        """
        with self.ui_state.lock:
            self.ui_state.energy_data.update(energy_dict)
        terms = list(energy_dict.keys())
        self.ui_state.append_log(
            f"RESULT  | Energy data recorded: {terms}"
        )

    # ── RMSD data ─────────────────────────────────────────────────────────

    def record_rmsd_data(
        self,
        rmsd_data: list[tuple[float, float]],
    ) -> None:
        """
        Store RMSD time series for plotting.

        Args:
            rmsd_data: [(time_ps, rmsd_nm), ...]
        """
        with self.ui_state.lock:
            self.ui_state.rmsd_data = rmsd_data
            self.ui_state.md_complete_flag = True
        if rmsd_data:
            final_rmsd = rmsd_data[-1][1]
            self.ui_state.append_log(
                f"RESULT  | RMSD data recorded: "
                f"{len(rmsd_data)} frames, "
                f"final RMSD = {final_rmsd:.3f} nm"
            )

    # ── Final answer ──────────────────────────────────────────────────────

    def record_final_answer(self, answer: str) -> None:
        with self.ui_state.lock:
            self.ui_state.final_answer = answer
        self.ui_state.append_log(
            f"RESULT  | Final answer recorded "
            f"({len(answer)} chars)"
        )

    # ── XVG parser ────────────────────────────────────────────────────────

    @staticmethod
    def parse_xvg(xvg_path: str) -> tuple[list[float], list[float]]:
        """
        Parse a GROMACS .xvg file into (x_values, y_values).
        Skips comment lines starting with # or @.

        Returns:
            (x_values, y_values) — both as lists of floats.
        """
        x_vals: list[float] = []
        y_vals: list[float] = []

        path = Path(xvg_path)
        if not path.exists():
            return x_vals, y_vals

        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("@"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    x_vals.append(float(parts[0]))
                    y_vals.append(float(parts[1]))
                except ValueError:
                    continue

        return x_vals, y_vals

    @staticmethod
    def parse_energy_xvg(
        xvg_path: str,
        term_name: str = "Potential",
    ) -> list[float]:
        """
        Parse a single energy term from an .xvg file.
        Returns list of y-values (energy in kJ/mol).
        """
        _, y_vals = ResultsWriter.parse_xvg(xvg_path)
        return y_vals

    @staticmethod
    def parse_rmsd_xvg(xvg_path: str) -> list[tuple[float, float]]:
        """
        Parse an RMSD .xvg file.
        Returns list of (time_ps, rmsd_nm) tuples.
        """
        x_vals, y_vals = ResultsWriter.parse_xvg(xvg_path)
        return list(zip(x_vals, y_vals))