"""Compare QAOA (or any sampler) against the exact classical baseline.

Definitions (all reported, none hidden):
  feasibility_rate          feasible samples / all samples
  best_measured             lowest-ENERGY sampled bitstring (may be infeasible!)
  best_feasible             feasible sampled bitstring with the highest objective F
  exact optimum             from ``classical_solver`` (independent of the quantum code)
  optimal_probability       fraction of ALL samples that equal an exact optimum
  optimality_gap_abs        F* - F(best_feasible)                          (>= 0)
  optimality_gap_range      gap divided by (F* - F_worst_feasible), in [0, 1]; safe when F* <= 0
  optimality_gap_rel        gap / |F*|, reported only if |F*| > REL_GAP_MIN_REFERENCE else NaN
A run with no feasible sample is reported with ``found_feasible = False`` and NaN gaps; it is
never dropped.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .classical_solver import ClassicalResult
from .constraints import bitstring_to_selection, is_feasible
from .objective import Formulation, objective_value
from .problem import MedicalDeploymentProblem
from .qaoa_solver import QaoaResult
from .quantum_model import all_selections

REL_GAP_MIN_REFERENCE = 1e-6


def _selection_set(x) -> tuple:
    return tuple(int(i) for i in np.flatnonzero(x))


def sample_metrics(problem: MedicalDeploymentProblem, form: Formulation, classical: ClassicalResult,
                   counts: Dict[str, int], energy_table: Optional[np.ndarray] = None) -> dict:
    """Metrics for a dictionary of Qiskit measurement counts."""
    n = problem.n
    total = int(sum(counts.values()))
    feasible_samples = 0
    optimal_samples = 0
    best_feasible_x, best_feasible_f = None, -math.inf
    best_measured_bits, best_measured_energy = None, math.inf
    optimal_sets = set(classical.optimal_sets)
    for bits, c in counts.items():
        x = bitstring_to_selection(bits, n)
        feasible = is_feasible(x, problem.num_teams)
        if energy_table is not None:
            e = float(energy_table[int(bits.replace(" ", ""), 2)])
            if e < best_measured_energy:
                best_measured_energy, best_measured_bits = e, bits
        if feasible:
            feasible_samples += c
            if _selection_set(x) in optimal_sets:
                optimal_samples += c
            f = objective_value(problem, form, x)
            if f > best_feasible_f:
                best_feasible_f, best_feasible_x = f, x

    found = best_feasible_x is not None
    f_star = classical.best_value
    f_worst = min(classical.all_values.values())
    gap_abs = max(0.0, f_star - best_feasible_f) if found else math.nan  # clip float round-off
    spread = f_star - f_worst
    gap_range = (gap_abs / spread) if (found and spread > 0) else math.nan
    gap_rel = (gap_abs / abs(f_star)) if (found and abs(f_star) > REL_GAP_MIN_REFERENCE) else math.nan
    names = problem.names
    return {
        "total_samples": total,
        "feasible_samples": feasible_samples,
        "infeasible_samples": total - feasible_samples,
        "feasibility_rate": feasible_samples / total,
        "optimal_samples": optimal_samples,
        "optimal_probability": optimal_samples / total,
        "found_feasible": found,
        "best_feasible_selection": [names[i] for i in np.flatnonzero(best_feasible_x)] if found else None,
        "best_feasible_objective": best_feasible_f if found else math.nan,
        "found_exact_optimum": bool(found and _selection_set(best_feasible_x) in optimal_sets),
        "optimality_gap_abs": gap_abs,
        "optimality_gap_range": gap_range,
        "optimality_gap_rel": gap_rel,
        "best_measured_bitstring": best_measured_bits,
        "best_measured_is_feasible": (is_feasible(bitstring_to_selection(best_measured_bits, n), problem.num_teams)
                                      if best_measured_bits else None),
    }


def evaluate_run(problem: MedicalDeploymentProblem, form: Formulation, classical: ClassicalResult,
                 result: QaoaResult, model_label: str = "") -> dict:
    """One flat, machine-readable record (CSV/JSON friendly) for a QAOA run."""
    from .quantum_model import build_qubo, hamiltonian_diagonal, ising_to_operator, qubo_to_ising
    table = hamiltonian_diagonal(ising_to_operator(qubo_to_ising(build_qubo(problem, form, result.penalty))))
    row = sample_metrics(problem, form, classical, result.counts, table)

    x_all = all_selections(problem.n)
    feasible_mask = x_all.sum(axis=1) == problem.num_teams
    optimal_idx = [sum(1 << i for i in s) for s in classical.optimal_sets]
    row.update({
        "model": model_label or form.model,
        "mixer": result.config.mixer, "depth": result.config.depth, "shots": result.config.shots,
        "seed": result.config.seed, "penalty": result.penalty, "init": result.config.init,
        "optimizer": result.config.optimizer, "max_iterations": result.config.max_iterations,
        "rhobeg": result.config.rhobeg, "tolerance": result.config.tolerance,
        "exact_optimum": [problem.names[i] for i in classical.chosen_set],
        "exact_optimum_value": classical.best_value,
        "ideal_feasibility": float(result.ideal_probabilities[feasible_mask].sum()),
        "ideal_optimal_probability": float(result.ideal_probabilities[optimal_idx].sum()),
        "ideal_energy": result.ideal_energy, "sampled_energy": result.final_sampled_energy,
        "evaluations_stage": result.num_evaluations, "evaluations_cumulative": result.cumulative_evaluations,
        "runtime_total_s": result.total_seconds, "runtime_transpile_s": result.transpile_seconds,
        "runtime_optimize_s": result.optimize_seconds, "runtime_final_sampling_s": result.final_sampling_seconds,
        "runtime_warmup_s": result.warmup_seconds,
        "circuit_depth": result.circuit_depth,
        "two_qubit_gates": sum(v for k, v in result.circuit_gate_counts.items() if k in ("rzz", "cx", "rxx", "ryy")),
        "initial_params": [float(v) for v in result.initial_params],
        "optimal_params": [float(v) for v in result.optimal_params],
    })
    return row


def random_baseline(problem: MedicalDeploymentProblem, form: Formulation, classical: ClassicalResult,
                    shots: int, seed: int, feasible_only: bool = False) -> dict:
    """CONTROL, not a quantum method: uniform random bitstrings (or random feasible selections).

    Shows what "no optimisation at all" achieves with the same number of samples.
    """
    rng = np.random.default_rng(seed)
    n = problem.n
    counts: Dict[str, int] = {}
    if feasible_only:
        sets = list(classical.all_values.keys())
        for idx in rng.integers(0, len(sets), shots):
            bits = "".join("1" if i in sets[idx] else "0" for i in reversed(range(n)))
            counts[bits] = counts.get(bits, 0) + 1
    else:
        for k in rng.integers(0, 2 ** n, shots):
            bits = format(int(k), f"0{n}b")
            counts[bits] = counts.get(bits, 0) + 1
    row = sample_metrics(problem, form, classical, counts)
    row.update({"method": "random_feasible" if feasible_only else "random_uniform", "shots": shots, "seed": seed})
    return row


def aggregate(rows: Iterable[dict], by: List[str], metrics: List[str]) -> pd.DataFrame:
    """mean / std / min / max of selected metrics, grouped by configuration columns.

    ``std`` uses ddof=0 (population std over the seeds that were run). Rows with NaN gaps (runs
    with no feasible sample) are kept in the counts via ``runs`` and ``runs_without_feasible``.
    """
    df = pd.DataFrame(list(rows))
    grouped = df.groupby(by, dropna=False)
    out = grouped[metrics].agg(["mean", "std", "min", "max"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out["runs"] = grouped.size()
    out["runs_without_feasible"] = grouped["found_feasible"].apply(lambda s: int((~s).sum()))
    out["runs_found_exact_optimum"] = grouped["found_exact_optimum"].apply(lambda s: int(s.sum()))
    return out.reset_index()