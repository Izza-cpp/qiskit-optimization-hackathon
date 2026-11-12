"""Objective function (MAXIMISATION convention).

    F(x) = sum_i s_i x_i  -  w_O * sum_{i<j} O_ij x_i x_j

    s_i  = w_P * P_i/max(P)  +  w_R * R_i/100  -  w_T * T_i/max(T)

Benefits (patients, outbreak risk) add; cost (travel time) subtracts. Each
quantity is normalised to ~[0, 1] first so that units (people, score, minutes)
can be combined. The last term (model C only, w_O > 0) penalises sending two
teams to locations whose service areas overlap. With w_O = 0 the objective is
linear in x.

Higher F is better. The quantum module converts this to an energy H = -F + penalty.

Three named formulations are used in the project:
    "A"  patient need and travel time only            (outbreak weight ignored)
    "B"  outbreak-aware                                (as in the strategy document)
    "C"  B plus pairwise service-area overlap penalty  (the extension)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

from .problem import MedicalDeploymentProblem, RISK_SCALE, Weights

MODELS = ("A", "B", "C")
DEFAULT_WEIGHTS_AB = Weights(patients=0.5, outbreak=0.3, travel=0.2, overlap=0.0)
DEFAULT_OVERLAP_WEIGHT = 0.3


def normalised(problem: MedicalDeploymentProblem) -> Dict[str, np.ndarray]:
    """Normalised inputs: P/max(P), R/100, T/max(T)."""
    return {
        "P": problem.patients / np.max(problem.patients),
        "R": problem.outbreak_risk / RISK_SCALE,
        "T": problem.travel_minutes / np.max(problem.travel_minutes),
    }


@dataclass(frozen=True)
class Formulation:
    """A fully specified objective: which model + which weights."""

    model: str = "B"
    weights: Weights = DEFAULT_WEIGHTS_AB

    def __post_init__(self) -> None:
        if self.model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}")
        if self.model == "C" and self.weights.overlap <= 0:
            raise ValueError("model C needs weights.overlap > 0")
        if self.model in ("A", "B") and self.weights.overlap != 0:
            raise ValueError("models A and B must have weights.overlap == 0")

    @property
    def outbreak_weight(self) -> float:
        return 0.0 if self.model == "A" else self.weights.outbreak


def make_formulation(model: str, weights: Weights = DEFAULT_WEIGHTS_AB,
                     overlap: float = DEFAULT_OVERLAP_WEIGHT) -> Formulation:
    """Build a Formulation for model A, B or C from base weights."""
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {model!r}")
    w = weights.with_(overlap=overlap if model == "C" else 0.0)
    return Formulation(model=model, weights=w)


def location_scores(problem: MedicalDeploymentProblem, form: Formulation) -> np.ndarray:
    """Per-location linear score s_i (benefit minus travel cost)."""
    nrm = normalised(problem)
    w = form.weights
    return w.patients * nrm["P"] + form.outbreak_weight * nrm["R"] - w.travel * nrm["T"]


def pair_penalties(problem: MedicalDeploymentProblem, form: Formulation) -> np.ndarray:
    """Symmetric matrix W_ij = w_O * O_ij (zero diagonal); zero unless model C."""
    if form.weights.overlap == 0:
        return np.zeros((problem.n, problem.n))
    return form.weights.overlap * problem.overlap_matrix()


def objective_value(problem: MedicalDeploymentProblem, form: Formulation,
                    x: Sequence[int]) -> float:
    """F(x) for a binary vector x (higher is better). Does NOT check cardinality."""
    xv = np.asarray(x, dtype=float)
    if xv.shape != (problem.n,):
        raise ValueError(f"x must have length {problem.n}")
    if not np.all((xv == 0) | (xv == 1)):
        raise ValueError("x must be binary")
    s = location_scores(problem, form)
    w = pair_penalties(problem, form)
    return float(s @ xv - 0.5 * xv @ w @ xv)  # 0.5 because W is symmetric


def objective_breakdown(problem: MedicalDeploymentProblem, form: Formulation,
                        x: Sequence[int]) -> Dict[str, float]:
    """Interpretable components for a selection (raw units + the weighted F)."""
    xv = np.asarray(x, dtype=int)
    idx = np.flatnonzero(xv)
    w = pair_penalties(problem, form)
    overlap_raw = float(sum(problem.overlap_matrix()[i, j] for a, i in enumerate(idx) for j in idx[a + 1:]))
    return {
        "selected": [problem.names[i] for i in idx],
        "patients_covered": float(problem.patients[idx].sum()),
        "outbreak_risk_sum": float(problem.outbreak_risk[idx].sum()),
        "total_travel_min": float(problem.travel_minutes[idx].sum()),
        "overlap_sum": overlap_raw,
        "objective_F": objective_value(problem, form, xv),
    }
