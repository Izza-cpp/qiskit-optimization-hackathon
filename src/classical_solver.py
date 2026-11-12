"""Exact classical baseline: enumerate all C(10,3) = 120 feasible selections.

INDEPENDENCE: this module deliberately does NOT import the quantum modules and
does NOT use the numpy score/penalty helpers of ``objective``. It re-computes F
for every combination with plain Python loops straight from the raw data, so a
bug in the vectorised objective or in the QUBO would NOT silently propagate here.
The test-suite cross-checks both implementations.

NOTE (honest limitation): for the LINEAR models A and B the exact optimum is
simply the K largest scores, so a classical solver is trivially optimal. Model C
(pairwise overlap) is a genuine quadratic problem. Exhaustive enumeration is
used here because 120 candidates is tiny.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .objective import Formulation
from .problem import MedicalDeploymentProblem, RISK_SCALE

TIE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class ClassicalResult:
    """Outcome of exhaustive enumeration."""

    num_candidates: int                       # must be C(n, K)
    best_value: float                         # F* (maximum)
    optimal_sets: Tuple[Tuple[int, ...], ...]  # ALL selections within TIE_TOLERANCE of F*
    chosen_set: Tuple[int, ...]               # deterministic pick: lexicographically smallest optimum
    all_values: Dict[Tuple[int, ...], float] = field(repr=False, default_factory=dict)
    ranking: List[Tuple[Tuple[int, ...], float]] = field(repr=False, default_factory=list)

    @property
    def has_tie(self) -> bool:
        return len(self.optimal_sets) > 1

    def optimal_bitstrings_x(self, n: int) -> List[Tuple[int, ...]]:
        """Optimal selections as binary vectors x."""
        return [tuple(1 if i in s else 0 for i in range(n)) for s in self.optimal_sets]


def _direct_objective(problem: MedicalDeploymentProblem, form: Formulation,
                      chosen: Tuple[int, ...]) -> float:
    """Plain-Python F(chosen): sum of scores minus pairwise overlap penalty."""
    w = form.weights
    max_p = max(problem.patients)
    max_t = max(problem.travel_minutes)
    total = 0.0
    for i in chosen:
        total += w.patients * problem.patients[i] / max_p
        total += form.outbreak_weight * problem.outbreak_risk[i] / RISK_SCALE
        total -= w.travel * problem.travel_minutes[i] / max_t
    if w.overlap > 0:
        overlap = problem.overlap_matrix()
        for i, j in itertools.combinations(chosen, 2):
            total -= w.overlap * overlap[i, j]
    return float(total)


def solve_exact(problem: MedicalDeploymentProblem, form: Formulation) -> ClassicalResult:
    """Evaluate every feasible selection and return the exact optimum(s)."""
    values: Dict[Tuple[int, ...], float] = {}
    for chosen in itertools.combinations(range(problem.n), problem.num_teams):
        values[chosen] = _direct_objective(problem, form, chosen)
    best = max(values.values())
    optimal = tuple(sorted(c for c, v in values.items() if v >= best - TIE_TOLERANCE))
    ranking = sorted(values.items(), key=lambda kv: (-kv[1], kv[0]))
    return ClassicalResult(
        num_candidates=len(values), best_value=best, optimal_sets=optimal,
        chosen_set=optimal[0], all_values=values, ranking=ranking,
    )
