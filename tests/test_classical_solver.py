import itertools
import math
import inspect

import numpy as np
import pytest

import src.classical_solver as cs
from src.classical_solver import solve_exact
from src.objective import location_scores, make_formulation, objective_value
from src.problem import MedicalDeploymentProblem, default_problem
from src.quantum_model import all_selections

P = default_problem()


def names(result):
    return [P.names[i] for i in result.chosen_set]


def test_enumerates_exactly_120():
    r = solve_exact(P, make_formulation("B"))
    assert r.num_candidates == math.comb(10, 3) == 120
    assert all(len(c) == 3 for c in r.all_values)


@pytest.mark.parametrize("model,expected", [("A", ["L1", "L3", "L7"]), ("B", ["L1", "L3", "L5"]),
                                            ("C", ["L1", "L3", "L9"])])
def test_known_optima(model, expected):
    r = solve_exact(P, make_formulation(model))
    assert names(r) == expected and not r.has_tie


def test_three_models_choose_different_deployments():
    sets = {m: solve_exact(P, make_formulation(m)).chosen_set for m in "ABC"}
    assert len(set(sets.values())) == 3


def test_agrees_with_independent_vectorised_objective_over_all_1024():
    """Cross-check two independent implementations; infeasible states must not beat the optimum."""
    for model in "ABC":
        f = make_formulation(model)
        r = solve_exact(P, f)
        feasible_best = -math.inf
        for x in all_selections(10):
            if x.sum() == 3:
                feasible_best = max(feasible_best, objective_value(P, f, x))
        assert r.best_value == pytest.approx(feasible_best, abs=1e-12)


def test_linear_models_equal_top_k_scores():
    for model in "AB":
        f = make_formulation(model)
        top3 = tuple(sorted(np.argsort(-location_scores(P, f))[:3]))
        assert solve_exact(P, f).chosen_set == top3


def test_ranking_sorted_and_best_first():
    r = solve_exact(P, make_formulation("C"))
    values = [v for _, v in r.ranking]
    assert values == sorted(values, reverse=True)
    assert r.ranking[0][0] == r.chosen_set


def test_ties_are_reported_and_resolved_deterministically():
    twins = MedicalDeploymentProblem(
        names=("a", "b", "c", "d"), patients=np.array([10., 10., 10., 1.]),
        travel_minutes=np.array([5., 5., 5., 5.]), outbreak_risk=np.array([50., 50., 50., 0.]),
        coordinates_km=np.zeros((4, 2)), num_teams=2)
    r = solve_exact(twins, make_formulation("B"))
    assert r.has_tie and len(r.optimal_sets) == 3
    assert r.chosen_set == (0, 1)       # lexicographically smallest


def test_module_does_not_import_quantum_code():
    src = inspect.getsource(cs)
    for forbidden in ("qiskit", "qaoa_solver", "quantum_model"):
        assert f"import {forbidden}" not in src and f"from {forbidden}" not in src
        assert f"from .{forbidden}" not in src


def test_overlap_term_makes_sorting_wrong_but_linear_models_are_sortable():
    """Why this is a real optimisation problem: with the overlap term, 'top-3 by score' is NOT optimal."""
    fc = make_formulation("C")
    top3 = tuple(sorted(int(i) for i in np.argsort(-location_scores(P, fc))[:3]))
    best = solve_exact(P, fc)
    assert top3 != best.chosen_set
    x = np.zeros(10, dtype=int); x[list(top3)] = 1
    assert objective_value(P, fc, x) < best.best_value - 0.05      # clearly suboptimal, not a rounding tie
    for model in "AB":                                             # without overlap, sorting IS exact
        f = make_formulation(model)
        assert tuple(sorted(int(i) for i in np.argsort(-location_scores(P, f))[:3])) == solve_exact(P, f).chosen_set
