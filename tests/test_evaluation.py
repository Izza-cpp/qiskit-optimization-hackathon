import math

import numpy as np
import pytest

from src.classical_solver import ClassicalResult, solve_exact
from src.evaluation import aggregate, random_baseline, sample_metrics
from src.objective import make_formulation, objective_value
from src.problem import default_problem
from src.constraints import selection_to_bitstring

P = default_problem()
F = make_formulation("B")
C = solve_exact(P, F)


def bits(*one_based):
    x = [0] * 10
    for i in one_based:
        x[i - 1] = 1
    return selection_to_bitstring(x)


def test_metrics_mixed_counts():
    counts = {bits(1, 3, 5): 40,     # exact optimum (feasible)
              bits(1, 3, 9): 30,     # feasible, suboptimal
              bits(1, 3): 20,        # infeasible: 2 teams
              bits(1, 2, 3, 4): 10}  # infeasible: 4 teams
    m = sample_metrics(P, F, C, counts)
    assert m["total_samples"] == 100 and m["feasible_samples"] == 70 and m["infeasible_samples"] == 30
    assert m["feasibility_rate"] == pytest.approx(0.7)
    assert m["optimal_samples"] == 40 and m["optimal_probability"] == pytest.approx(0.4)
    assert m["found_exact_optimum"] and m["best_feasible_selection"] == ["L1", "L3", "L5"]
    assert m["optimality_gap_abs"] == pytest.approx(0.0, abs=1e-12)


def test_gap_when_optimum_not_sampled():
    m = sample_metrics(P, F, C, {bits(1, 3, 9): 10})
    expected = C.best_value - objective_value(P, F, np.array([1, 0, 1, 0, 0, 0, 0, 0, 1, 0]))
    assert not m["found_exact_optimum"]
    assert m["optimality_gap_abs"] == pytest.approx(expected) and expected > 0
    assert 0 < m["optimality_gap_range"] < 1
    assert m["optimality_gap_rel"] == pytest.approx(expected / C.best_value)


def test_no_feasible_sample_is_reported_not_hidden():
    m = sample_metrics(P, F, C, {bits(1, 3): 50, bits(): 50})
    assert m["found_feasible"] is False and m["feasibility_rate"] == 0.0
    assert math.isnan(m["optimality_gap_abs"]) and math.isnan(m["optimality_gap_range"])
    assert m["total_samples"] == 100 and m["best_feasible_selection"] is None


def test_relative_gap_is_safe_for_zero_reference():
    zero = ClassicalResult(num_candidates=2, best_value=0.0, optimal_sets=((0, 1, 2),), chosen_set=(0, 1, 2),
                           all_values={(0, 1, 2): 0.0, (0, 1, 3): -1.0}, ranking=[])
    m = sample_metrics(P, F, zero, {bits(1, 2, 4): 5})
    assert math.isnan(m["optimality_gap_rel"])                 # no division by ~0
    assert not math.isnan(m["optimality_gap_range"])           # range-normalised gap still defined


def test_best_measured_can_be_infeasible():
    table = np.zeros(1024)
    table[int(bits(1, 3), 2)] = -5.0          # pretend a 2-team string has the lowest energy
    m = sample_metrics(P, F, C, {bits(1, 3): 1, bits(1, 3, 5): 1}, energy_table=table)
    assert m["best_measured_bitstring"] == bits(1, 3) and m["best_measured_is_feasible"] is False
    assert m["best_feasible_selection"] == ["L1", "L3", "L5"]


def test_random_control_matches_theory():
    uniform = random_baseline(P, F, C, shots=200_000, seed=0)
    assert uniform["feasibility_rate"] == pytest.approx(120 / 1024, abs=0.003)
    feas = random_baseline(P, F, C, shots=120_000, seed=0, feasible_only=True)
    assert feas["feasibility_rate"] == 1.0
    assert feas["optimal_probability"] == pytest.approx(1 / 120, abs=0.0015)


def test_aggregate_statistics():
    rows = [dict(g="a", v=1.0, found_feasible=True, found_exact_optimum=True),
            dict(g="a", v=3.0, found_feasible=True, found_exact_optimum=False),
            dict(g="b", v=5.0, found_feasible=False, found_exact_optimum=False)]
    out = aggregate(rows, ["g"], ["v"]).set_index("g")
    assert out.loc["a", "v_mean"] == 2.0 and out.loc["a", "v_min"] == 1.0 and out.loc["a", "v_max"] == 3.0
    assert out.loc["a", "runs"] == 2 and out.loc["a", "runs_found_exact_optimum"] == 1
    assert out.loc["b", "runs_without_feasible"] == 1
