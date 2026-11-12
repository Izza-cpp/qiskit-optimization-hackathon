from fractions import Fraction as Fr
import itertools

import numpy as np
import pytest

from src.objective import (Formulation, location_scores, make_formulation, normalised, objective_breakdown,
                           objective_value)
from src.problem import Weights, default_problem

P = default_problem()


def x_of(*one_based):
    x = np.zeros(10, dtype=int)
    for i in one_based:
        x[i - 1] = 1
    return x


def test_normalisation_ranges():
    nrm = normalised(P)
    assert nrm["P"].max() == 1.0 and nrm["T"].max() == 1.0
    assert nrm["R"].max() == 0.95
    assert all(v.min() >= 0 for v in nrm.values())


def test_hand_calculation_model_B_L1_L3_L5():
    # w = (0.5, 0.3, 0.2); P/110, R/100, T/60. Exact rational arithmetic, no floats.
    def s(p, r, t):
        return Fr(1, 2) * Fr(p, 110) + Fr(3, 10) * Fr(r, 100) - Fr(1, 5) * Fr(t, 60)
    expected = s(80, 90, 20) + s(110, 85, 50) + s(70, 75, 25)
    got = objective_value(P, make_formulation("B"), x_of(1, 3, 5))
    assert got == pytest.approx(float(expected), abs=1e-12)
    assert got == pytest.approx(1.6151515151515, abs=1e-9)


def test_model_A_drops_only_the_outbreak_term():
    a, b = make_formulation("A"), make_formulation("B")
    x = x_of(1, 3, 7)
    diff = objective_value(P, b, x) - objective_value(P, a, x)
    expected = 0.3 * (90 + 85 + 30) / 100
    assert diff == pytest.approx(expected, abs=1e-12)


def test_model_C_subtracts_pairwise_overlap():
    b, c = make_formulation("B"), make_formulation("C")
    x = x_of(1, 3, 5)
    o = P.overlap_matrix()
    pairs = o[0, 2] + o[0, 4] + o[2, 4]
    assert objective_value(P, b, x) - objective_value(P, c, x) == pytest.approx(0.3 * pairs, abs=1e-12)
    assert pairs > 0.5


def test_direction_benefit_up_cost_down():
    """More patients / more risk must RAISE a location's score; longer travel must LOWER it."""
    from dataclasses import replace
    f = make_formulation("B")
    base = location_scores(P, f)
    # location index 3 (L4) is not the max of any column, so normalisers stay unchanged
    more_patients = P.patients.copy(); more_patients[3] += 5
    more_risk = P.outbreak_risk.copy(); more_risk[3] += 5
    longer_travel = P.travel_minutes.copy(); longer_travel[3] += 5
    assert location_scores(replace(P, patients=more_patients), f)[3] > base[3]
    assert location_scores(replace(P, outbreak_risk=more_risk), f)[3] > base[3]
    assert location_scores(replace(P, travel_minutes=longer_travel), f)[3] < base[3]


def test_objective_value_matches_scores_for_linear_models():
    for model in "AB":
        f = make_formulation(model)
        s = location_scores(P, f)
        for combo in itertools.islice(itertools.combinations(range(10), 3), 30):
            x = np.zeros(10, dtype=int)
            x[list(combo)] = 1
            assert objective_value(P, f, x) == pytest.approx(s[list(combo)].sum(), abs=1e-12)


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        objective_value(P, make_formulation("B"), [1, 0, 1])
    with pytest.raises(ValueError):
        objective_value(P, make_formulation("B"), [2] + [0] * 9)
    with pytest.raises(ValueError):
        make_formulation("Z")
    with pytest.raises(ValueError):
        Formulation("C", Weights(0.5, 0.3, 0.2, 0.0))   # model C needs overlap > 0
    with pytest.raises(ValueError):
        Formulation("B", Weights(0.5, 0.3, 0.2, 0.1))   # model B must not have overlap


def test_breakdown_raw_units():
    d = objective_breakdown(P, make_formulation("B"), x_of(1, 3, 5))
    assert d["selected"] == ["L1", "L3", "L5"]
    assert d["patients_covered"] == 80 + 110 + 70
    assert d["total_travel_min"] == 20 + 50 + 25
    assert d["outbreak_risk_sum"] == 90 + 85 + 75
