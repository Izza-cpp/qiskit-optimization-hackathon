import numpy as np
import pytest

from src.problem import MedicalDeploymentProblem, Weights, default_problem, TRAVEL_SPEED_KM_PER_MIN


def _kwargs(**over):
    base = dict(names=("a", "b", "c"), patients=np.array([1., 2., 3.]), travel_minutes=np.array([1., 2., 3.]),
                outbreak_risk=np.array([10., 20., 30.]), coordinates_km=np.zeros((3, 2)), num_teams=2)
    base.update(over)
    return base


def test_default_problem_shape_and_teams():
    p = default_problem()
    assert p.n == 10 and p.num_teams == 3
    assert p.patients.shape == p.travel_minutes.shape == p.outbreak_risk.shape == (10,)


def test_coordinates_consistent_with_travel_time():
    p = default_problem()
    dist = np.linalg.norm(p.coordinates_km, axis=1)
    assert np.allclose(dist / TRAVEL_SPEED_KM_PER_MIN, p.travel_minutes)


def test_wrong_length_rejected():
    with pytest.raises(ValueError):
        MedicalDeploymentProblem(**_kwargs(patients=np.array([1., 2.])))


def test_negative_and_nan_rejected():
    with pytest.raises(ValueError):
        MedicalDeploymentProblem(**_kwargs(patients=np.array([1., -2., 3.])))
    with pytest.raises(ValueError):
        MedicalDeploymentProblem(**_kwargs(travel_minutes=np.array([1., np.nan, 3.])))


def test_risk_above_scale_rejected():
    with pytest.raises(ValueError):
        MedicalDeploymentProblem(**_kwargs(outbreak_risk=np.array([10., 20., 101.])))


def test_too_many_teams_and_duplicate_names_rejected():
    with pytest.raises(ValueError):
        MedicalDeploymentProblem(**_kwargs(num_teams=4))
    with pytest.raises(ValueError):
        MedicalDeploymentProblem(**_kwargs(names=("a", "a", "c")))


def test_weights_must_sum_to_one_and_be_nonnegative():
    Weights(0.5, 0.3, 0.2)
    with pytest.raises(ValueError):
        Weights(0.5, 0.3, 0.3)
    with pytest.raises(ValueError):
        Weights(1.2, -0.1, -0.1)


def test_overlap_matrix_properties():
    o = default_problem().overlap_matrix()
    assert np.allclose(o, o.T) and np.allclose(np.diag(o), 0)
    assert o.min() >= 0 and o.max() <= 1
    assert o.max() > 0, "some pairs must overlap or model C is pointless"
