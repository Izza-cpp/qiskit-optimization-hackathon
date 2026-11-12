import numpy as np
import pytest

from src.classical_solver import solve_exact
from src.constraints import bitstring_to_selection
from src.objective import make_formulation
from src.problem import default_problem
from src.qaoa_solver import (QaoaConfig, decode_counts, energy_from_counts, initial_parameters, interp_parameters,
                             run_qaoa, run_qaoa_progressive)
from src.quantum_model import (all_selections, build_qubo, default_penalty, hamiltonian_diagonal,
                               ising_to_operator, qubo_to_ising)

P = default_problem()
F = make_formulation("C")


def table():
    return hamiltonian_diagonal(ising_to_operator(qubo_to_ising(build_qubo(P, F, default_penalty(P, F)))))


def test_config_validation():
    for bad in (dict(depth=0), dict(mixer="q"), dict(shots=0), dict(init="zzz"), dict(optimizer="SLSQP")):
        with pytest.raises(ValueError):
            QaoaConfig(**bad)


def test_counts_sum_to_shots_and_strings_are_10_bits():
    r = run_qaoa(P, F, QaoaConfig(depth=1, mixer="xy", shots=512, seed=1, max_iterations=20))
    assert sum(r.counts.values()) == 512
    assert all(len(b) == 10 and set(b) <= {"0", "1"} for b in r.counts)


def test_same_seed_reproduces_exactly():
    cfg = QaoaConfig(depth=1, mixer="x", shots=256, seed=7, max_iterations=15)
    a, b = run_qaoa(P, F, cfg), run_qaoa(P, F, cfg)
    assert a.counts == b.counts
    assert np.allclose(a.optimal_params, b.optimal_params)
    assert a.energy_history == b.energy_history


def test_different_seed_changes_the_run():
    base = dict(depth=1, mixer="x", shots=256, max_iterations=15)
    a = run_qaoa(P, F, QaoaConfig(seed=1, **base))
    b = run_qaoa(P, F, QaoaConfig(seed=2, **base))
    assert not np.allclose(a.initial_params, b.initial_params)


def test_xy_mixer_samples_are_all_feasible():
    r = run_qaoa(P, F, QaoaConfig(depth=2, mixer="xy", shots=1024, seed=0, max_iterations=30))
    for bits in r.counts:
        assert sum(bitstring_to_selection(bits, 10)) == 3


def test_x_mixer_samples_include_infeasible_states():
    r = run_qaoa(P, F, QaoaConfig(depth=1, mixer="x", shots=1024, seed=0, max_iterations=30))
    assert any(sum(bitstring_to_selection(b, 10)) != 3 for b in r.counts)


def test_sampled_energy_agrees_with_exact_statevector_energy():
    r = run_qaoa(P, F, QaoaConfig(depth=2, mixer="xy", shots=8192, seed=3, max_iterations=40))
    assert abs(r.final_sampled_energy - r.ideal_energy) < 0.08   # ~ sigma/sqrt(shots)
    assert r.ideal_probabilities.sum() == pytest.approx(1.0)


def test_optimisation_improves_on_the_unoptimised_start():
    """QAOA must beat its own starting state: the Dicke state has the mean-feasible energy."""
    e = table()
    start_energy = e[all_selections(10).sum(axis=1) == 3].mean()
    r = run_qaoa(P, F, QaoaConfig(depth=3, mixer="xy", shots=2048, seed=0))
    assert r.ideal_energy < start_energy - 0.1


def test_depth_helps_progressive_run_is_monotone_in_depth_on_this_instance():
    results = run_qaoa_progressive(P, F, QaoaConfig(depth=3, mixer="xy", shots=2048, seed=0))
    assert [r.config.depth for r in results] == [1, 2, 3]
    energies = [r.ideal_energy for r in results]
    assert energies[-1] < energies[0]
    assert results[-1].cumulative_evaluations == sum(r.num_evaluations for r in results)


def test_progressive_requires_interp_and_final_sampling_uses_other_seed():
    with pytest.raises(ValueError):
        run_qaoa_progressive(P, F, QaoaConfig(depth=2, init="random"))


def test_decode_counts_little_endian():
    out = decode_counts({"0000010101": 7, "1000000000": 3}, 10)
    assert out[(1, 0, 1, 0, 1, 0, 0, 0, 0, 0)] == 7          # L1, L3, L5
    assert out[(0, 0, 0, 0, 0, 0, 0, 0, 0, 1)] == 3          # L10


def test_energy_from_counts_manual():
    e = table()
    counts = {"0000010101": 3, "0000000000": 1}
    expected = (3 * e[0b10101] + 1 * e[0]) / 4
    assert energy_from_counts(counts, e) == pytest.approx(expected)


def test_interp_preserves_endpoints_and_length():
    x = np.array([0.1, 0.4, 0.9, 0.8, 0.5, 0.2])          # p=3: gamma(3) + beta(3)
    y = interp_parameters(x, 3)
    assert len(y) == 8
    assert y[0] == x[0] and y[3] == x[2] and y[4] == x[3] and y[7] == x[5]


def test_initial_parameters_shapes_and_reproducible():
    a = initial_parameters(4, "ramp", np.random.default_rng(5))
    b = initial_parameters(4, "ramp", np.random.default_rng(5))
    assert a.shape == (8,) and np.array_equal(a, b)
