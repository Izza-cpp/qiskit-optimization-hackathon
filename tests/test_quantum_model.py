import itertools

import numpy as np
import pytest
import scipy.sparse.linalg as spla
from qiskit.quantum_info import SparsePauliOp, Statevector

from src.classical_solver import solve_exact
from src.objective import make_formulation
from src.problem import default_problem
from src.quantum_model import (all_selections, analytic_penalty_bound, bind_parameters, build_qaoa_circuit,
                               build_qubo, cost_scale, default_penalty, dicke_amplitudes, energies_classical,
                               exact_min_penalty, hamiltonian_diagonal, ising_to_operator, objective_over_all,
                               ordered_parameters, qubo_to_ising, verify_model_consistency, with_measurements)
from src.objective import location_scores, pair_penalties

P = default_problem()
N, K = 10, 3


@pytest.fixture(scope="module", params=["A", "B", "C"])
def setup(request):
    f = make_formulation(request.param)
    lam = default_penalty(P, f)
    qubo = build_qubo(P, f, lam)
    ising = qubo_to_ising(qubo)
    return f, lam, qubo, ising


def test_qubo_coefficients_match_hand_expansion(setup):
    f, lam, qubo, _ = setup
    s, w = location_scores(P, f), pair_penalties(P, f)
    assert np.allclose(qubo.c, -s - 5 * lam)                 # c_i = -s_i + lam(1-2K), K=3
    assert qubo.offset == pytest.approx(9 * lam)             # lam K^2
    off = ~np.eye(N, dtype=bool)
    assert np.allclose(qubo.q[off], (2 * lam + w)[off])      # q_ij = 2 lam + W_ij
    assert np.allclose(np.diag(qubo.q), 0) and np.allclose(qubo.q, qubo.q.T)


def test_qubo_energy_equals_classical_energy_for_all_1024(setup):
    f, lam, qubo, _ = setup
    ref = energies_classical(P, f, lam)
    got = np.array([qubo.energy(x) for x in all_selections(N)])
    assert np.max(np.abs(got - ref)) < 1e-12


def test_hamiltonian_diagonal_equals_classical_energy_for_all_1024(setup):
    f, lam, _, ising = setup
    diag = hamiltonian_diagonal(ising_to_operator(ising))
    assert np.max(np.abs(diag - energies_classical(P, f, lam))) < 1e-12


def test_hamiltonian_is_diagonal_only_I_and_Z(setup):
    op = ising_to_operator(setup[3])
    for label in op.paulis.to_labels():
        assert set(label) <= {"I", "Z"}


def test_qubit_to_variable_convention():
    """Basis state integer k has x_i = (k>>i)&1 and Z_i|1> = -|1> (so x_i = (1-Z_i)/2)."""
    f = make_formulation("B")
    lam = 0.5
    diag = hamiltonian_diagonal(ising_to_operator(qubo_to_ising(build_qubo(P, f, lam))))
    x = np.zeros(N, dtype=int); x[[0, 2, 4]] = 1            # L1, L3, L5
    k = sum(1 << i for i in range(N) if x[i])
    assert k == 0b0000010101 == 21
    assert diag[k] == pytest.approx(energies_classical(P, f, lam)[k])
    assert all_selections(N)[k].tolist() == x.tolist()


def test_ground_state_is_the_classical_optimum_when_penalty_is_large_enough(setup):
    f, lam, _, ising = setup
    diag = hamiltonian_diagonal(ising_to_operator(ising))
    ground = all_selections(N)[int(np.argmin(diag))]
    assert ground.sum() == K
    assert tuple(np.flatnonzero(ground)) == solve_exact(P, f).chosen_set


def test_penalty_threshold_is_tight_on_both_sides():
    """Above lambda*: ground state feasible. Clearly below it: an infeasible state wins."""
    for model in "ABC":
        f = make_formulation(model)
        star = exact_min_penalty(P, f)
        assert star > 0
        for lam, should_be_feasible in ((1.01 * star + 1e-9, True), (0.5 * star, False)):
            diag = energies_classical(P, f, lam)
            assert bool(all_selections(N)[int(np.argmin(diag))].sum() == K) is should_be_feasible
        assert analytic_penalty_bound(P, f) >= star - 1e-12


def test_consistency_report():
    rep = verify_model_consistency(P, make_formulation("C"), 0.7)
    assert rep["num_states_checked"] == 1024
    assert rep["max_abs_diff_hamiltonian_vs_classical"] < 1e-12
    assert rep["max_abs_diff_qubo_vs_classical"] < 1e-12


def test_objective_over_all_indexing():
    f = make_formulation("B")
    f_all = objective_over_all(P, f)
    assert f_all.shape == (1024,)
    assert f_all[21] == pytest.approx(solve_exact(P, f).best_value)


# --------------------------------- circuits --------------------------------
def test_dicke_state_is_uniform_over_weight_k():
    amps = dicke_amplitudes(N, K)
    probs = np.abs(amps) ** 2
    weights = all_selections(N).sum(axis=1)
    assert probs.sum() == pytest.approx(1.0)
    assert np.all(probs[weights != K] == 0)
    assert np.allclose(probs[weights == K], 1 / 120)


def test_circuit_shape_and_parameters(setup):
    ising = setup[3]
    for mixer in ("x", "xy"):
        qc = build_qaoa_circuit(ising, K, depth=3, mixer=mixer)
        assert qc.num_qubits == N
        names = [q.name for q in ordered_parameters(qc)]
        assert names == ["gamma[0]", "gamma[1]", "gamma[2]", "beta[0]", "beta[1]", "beta[2]"]
        assert with_measurements(qc).num_clbits == N


def test_invalid_circuit_arguments(setup):
    with pytest.raises(ValueError):
        build_qaoa_circuit(setup[3], K, depth=0)
    with pytest.raises(ValueError):
        build_qaoa_circuit(setup[3], K, depth=1, mixer="z")
    with pytest.raises(ValueError):
        bind_parameters(build_qaoa_circuit(setup[3], K, 2), [0.1, 0.2])


def test_x_mixer_circuit_equals_exact_matrix_evolution():
    """The gate sequence must equal exp(-i beta sum X) exp(-i gamma H_C/scale) |+>^n (fidelity 1)."""
    f = make_formulation("C")
    ising = qubo_to_ising(build_qubo(P, f, default_penalty(P, f)))
    gamma, beta = 0.37, 0.61
    state = Statevector.from_instruction(bind_parameters(build_qaoa_circuit(ising, K, 1, "x"), [gamma, beta])).data
    hc = ising_to_operator(ising, include_constant=False).to_matrix(sparse=True)
    mix = SparsePauliOp.from_sparse_list([("X", [i], 1.0) for i in range(N)], num_qubits=N).to_matrix(sparse=True)
    psi = np.ones(2 ** N, dtype=complex) / np.sqrt(2 ** N)
    psi = spla.expm_multiply(-1j * gamma / cost_scale(ising) * hc, psi)
    psi = spla.expm_multiply(-1j * beta * mix, psi)
    assert abs(np.vdot(psi, state)) ** 2 == pytest.approx(1.0, abs=1e-9)


def test_wrong_gamma_sign_would_be_detected():
    """Guards the test above: a different gamma must NOT give fidelity 1 (the test can fail)."""
    f = make_formulation("C")
    ising = qubo_to_ising(build_qubo(P, f, default_penalty(P, f)))
    state = Statevector.from_instruction(bind_parameters(build_qaoa_circuit(ising, K, 1, "x"), [0.37, 0.61])).data
    hc = ising_to_operator(ising, include_constant=False).to_matrix(sparse=True)
    mix = SparsePauliOp.from_sparse_list([("X", [i], 1.0) for i in range(N)], num_qubits=N).to_matrix(sparse=True)
    psi = np.ones(2 ** N, dtype=complex) / np.sqrt(2 ** N)
    psi = spla.expm_multiply(+1j * 0.37 / cost_scale(ising) * hc, psi)
    psi = spla.expm_multiply(-1j * 0.61 * mix, psi)
    assert abs(np.vdot(psi, state)) ** 2 < 0.999


@pytest.mark.parametrize("params", [[0.4, 0.9], [1.7, 0.2]])
def test_xy_circuit_preserves_hamming_weight(setup, params):
    qc = build_qaoa_circuit(setup[3], K, depth=1, mixer="xy")
    probs = Statevector.from_instruction(bind_parameters(qc, params)).probabilities()
    weights = all_selections(N).sum(axis=1)
    assert probs[weights != K].sum() < 1e-12
    assert probs[weights == K].sum() == pytest.approx(1.0)


def test_xy_mixer_actually_moves_probability_between_feasible_states(setup):
    qc = build_qaoa_circuit(setup[3], K, depth=1, mixer="xy")
    probs = Statevector.from_instruction(bind_parameters(qc, [0.4, 0.9])).probabilities()
    weights = all_selections(N).sum(axis=1)
    assert not np.allclose(probs[weights == K], 1 / 120, atol=1e-4)


def test_x_mixer_leaves_the_feasible_subspace(setup):
    qc = build_qaoa_circuit(setup[3], K, depth=1, mixer="x")
    probs = Statevector.from_instruction(bind_parameters(qc, [0.4, 0.9])).probabilities()
    assert probs[all_selections(N).sum(axis=1) != K].sum() > 0.5
