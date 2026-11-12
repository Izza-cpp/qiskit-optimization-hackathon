"""Tests for the scaling instance generator, classical heuristics and the shallow start option."""
import itertools

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from src import heuristics as H
from src.classical_solver import solve_exact
from src.instances import random_problem
from src.objective import make_formulation
from src.problem import default_problem
from src.quantum_model import bind_parameters, build_qaoa_circuit, build_qubo, default_penalty, qubo_to_ising


def _fn(problem, form):
    from experiments.run_extra import _objective_fn
    return _objective_fn(problem, form)


def test_random_problem_is_valid_and_reproducible():
    a, b = random_problem(12, 4, seed=5), random_problem(12, 4, seed=5)
    assert a.n == 12 and a.num_teams == 4
    assert np.array_equal(a.patients, b.patients) and np.array_equal(a.coordinates_km, b.coordinates_km)
    assert not np.array_equal(a.patients, random_problem(12, 4, seed=6).patients)
    with pytest.raises(ValueError):
        random_problem(5, 5, seed=0)


def test_objective_fn_matches_exact_solver_values():
    p, f = default_problem(), make_formulation("C")
    exact = solve_exact(p, f)
    fn = _fn(p, f)
    for sel, val in list(exact.all_values.items())[:20]:
        assert fn(sel) == pytest.approx(val, abs=1e-12)


def test_heuristics_return_feasible_sets_within_budget():
    p, f = default_problem(), make_formulation("C")
    fn = _fn(p, f)
    rng = np.random.default_rng(0)
    for method in (H.random_search, H.local_search, H.simulated_annealing):
        sel, used = method(p.n, p.num_teams, fn, rng, 30)
        assert len(set(sel)) == p.num_teams and all(0 <= i < p.n for i in sel)
        assert used <= 30 + p.num_teams * (p.n - p.num_teams)   # local search may finish one sweep
    g, used = H.greedy(p.n, p.num_teams, fn)
    assert len(g) == p.num_teams and used == sum(p.n - j for j in range(p.num_teams))


def test_heuristics_never_beat_exact_optimum():
    p, f = default_problem(), make_formulation("C")
    exact = solve_exact(p, f)
    fn = _fn(p, f)
    rng = np.random.default_rng(1)
    for method in (H.random_search, H.local_search, H.simulated_annealing):
        sel, _ = method(p.n, p.num_teams, fn, rng, 60)
        assert fn(sel) <= exact.best_value + 1e-12


def test_basis_start_preserves_hamming_weight_and_dicke_is_default():
    p, f = default_problem(), make_formulation("C")
    ising = qubo_to_ising(build_qubo(p, f, default_penalty(p, f)))
    qc = bind_parameters(build_qaoa_circuit(ising, 3, 1, "xy", start="basis"), [0.3, 0.4])
    probs = Statevector.from_instruction(qc).probabilities()
    weights = np.array([bin(k).count("1") for k in range(2 ** p.n)])
    assert probs[weights == 3].sum() == pytest.approx(1.0, abs=1e-9)
    from qiskit import transpile
    gates = ["cx", "rz", "sx", "x"]
    deep = transpile(build_qaoa_circuit(ising, 3, 1, "xy"), basis_gates=gates, optimization_level=0).depth()
    shallow = transpile(build_qaoa_circuit(ising, 3, 1, "xy", start="basis"), basis_gates=gates, optimization_level=0).depth()
    assert deep > 5 * shallow
    with pytest.raises(ValueError):
        build_qaoa_circuit(ising, 3, 1, "xy", start="nope")
