"""Quantum representation: QUBO -> Ising cost Hamiltonian -> QAOA circuits.

PIPELINE
--------
1. Energy to MINIMISE (QAOA minimises):   H(x) = -F(x) + lambda * (sum_i x_i - K)^2
2. Expand with x_i^2 = x_i  ->  QUBO:
       H(x) = offset + sum_i c_i x_i + sum_{i<j} q_ij x_i x_j
       c_i  = -s_i + lambda*(1 - 2K)          (s_i: linear score of location i)
       q_ij = 2*lambda + W_ij                 (W_ij = w_O * overlap_ij, 0 for models A, B)
       offset = lambda*K^2
3. Ising map  x_i = (1 - Z_i)/2  ->  H_C = const + sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j
       h_i  = -c_i/2 - (1/4) * sum_{j != i} q_ij
       J_ij = q_ij / 4
       const = offset + sum_i c_i/2 + sum_{i<j} q_ij/4
4. QUBIT <-> VARIABLE: qubit i is x_i (location L(i+1)). Z|0> = +|0>, so
   bit 0 <-> x=0 (no team), bit 1 <-> x=1 (team deployed). Basis-state integer k
   has x_i = (k >> i) & 1.

TWO QAOA DESIGNS (selected with ``mixer=``):
  "x"  : uniform superposition |+>^n, X-mixer, constraint enforced ONLY by the penalty.
         Samples may be infeasible (wrong number of teams).
  "xy" : initial state = Dicke state |D^n_K> (equal superposition of all weight-K
         bitstrings), XY ring mixer. The XY mixer commutes with the total
         Hamming weight, so EVERY sample is feasible by construction. The penalty
         term is then a constant on the feasible subspace (it does not change
         which selection is best), but it is kept so both designs share one H_C.
These two designs are NOT equivalent; experiments compare them.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import StatePreparation, XXPlusYYGate
from qiskit.quantum_info import SparsePauliOp

from .objective import Formulation, location_scores, objective_value, pair_penalties
from .problem import MedicalDeploymentProblem

MIXERS = ("x", "xy")


# ----------------------------------------------------------------------------
# QUBO
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Qubo:
    """H(x) = offset + c . x + sum_{i<j} q_ij x_i x_j   (minimise)."""

    c: np.ndarray        # linear coefficients, shape (n,)
    q: np.ndarray        # symmetric pair coefficients, zero diagonal, shape (n, n)
    offset: float
    penalty: float
    num_teams: int

    @property
    def n(self) -> int:
        return self.c.shape[0]

    def energy(self, x) -> float:
        xv = np.asarray(x, dtype=float)
        return float(self.offset + self.c @ xv + 0.5 * xv @ self.q @ xv)


def build_qubo(problem: MedicalDeploymentProblem, form: Formulation, penalty: float) -> Qubo:
    """Build the penalised QUBO described in the module docstring."""
    if penalty < 0 or not np.isfinite(penalty):
        raise ValueError("penalty must be finite and >= 0")
    n, k = problem.n, problem.num_teams
    s = location_scores(problem, form)
    w = pair_penalties(problem, form)
    c = -s + penalty * (1 - 2 * k)
    q = 2 * penalty * (np.ones((n, n)) - np.eye(n)) + w
    return Qubo(c=c, q=q, offset=penalty * k * k, penalty=penalty, num_teams=k)


# ----------------------------------------------------------------------------
# Penalty selection
# ----------------------------------------------------------------------------
def all_selections(n: int) -> np.ndarray:
    """All 2^n binary vectors; row k satisfies x_i = (k >> i) & 1 (same as Qiskit's index)."""
    ks = np.arange(2 ** n)
    return ((ks[:, None] >> np.arange(n)[None, :]) & 1).astype(int)


def objective_over_all(problem: MedicalDeploymentProblem, form: Formulation) -> np.ndarray:
    """F(x) for all 2^n bitstrings, indexed by the integer k of the basis state."""
    return np.array([objective_value(problem, form, x) for x in all_selections(problem.n)])


def exact_min_penalty(problem: MedicalDeploymentProblem, form: Formulation) -> float:
    """Smallest lambda* such that every lambda > lambda* makes the QUBO ground state feasible.

    An infeasible x with m selected has energy -F(x) + lambda*(m-K)^2. It cannot beat the
    best feasible energy -F* iff  lambda*(m-K)^2 > F(x) - F*. Hence
        lambda* = max over infeasible x of  (F(x) - F*) / (m - K)^2     (floored at 0).
    Computed by brute force over all 2^n states; this is a VERIFICATION tool for a small
    instance, not part of the quantum algorithm.
    """
    x_all = all_selections(problem.n)
    f = objective_over_all(problem, form)
    m = x_all.sum(axis=1)
    feasible = m == problem.num_teams
    f_star = f[feasible].max()
    gaps = (f[~feasible] - f_star) / (m[~feasible] - problem.num_teams) ** 2
    return float(max(0.0, gaps.max()))


def analytic_penalty_bound(problem: MedicalDeploymentProblem, form: Formulation) -> float:
    """Closed-form SUFFICIENT penalty bound that needs no enumeration of 2^n states.

    For every x, F(x) <= U := sum_i max(s_i, 0)   (the overlap term can only lower F).
    The optimum satisfies F* >= L := (sum of the K smallest scores) - sum_{i<j} W_ij,
    because L lower-bounds F of one particular feasible selection.
    Any infeasible x has (m-K)^2 >= 1, so lambda > U - L guarantees
    lambda*(m-K)^2 > F(x) - F*, i.e. no infeasible state can beat the optimum.
    Always >= :func:`exact_min_penalty`, usually much more conservative.
    """
    s = location_scores(problem, form)
    w = pair_penalties(problem, form)
    upper_any = float(np.maximum(s, 0).sum())
    k = problem.num_teams
    lower_feasible = float(np.sort(s)[:k].sum() - np.triu(w, 1).sum())
    return max(0.0, upper_any - lower_feasible)


def default_penalty(problem: MedicalDeploymentProblem, form: Formulation, margin: float = 1.25) -> float:
    """Default lambda = margin * exact threshold (+ small epsilon), so feasibility is guaranteed."""
    return margin * exact_min_penalty(problem, form) + 1e-3


# ----------------------------------------------------------------------------
# Ising / Hamiltonian
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Ising:
    """H_C = const + sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j."""

    h: np.ndarray
    j: np.ndarray        # symmetric, zero diagonal
    const: float

    @property
    def n(self) -> int:
        return self.h.shape[0]


def qubo_to_ising(qubo: Qubo) -> Ising:
    """x_i = (1 - Z_i)/2 substitution (derivation in the module docstring)."""
    h = -qubo.c / 2 - qubo.q.sum(axis=1) / 4
    j = qubo.q / 4
    const = qubo.offset + qubo.c.sum() / 2 + np.triu(qubo.q, 1).sum() / 4
    return Ising(h=h, j=j, const=float(const))


def ising_to_operator(ising: Ising, include_constant: bool = True) -> SparsePauliOp:
    """Cost Hamiltonian as a Qiskit SparsePauliOp (qubit i = variable x_i)."""
    n = ising.n
    terms = [("Z", [i], float(ising.h[i])) for i in range(n)]
    terms += [("ZZ", [i, k], float(ising.j[i, k])) for i in range(n) for k in range(i + 1, n)]
    if include_constant:
        terms.append(("", [], ising.const))
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n).simplify()


def hamiltonian_diagonal(op: SparsePauliOp) -> np.ndarray:
    """Energy of every computational basis state k (x_i = (k>>i)&1) from the operator itself."""
    return np.real(op.to_matrix(sparse=True).diagonal())


def energies_classical(problem: MedicalDeploymentProblem, form: Formulation, penalty: float) -> np.ndarray:
    """Reference energies -F(x) + lambda (sum x - K)^2 for all 2^n states (no QUBO code involved)."""
    x_all = all_selections(problem.n)
    f = objective_over_all(problem, form)
    return -f + penalty * (x_all.sum(axis=1) - problem.num_teams) ** 2


def verify_model_consistency(problem: MedicalDeploymentProblem, form: Formulation, penalty: float) -> dict:
    """Max |difference| between (a) classical energies, (b) QUBO energies, (c) Hamiltonian diagonal."""
    qubo = build_qubo(problem, form, penalty)
    op = ising_to_operator(qubo_to_ising(qubo))
    classical = energies_classical(problem, form, penalty)
    x_all = all_selections(problem.n)
    qubo_e = np.array([qubo.energy(x) for x in x_all])
    ham_e = hamiltonian_diagonal(op)
    return {
        "max_abs_diff_qubo_vs_classical": float(np.max(np.abs(qubo_e - classical))),
        "max_abs_diff_hamiltonian_vs_classical": float(np.max(np.abs(ham_e - classical))),
        "num_states_checked": int(len(x_all)),
        "ground_state_index": int(np.argmin(ham_e)),
        "ground_state_is_feasible": bool(x_all[int(np.argmin(ham_e))].sum() == problem.num_teams),
    }


# ----------------------------------------------------------------------------
# Circuits
# ----------------------------------------------------------------------------
def dicke_amplitudes(n: int, k: int) -> np.ndarray:
    """State vector of the Dicke state |D^n_k>: uniform over all weight-k basis states."""
    amps = np.zeros(2 ** n, dtype=complex)
    for ones in itertools.combinations(range(n), k):
        amps[sum(1 << i for i in ones)] = 1.0
    return amps / np.sqrt(np.count_nonzero(amps))


def cost_scale(ising: Ising) -> float:
    """Largest |coefficient| in H_C; angles are applied to H_C/scale to keep gamma ~ O(1)."""
    biggest = max(float(np.max(np.abs(ising.h))), float(np.max(np.abs(ising.j))))
    return biggest if biggest > 0 else 1.0


def _ring_pairs(n: int) -> Tuple[list, list]:
    """Ring pairs split into two layers of disjoint pairs (even bonds, odd bonds)."""
    even = [(i, i + 1) for i in range(0, n - 1, 2)]
    odd = [(i, (i + 1) % n) for i in range(1, n, 2)]
    if n % 2 == 1:  # odd ring: the closing bond (n-1, 0) clashes; give it its own layer
        odd = [(i, i + 1) for i in range(1, n - 1, 2)]
        return even, odd + [(n - 1, 0)]
    return even, odd


def build_qaoa_circuit(ising: Ising, num_teams: int, depth: int, mixer: str = "x") -> QuantumCircuit:
    """Parameterised QAOA ansatz (no measurement), parameters gamma[0..p-1], beta[0..p-1].

    Layer l:  cost  U_C(gamma_l) = exp(-i gamma_l H_C/scale)  via RZ / RZZ gates
              mixer U_M(beta_l)  = RX(2 beta_l) on each qubit          (mixer="x")
                                 = XX+YY rotations on a ring of qubits  (mixer="xy")
    """
    if depth < 1:
        raise ValueError("depth p must be >= 1")
    if mixer not in MIXERS:
        raise ValueError(f"mixer must be one of {MIXERS}")
    n = ising.n
    scale = cost_scale(ising)
    gamma = ParameterVector("gamma", depth)
    beta = ParameterVector("beta", depth)
    qc = QuantumCircuit(n, name=f"QAOA_p{depth}_{mixer}")

    if mixer == "x":
        qc.h(range(n))                                          # |+>^n
    else:
        qc.append(StatePreparation(dicke_amplitudes(n, num_teams)), range(n))  # Dicke state
    qc.barrier()

    for layer in range(depth):
        for i in range(n):                                       # cost layer: single-qubit Z terms
            if ising.h[i] != 0:
                qc.rz(2 * gamma[layer] * float(ising.h[i]) / scale, i)
        for i in range(n):                                       # cost layer: ZZ couplings
            for k in range(i + 1, n):
                if ising.j[i, k] != 0:
                    qc.rzz(2 * gamma[layer] * float(ising.j[i, k]) / scale, i, k)
        if mixer == "x":
            for i in range(n):
                qc.rx(2 * beta[layer], i)
        else:
            for layer_pairs in _ring_pairs(n):
                for a, b in layer_pairs:
                    qc.append(XXPlusYYGate(2 * beta[layer], 0.0), [a, b])
        qc.barrier()
    return qc


def with_measurements(qc: QuantumCircuit) -> QuantumCircuit:
    """Copy of ``qc`` with all qubits measured into a classical register of equal size."""
    out = qc.copy()
    out.measure_all()
    return out


def ordered_parameters(qc: QuantumCircuit) -> list:
    """Circuit parameters in the fixed order [gamma_0..gamma_{p-1}, beta_0..beta_{p-1}].

    Works for the original and for transpiled circuits (parameter names are preserved).
    """
    def index(param) -> int:
        return int(param.name.split("[")[1].rstrip("]"))

    gammas = sorted((q for q in qc.parameters if q.name.startswith("gamma")), key=index)
    betas = sorted((q for q in qc.parameters if q.name.startswith("beta")), key=index)
    return gammas + betas


def bind_parameters(qc: QuantumCircuit, values) -> QuantumCircuit:
    """Bind ``values`` given in the order of :func:`ordered_parameters`."""
    params = ordered_parameters(qc)
    if len(values) != len(params):
        raise ValueError(f"expected {len(params)} parameter values, got {len(values)}")
    return qc.assign_parameters(dict(zip(params, [float(v) for v in values])))
