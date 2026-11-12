"""QAOA workflow on the Qiskit Aer simulator.

WHICH STEP IS QUANTUM, WHICH IS CLASSICAL
-----------------------------------------
QUANTUM (simulated by Aer): state preparation, cost/mixer layers, projective
    measurement. ``backend.run`` returns real measurement samples (counts).
CLASSICAL: decoding bitstrings, computing the sampled expected energy from counts,
    and the COBYLA optimiser that proposes new (gamma, beta). No brute force and no
    classical solution is used inside the optimisation loop; the exact solver is
    only used afterwards, in ``evaluation``, as the reference.

PARAMETER INITIALISATION ("interp", default)
    Depth 1 is optimised from a jittered ramp start; each further depth p+1 starts from
    the INTERP interpolation of the optimum at depth p (Zhou et al., PRX 10, 021067, 2020).
    This is a standard heuristic, implemented here from its published description.

Reproducibility: the start point comes from ``numpy.random.default_rng(seed)``; every Aer call
in the optimisation loop uses ``seed_simulator=seed`` (so the objective is a deterministic
function of the parameters, which COBYLA needs); the FINAL sampling uses a different, fixed
seed (``seed + FINAL_SAMPLING_SEED_OFFSET``) so reported samples are independent of the
optimiser's samples.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

from .constraints import bitstring_to_selection
from .objective import Formulation
from .problem import MedicalDeploymentProblem
from .quantum_model import (Ising, MIXERS, bind_parameters, build_qaoa_circuit, build_qubo,
                            default_penalty, hamiltonian_diagonal, ising_to_operator,
                            qubo_to_ising, with_measurements)

FINAL_SAMPLING_SEED_OFFSET = 10_000
INITS = ("interp", "ramp", "random")


@dataclass(frozen=True)
class QaoaConfig:
    """All settings of one QAOA run (everything needed to reproduce it)."""

    depth: int = 2
    mixer: str = "xy"                # "x" (penalty only) or "xy" (Hamming-weight preserving)
    shots: int = 2048                # used in the optimisation loop AND for the final sampling
    seed: int = 0
    penalty: Optional[float] = None  # None -> default_penalty (guaranteed feasible ground state)
    optimizer: str = "COBYLA"
    max_iterations: int = 300        # per optimisation stage
    rhobeg: float = 0.3              # COBYLA initial trust-region step
    tolerance: float = 1e-4          # COBYLA final tolerance
    init: str = "interp"             # "interp", "ramp" or "random"
    start: str = "dicke"             # XY mixer start state: "dicke" (default) or shallow "basis"

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError("depth must be >= 1")
        if self.mixer not in MIXERS:
            raise ValueError(f"mixer must be one of {MIXERS}")
        if self.shots < 1:
            raise ValueError("shots must be >= 1")
        if self.init not in INITS:
            raise ValueError(f"init must be one of {INITS}")
        if self.optimizer.upper() != "COBYLA":
            raise ValueError("only the COBYLA optimiser is wired up")


@dataclass
class QaoaResult:
    """Everything produced by one QAOA run (one circuit depth)."""

    config: QaoaConfig                      # config.depth is the depth of THIS result
    penalty: float
    initial_params: np.ndarray
    optimal_params: np.ndarray
    counts: Dict[str, int]                  # FINAL sampling, Qiskit little-endian strings
    energy_history: List[float]             # sampled <H_C> at every evaluation of THIS stage
    num_evaluations: int                    # circuit executions in this stage's optimisation
    cumulative_evaluations: int             # including warm-up stages (interp)
    final_sampled_energy: float
    ideal_probabilities: np.ndarray = field(repr=False, default=None)  # exact Statevector check
    ideal_energy: float = float("nan")
    transpile_seconds: float = 0.0
    optimize_seconds: float = 0.0
    final_sampling_seconds: float = 0.0
    warmup_seconds: float = 0.0             # time spent in the shallower interp stages
    circuit_depth: int = 0
    circuit_gate_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def total_seconds(self) -> float:
        """transpile + optimisation loop + final sampling (+ warm-up stages). Excludes diagnostics."""
        return (self.transpile_seconds + self.optimize_seconds
                + self.final_sampling_seconds + self.warmup_seconds)


def initial_parameters(depth: int, init: str, rng: np.random.Generator) -> np.ndarray:
    """Layout [gamma_0..gamma_{p-1}, beta_0..beta_{p-1}] for a direct (non-interp) start."""
    if init == "random":
        gamma = rng.uniform(0.0, np.pi, depth)
        beta = rng.uniform(0.0, np.pi / 2, depth)
    else:  # ramp (also the depth-1 start of interp): gamma grows, beta shrinks, small jitter
        frac = (np.arange(depth) + 0.5) / depth
        gamma = 0.8 * frac + rng.normal(0, 0.05, depth)
        beta = 0.8 * (1 - frac) + rng.normal(0, 0.05, depth)
    return np.concatenate([gamma, beta])


def interp_parameters(params: np.ndarray, depth: int) -> np.ndarray:
    """INTERP: map optimal depth-``depth`` parameters to a start for depth+1 (Zhou et al. 2020)."""
    def stretch(a: np.ndarray) -> np.ndarray:
        out = np.zeros(depth + 1)
        for i in range(depth + 1):
            if i == 0:
                out[i] = a[0]
            elif i == depth:
                out[i] = a[-1]
            else:
                out[i] = (i / depth) * a[i - 1] + ((depth - i) / depth) * a[i]
        return out

    return np.concatenate([stretch(params[:depth]), stretch(params[depth:])])


def energy_from_counts(counts: Dict[str, int], energy_table: np.ndarray) -> float:
    """Sampled <H_C> = sum_b count(b)/shots * E[int(b, 2)]  (bit k of the integer <-> qubit k)."""
    total = sum(counts.values())
    return float(sum(c * energy_table[int(b.replace(" ", ""), 2)] for b, c in counts.items()) / total)


def _optimise_stage(ising: Ising, energy_table: np.ndarray, num_teams: int, backend: AerSimulator,
                    config: QaoaConfig, depth: int, x0: np.ndarray):
    """Transpile once, then run COBYLA at one depth. Returns (x, history, nfev, compiled, ansatz, t_transpile, t_opt)."""
    ansatz = build_qaoa_circuit(ising, num_teams, depth, config.mixer, config.start)
    t0 = time.perf_counter()
    compiled = transpile(with_measurements(ansatz), backend, optimization_level=1,
                         seed_transpiler=config.seed)
    t_transpile = time.perf_counter() - t0
    history: List[float] = []

    def cost(params: np.ndarray) -> float:
        counts = backend.run(bind_parameters(compiled, params), shots=config.shots,
                             seed_simulator=config.seed).result().get_counts()   # QUANTUM EXECUTION
        value = energy_from_counts(counts, energy_table)                          # classical post-processing
        history.append(value)
        return value

    t1 = time.perf_counter()
    res = minimize(cost, x0, method="COBYLA",
                   options={"maxiter": config.max_iterations, "rhobeg": config.rhobeg, "tol": config.tolerance})
    t_opt = time.perf_counter() - t1
    return np.asarray(res.x), history, len(history), compiled, ansatz, t_transpile, t_opt


def run_qaoa_progressive(problem: MedicalDeploymentProblem, form: Formulation,
                         config: QaoaConfig) -> List[QaoaResult]:
    """Run QAOA for depths 1..config.depth with INTERP warm starts; one result per depth.

    Only valid with ``init == 'interp'``. Efficient for depth sweeps because the shallower
    stages are the warm-up of the deeper ones.
    """
    if config.init != "interp":
        raise ValueError("run_qaoa_progressive requires init='interp'")
    return _run(problem, form, config, stages=list(range(1, config.depth + 1)))


def run_qaoa(problem: MedicalDeploymentProblem, form: Formulation, config: QaoaConfig) -> QaoaResult:
    """Run the full hybrid loop at ``config.depth`` and return structured results."""
    stages = list(range(1, config.depth + 1)) if config.init == "interp" else [config.depth]
    return _run(problem, form, config, stages)[-1]


def _run(problem: MedicalDeploymentProblem, form: Formulation, config: QaoaConfig,
         stages: List[int]) -> List[QaoaResult]:
    penalty = config.penalty if config.penalty is not None else default_penalty(problem, form)
    ising = qubo_to_ising(build_qubo(problem, form, penalty))
    energy_table = hamiltonian_diagonal(ising_to_operator(ising))   # lookup to score SAMPLED bitstrings
    backend = AerSimulator(method="statevector")
    rng = np.random.default_rng(config.seed)

    results: List[QaoaResult] = []
    x = None
    cumulative_evals = 0
    cumulative_time = 0.0
    for depth in stages:
        if x is None:
            x0 = initial_parameters(depth, "random" if config.init == "random" else "ramp", rng)
        else:
            x0 = interp_parameters(x, depth - 1)
        x, history, nfev, compiled, ansatz, t_tr, t_opt = _optimise_stage(
            ising, energy_table, problem.num_teams, backend, config, depth, x0)
        cumulative_evals += nfev

        t2 = time.perf_counter()
        final_counts = backend.run(bind_parameters(compiled, x), shots=config.shots,
                                   seed_simulator=config.seed + FINAL_SAMPLING_SEED_OFFSET
                                   ).result().get_counts()
        t_final = time.perf_counter() - t2

        # Diagnostics (excluded from timings): exact output distribution of the ideal circuit.
        probs = Statevector.from_instruction(bind_parameters(ansatz, x)).probabilities()

        stage_config = QaoaConfig(**{**config.__dict__, "depth": depth})
        results.append(QaoaResult(
            config=stage_config, penalty=float(penalty), initial_params=np.asarray(x0),
            optimal_params=x, counts=dict(final_counts), energy_history=history,
            num_evaluations=nfev, cumulative_evaluations=cumulative_evals,
            final_sampled_energy=energy_from_counts(final_counts, energy_table),
            ideal_probabilities=probs, ideal_energy=float(probs @ energy_table),
            transpile_seconds=t_tr, optimize_seconds=t_opt, final_sampling_seconds=t_final,
            warmup_seconds=cumulative_time, circuit_depth=int(compiled.depth()),
            circuit_gate_counts={k: int(v) for k, v in compiled.count_ops().items()},
        ))
        cumulative_time += t_tr + t_opt + t_final
    return results


def decode_counts(counts: Dict[str, int], n: int) -> Dict[tuple, int]:
    """Aggregate counts by decoded selection vector x (little-endian handled)."""
    out: Dict[tuple, int] = {}
    for bitstring, c in counts.items():
        x = bitstring_to_selection(bitstring, n)
        out[x] = out.get(x, 0) + c
    return out
