# Quantum-Assisted Optimization of Mobile Medical Team Deployment During Disasters

Qiskit Fall Fest 2026 hackathon submission, domain **Disaster Response**. QAOA on the Qiskit Aer simulator, verified against an exact classical baseline.

> **Honest scope.** 10 qubits are classically trivial (120 candidate deployments). This project demonstrates a correct, verified quantum
> formulation and a working hybrid loop. It does **not** claim quantum speedup or advantage. All data are **synthetic**.

## 1. Problem
A disaster has hit 10 locations. Three interchangeable mobile medical teams start at one depot. Choose the 3 locations that receive a team,
balancing **patients needing care**, **outbreak risk** (e.g. cholera, dengue) and **travel time**. Users: health departments and field coordinators.

**Assumptions (modelling choices, not facts):** teams are interchangeable; one depot; at most one team per location; no routing, scheduling,
traffic or team-specific skills. This is a *selection* problem, not vehicle routing and not distinguishable-team assignment.

## 2. Formulation
Variables `x_i ∈ {0,1}` (team sent to location `L(i+1)`), qubit `i` = `x_i`, measurement 0 = no team, 1 = team. Constraint `Σ x_i = 3` (C(10,3) = 120 feasible of 2^10 = 1024 bitstrings).

```
maximise F(x) = Σ_i s_i x_i  -  w_O Σ_{i<j} O_ij x_i x_j
s_i = w_P·P_i/max(P) + w_R·R_i/100 − w_T·T_i/max(T)           (patients, risk benefit; travel cost)
```
Three formulations (same data, same constraint):

| Model | Meaning | Weights |
|---|---|---|
| **A** | patients + travel only (no outbreak term) | w_P=0.5, w_T=0.2 (same as B, not renormalised) |
| **B** | outbreak-aware (as in the strategy document) | w_P=0.5, w_R=0.3, w_T=0.2 |
| **C** | B + pairwise **service-area overlap** penalty (our extension) | as B, w_O=0.3 |

`O_ij = max(0, 1 − d_ij/15 km)` from synthetic coordinates (travel time = distance / 0.5 km·min⁻¹). Why C: with w_O = 0 the objective is linear and the exact optimum is simply the 3 largest `s_i` (a sort), so a quadratic term is what makes this a genuine QUBO. Weights are proposed experimental choices, not medical truth; Section 7 shows how the answer moves with them.

### Dataset (`src/problem.py`, SYNTHETIC)
Based on the strategy document's example, with one change: **L7 = 100 patients, risk 30** (document: 95, 55). With the original numbers models A and B select the same deployment ({L1,L3,L5}), so the outbreak comparison would show nothing.

## 3. QUBO → Ising (`src/quantum_model.py`)
Energy to minimise: `H(x) = −F(x) + λ(Σx_i − 3)²`. With `x_i² = x_i`: `H = λ·9 + Σ c_i x_i + Σ_{i<j} q_ij x_i x_j`, `c_i = −s_i − 5λ`, `q_ij = 2λ + w_O O_ij`.
Substituting `x_i = (1 − Z_i)/2`: `h_i = −c_i/2 − ¼Σ_j q_ij`, `J_ij = q_ij/4`.
**λ is not arbitrary:** `exact_min_penalty` computes λ* = max over infeasible states of (F(x)−F*)/(m−K)², the smallest value above which every ground state is feasible; we use 1.25·λ* (λ* = 0.436 for model B, 0.356 for C). Tests prove QUBO energies and Hamiltonian diagonal equal the classical energies for **all 1024 states** (max error ~1e-14), and that below λ* an infeasible state becomes the ground state.
Qiskit is little-endian: bitstring `"0000010101"` = L1, L3, L5; decoding is unit-tested.

## 4. Quantum algorithm
QAOA, depth p, cost layer `RZ/RZZ` from `H_C`, circuits built in Qiskit, run on `AerSimulator`, classical COBYLA updates (γ, β) from **sampled** energies. Two designs, deliberately not equivalent:

| | `mixer="x"` | `mixer="xy"` (default) |
|---|---|---|
| Start | uniform superposition | Dicke state \|D(10,3)⟩ (amplitude loading) |
| Mixer | RX on every qubit | XX+YY ring (Hamming-weight preserving) |
| Constraint | penalty only; infeasible samples occur | every sample feasible by construction |
| Cost | 45 two-qubit gates per layer | ≈1013 CX for the Dicke start + 65 per layer (Dicke preparation dominates; depth ≈ 2000) |

Initial parameters: INTERP warm start across depths (Zhou et al. 2020), implemented from the paper's description. Seeds fix start points and simulator sampling; final samples use an independent seed. The exact solver is never used inside the optimisation loop.

## 5. Repository
```
src/problem.py          data + validation          src/objective.py       F(x), models A/B/C
src/constraints.py      cardinality, decoding      src/classical_solver.py exact baseline (120 candidates, independent)
src/quantum_model.py    QUBO, Ising, circuits      src/qaoa_solver.py     hybrid QAOA loop
src/evaluation.py       metrics vs exact + controls  tests/                110 tests
experiments/run_experiments.py   main experiments    experiments/run_extra.py  E8-E10 (heuristics, scaling, robustness)
experiments/make_architecture.py, build_notebook.py, build_report.py (-> docs/final_report.docx)
src/heuristics.py greedy/SA/local search    src/instances.py scaling instances
experiments/results/    CSV/JSON/PNG (generated)     docs/architecture.png, notebooks/experimentation.ipynb, video/demo_link.txt
```

## 6. Run it
From the repository root (tested with Python 3.12, Qiskit 2.6.2, Aer 0.18.2):
```
python -m venv .venv && .venv\Scripts\activate        # Windows (Linux/macOS: source .venv/bin/activate)
python -m pip install -r requirements.txt
python -m pytest -q                                    # 110 tests, ~1-3 min
python experiments/run_experiments.py --quick          # ~1 min smoke test
python experiments/run_experiments.py                  # full study, ~7-11 min (10.7 min on a 4-worker Windows laptop)
python -m experiments.run_extra                        # E8-E10 extra studies, ~15 min (n=14 alone ~7 min)
python experiments/build_report.py                     # rebuild docs/final_report.docx
python experiments/make_architecture.py                # redraw docs/architecture.png
python experiments/build_notebook.py && python -m jupyter nbconvert --to notebook --execute --inplace notebooks/experimentation.ipynb
```

## 7. Results (all numbers from `experiments/results/`, full run, 110 QAOA chains)
**Exact baseline (120 deployments each; no ties):**

| Model | Optimal deployment | F* | Patients | Risk sum | Travel (min) |
|---|---|---|---|---|---|
| A | L1, L3, L7 | 0.8848 | 290 | 205 | 130 |
| B | L1, L3, L5 | 1.6152 | 260 | 250 | 95 |
| C | L1, L3, L9 | 1.5207 | 255 | 255 | 100 |

Ignoring outbreak risk (A's deployment scored under B) costs 0.115, i.e. 7.1% of F*; ignoring overlap (B's deployment scored under C) costs 0.113, 7.4% (`e1_cross_evaluation.csv`). Over a 0.1-step weight grid the optimum takes 4 / 9 / 8 distinct values for A / B / C (`fig3_weight_sensitivity.png`), so conclusions depend on the weights.

**"Why not just sort?" (E7, `e7_why_not_sort.csv`, `fig9_why_not_sort.png`).** Without the overlap term the optimum is exactly "take the 3 highest scores". On our 6 × 11 grid (outbreak weight 0.0 to 0.5, overlap weight 0.0 to 1.0, travel weight fixed at 0.2) sorting is wrong in 40 of 66 cells: for every overlap weight above 0 once the outbreak weight is 0.2 or more, and never when it is 0.0 or 0.1 (there the top-3 locations are far apart, so overlap does not bite). So the overlap term matters precisely when outbreak risk pulls the clustered locations L1, L3, L5 to the top; this is a property of our synthetic geometry, not a general result. At weights 0.5/0.3/0.2: sorting gives L1, L3, L5 while the exact optimum is L1, L3, L9 at w_O = 0.1 to 0.3 and L1, L3, L6 from 0.4 up; the regret of sorting is 0.021 at w_O = 0.1, 0.113 at 0.3 and 0.588 at 1.0 (F* ≈ 1.5). A unit test (`test_overlap_term_makes_sorting_wrong...`) enforces this. Note that model C's best and second-best deployments differ by only 0.009 in F, so the optimum is nearly degenerate; the QAOA "optimum probability" numbers are for the exact best only.

**QAOA depth study, model C, 2048 shots, 10 seeds (ideal = exact statevector of the final circuit):**

| Mixer | p | P(exact optimum) mean ± std | Feasible samples | Mean runtime (s) |
|---|---|---|---|---|
| X + penalty | 1 / 3 / 5 | 0.0033 / 0.0092 / 0.0121 | 34% / 64% / 40% | 1.9 / 7.0 / 16.3 |
| XY + Dicke | 1 / 3 / 5 | 0.022 / 0.055 / 0.067 (±0.019 / 0.027 / 0.033) | 100% | 6.7 / 24.6 / 49.4 |
| Random feasible (control) | – | 0.0082 (theory 1/120) | 100% | – |
| Random 10-bit strings (control) | – | 0.0010 | 12% | – |

XY at p=5 puts ≈8× more probability on the exact optimum than random feasible sampling; the X-mixer reaches ≈1.5× and its feasibility is non-monotone in p. Runtime = transpile + optimisation + final sampling (+ shallower INTERP stages), measured with 4 parallel worker processes on a laptop CPU (Aer statevector simulator); absolute times depend on the machine.
**Caution:** the best *feasible sampled* deployment equals the exact optimum in all 100 depth-study runs, but that is **not evidence of QAOA skill**: with 2048 shots, random feasible sampling finds the optimum as well (miss probability ≈ 3e-8). Enrichment over random is the meaningful metric.
Other studies: shots 128–8192 (`summary_shots.csv`), penalty 0.25–4 × λ* (`summary_penalty.csv`: feasibility 8% at 0.25 λ* (4 of 5 runs found the optimum), 66% at λ*, 67–73% above; below λ* the QUBO ground state is infeasible), and QAOA recovers each model's *different* optimum in 5/5 seeds (`summary_models_qaoa.csv`, XY, p=4).

### 7b. Extra studies: heuristics, scaling, robustness, hardware size (`experiments/run_extra.py`, `docs/final_report.docx`)
**Classical heuristics (E8, model C, 300 repetitions; `e8_heuristics.csv`).** Probability of returning the exact optimum: local search 0.13 / 0.48 / 0.93 / 1.00 and simulated annealing 0.19 / 0.48 / 0.85 / 0.97 at 10 / 25 / 50 / 100 objective evaluations; random search 0.08 / 0.20 / 0.36 / 0.54; greedy construction finds the optimum with 27 evaluations. **Classical heuristics solve this instance easily; QAOA is not competitive here.**

**Scaling (E9, synthetic seeded instances, model C, XY-QAOA p=3, 2-3 seeds; `e9_scaling_*.csv`, `fig10_scaling.png`).**

| n | K | candidates | QAOA P(opt) | random P(opt) | enrichment | logical depth |
|---|---|---|---|---|---|---|
| 8 | 3 | 56 | 0.0671 | 0.0179 | 3.8× | 550 |
| 10 | 3 | 120 | 0.0389 | 0.0083 | 4.7× | 2089 |
| 12 | 4 | 495 | 0.0285 | 0.0020 | 14.1× | 8246 |
| 14 | 4 | 1001 | 0.0056 | 0.0010 | 5.6× | 32830 |

Enrichment over random sampling persists up to n = 14 on average, but the circuit depth (generic Dicke state) explodes (n=14: ~33k layers, ~400 s per run on 1 CPU) and the n = 14 result is noisy: one of two seeds (P=0.clear0005) was below random sampling (0.0010). Only 2-3 seeds per size, so this is indicative only.

**Robustness of the model-C optimum (E10a; 3000 data perturbations per level, `e10_robustness.csv`).**

| input noise σ | nominal optimum (L1,L3,L9) still exactly optimal | optimum contains L1 and L3 | nominal within top 5 of 120 |
|---|---|---|---|
| 0.05 | 0.53 | 1.00 | 1.00 |
| 0.10 | 0.46 | 0.99 | 0.99 |
| 0.20 | 0.32 | 0.85 | 0.82 |
| 0.30 | 0.21 | 0.67 | 0.62 |

The exact optimum is fragile (the best and second-best differ by 0.009), but "send teams to L1 and L3" is robust at low noise; the third location is the uncertain choice. XY-QAOA (p=5, 3 seeds) puts 8.5% of its samples on the exact optimum, 15.6% on the 2 deployments within 1% of F*, and 35.9% on the 8 deployments within 10% (random feasible sampling: 0.8% / 1.7% / 6.7%).

**Hardware size (E10c, `e10_transpiled_size.csv`; noise-free transpilation to a heavy-hex coupling map, ECR/RZ/SX/X basis; no hardware run).** Transpiled depth at p=1: X mixer 338 (195 ECR), XY + generic Dicke start 7471 (2821 ECR), XY + shallow basis start 334 (242 ECR). The XY circuit is far too deep for current devices. **Negative result:** replacing the Dicke start by the single basis state |1110000000⟩ (`QaoaConfig(start="basis")`) makes the circuit shallow but P(optimum) was 0.0 in 3 of 3 runs, so we keep the Dicke start.

## 8. Limitations
* No quantum advantage; 120 candidates are enumerated instantly. Noiseless simulation only; no hardware run (transpiled circuit sizes are reported in 7b); scaling was studied only up to n = 14 with 2-3 seeds.
* Synthetic data; weights, the 0.5 km/min speed and the overlap model (w_O = 0.3, 15 km) are assumptions we chose. Model C's overlap term is our own modelling extension.
* One depot, interchangeable teams, no routing/capacity/time dynamics; this is decision support, not an epidemiological or outbreak-prevention model.
* X-mixer QAOA wastes most samples on infeasible or poor states. XY fixes feasibility but its Dicke start (amplitude loading, ~1000 CX) dominates circuit size; a dedicated linear-depth Dicke circuit would be needed at scale.
* Per-depth success probabilities stay below 10% (QAOA at p ≤ 5 with a local optimiser); COBYLA can stall in local minima and results use 5-10 seeds only.

## 9. What is original vs existing
Existing methods: QAOA (Farhi et al. 2014), quantum alternating operator ansatz / XY mixers (Hadfield et al. 2019), INTERP initialisation (Zhou et al. 2020), Dicke states (Bärtschi & Eidenbenz 2019; we use plain amplitude loading, not their circuit), COBYLA (Powell). Ours: the medical-team deployment instance, the outbreak-aware vs patient-only comparison (A vs B), the overlap-penalised extension (C), the penalty-threshold analysis, and the experiments above. We make no novelty claim beyond this combination.

## 10. References
Farhi, Goldstone, Gutmann, *A Quantum Approximate Optimization Algorithm*, arXiv:1411.4028 (2014). Hadfield et al., *From the Quantum Approximate Optimization Algorithm to a Quantum Alternating Operator Ansatz*, Algorithms 12(2):34 (2019). Zhou et al., *QAOA: Performance, Mechanism, and Implementation on Near-Term Devices*, Phys. Rev. X 10, 021067 (2020). Bärtschi & Eidenbenz, *Deterministic Preparation of Dicke States*, FCT 2019. Qiskit and Qiskit Aer documentation (IBM Quantum). Please verify citation details before submitting.

Acknowledgements: IBM Qiskit, FAST Qiskit Fall Fest 2026.
