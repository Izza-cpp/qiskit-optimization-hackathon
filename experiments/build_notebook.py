"""Builds notebooks/experimentation.ipynb (then execute it with nbconvert; see README)."""
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
md("""# Quantum-assisted deployment of mobile medical teams (Qiskit Fall Fest 2026)

**Domain:** Disaster Response. **Question:** a disaster hit 10 locations; 3 interchangeable mobile medical teams start from one depot.
Which 3 locations should receive a team, trading off patients, outbreak risk and travel time?

Run from the repository root (`jupyter nbconvert --to notebook --execute notebooks/experimentation.ipynb`).
Everything here uses **synthetic** data and the code in `src/`. Runtime: about 1 minute."""),
code("""import sys, pathlib
ROOT = pathlib.Path.cwd()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from src.problem import default_problem
from src.objective import make_formulation, objective_breakdown
from src.classical_solver import solve_exact
from src.quantum_model import (build_qubo, qubo_to_ising, ising_to_operator, default_penalty, exact_min_penalty,
                               verify_model_consistency, build_qaoa_circuit)
from src.qaoa_solver import QaoaConfig, run_qaoa, decode_counts
from src.evaluation import evaluate_run, random_baseline
problem = default_problem()"""),
md("## 1. Input data (SYNTHETIC)\nTravel time = distance / 0.5 km/min, so coordinates and travel times agree by construction."),
code("problem.table()"),
md("""## 2. Mathematical formulation
Decision variable $x_i\\in\\{0,1\\}$ (team sent to location $i$), qubit $i$ = variable $x_i$, constraint $\\sum_i x_i = 3$.

Maximise $F(x)=\\sum_i s_i x_i - w_O\\sum_{i<j} O_{ij}x_ix_j$ with $s_i = w_P\\,P_i/\\max P + w_R\\,R_i/100 - w_T\\,T_i/\\max T$.

* **Model A**: no outbreak term. **Model B**: outbreak-aware (weights 0.5/0.3/0.2). **Model C**: B plus a pairwise *service-area overlap* penalty ($w_O=0.3$) that discourages sending two teams to nearby locations. C is our extension; with $w_O=0$ the problem is linear and trivially solved by sorting.

## 3. Exact classical baseline (all 120 deployments)"""),
code("""rows = []
for m in "ABC":
    r = solve_exact(problem, make_formulation(m))
    x = np.zeros(10, int); x[list(r.chosen_set)] = 1
    bd = objective_breakdown(problem, make_formulation(m), x)
    rows.append({"model": m, "candidates": r.num_candidates, "optimal": bd["selected"], "F*": round(r.best_value, 4),
                 "patients": bd["patients_covered"], "risk_sum": bd["outbreak_risk_sum"], "travel_min": bd["total_travel_min"],
                 "tie": r.has_tie})
pd.DataFrame(rows)"""),
md("The three formulations pick three different deployments, so outbreak-awareness and overlap really change the decision.\n\n## 4. QUBO and Ising Hamiltonian\n$H(x) = -F(x) + \\lambda(\\sum_i x_i-3)^2$, expanded with $x_i^2=x_i$; $x_i=(1-Z_i)/2$. $\\lambda$ is chosen above the exact threshold $\\lambda^*$ (below it an infeasible state would be the ground state)."),
code("""form = make_formulation("C")
lam_star, lam = exact_min_penalty(problem, form), default_penalty(problem, form)
print(f"lambda* = {lam_star:.4f}   lambda used = {lam:.4f}")
print(verify_model_consistency(problem, form, lam))
op = ising_to_operator(qubo_to_ising(build_qubo(problem, form, lam)))
print("Hamiltonian terms:", len(op), "Pauli terms on", op.num_qubits, "qubits")"""),
md("## 5. QAOA circuit (XY mixer + Dicke start keeps exactly 3 teams)"),
code("""ising = qubo_to_ising(build_qubo(problem, form, lam))
qc = build_qaoa_circuit(ising, 3, depth=1, mixer="xy")
print(qc.draw(output="text", fold=120, idle_wires=False).__str__()[:1500], "...")"""),
md("## 6. Run QAOA on the Aer simulator\nThe optimiser (COBYLA) is classical; every energy it sees comes from measured samples of the circuit."),
code("""exact = solve_exact(problem, form)
results = {}
for mixer in ("x", "xy"):
    res = run_qaoa(problem, form, QaoaConfig(depth=3, mixer=mixer, shots=2048, seed=0))
    results[mixer] = (res, evaluate_run(problem, form, exact, res))
summary = pd.DataFrame({mixer: {k: row[k] for k in ["feasibility_rate", "optimal_probability", "ideal_optimal_probability",
        "best_feasible_selection", "found_exact_optimum", "optimality_gap_range", "evaluations_cumulative", "runtime_total_s"]}
        for mixer, (_, row) in results.items()})
summary"""),
md("## 7. Interpret the output (decode Qiskit bitstrings: rightmost bit = qubit 0 = location L1)"),
code("""res, _ = results["xy"]
dec = decode_counts(res.counts, 10)
top = sorted(dec.items(), key=lambda kv: -kv[1])[:8]
pd.DataFrame([{"locations": ",".join(problem.names[i] for i in np.flatnonzero(x)), "count": c,
               "F": round(float(__import__('src.objective', fromlist=['x']).objective_value(problem, form, x)), 4),
               "feasible": sum(x) == 3} for x, c in top])"""),
md("## 8. Comparison with random sampling (a control, not a quantum method)"),
code("""rb = random_baseline(problem, form, exact, 2048, seed=0, feasible_only=True)
pd.DataFrame({"QAOA XY p=3": {"P(exact optimum)": results["xy"][1]["optimal_probability"]},
              "random feasible": {"P(exact optimum)": rb["optimal_probability"]},
              "uniform over 120": {"P(exact optimum)": 1/120}})"""),
md("## 9. Full experimental results (produced by `experiments/run_experiments.py`)"),
code("""res_dir = ROOT / "experiments" / "results"
from IPython.display import Image, display, Markdown
for f in ["summary_depth.csv", "summary_models_qaoa.csv"]:
    p = res_dir / f
    if p.exists():
        display(Markdown(f"**{f}**")); display(pd.read_csv(p).iloc[:, :8])
for f in ["fig1_instance_and_optima.png", "fig4_depth_study.png", "fig6_penalty_study.png", "fig8_output_distribution.png"]:
    if (res_dir / f).exists():
        display(Image(filename=str(res_dir / f)))"""),
md("""## 10. Limitations (honest summary)
* 10 qubits are classically trivial (120 candidates); **no quantum advantage is claimed or shown**.
* Data are synthetic; weights are modelling choices; one depot, interchangeable teams, no routing.
* Models A and B are linear (optimum = top-3 scores). Model C adds a genuine quadratic term; this extension is ours and its parameters ($w_O$, 15 km reference) are assumptions.
* QAOA with the X mixer + penalty often samples infeasible states; the XY mixer fixes feasibility but costs a deeper circuit (Dicke state preparation by amplitude loading).
* Results are noiseless-simulator results with a local optimiser; hardware noise was not studied."""),
]
Path(__file__).resolve().parents[1].joinpath("notebooks").mkdir(exist_ok=True)
nbf.write(nb, Path(__file__).resolve().parents[1] / "notebooks" / "experimentation.ipynb")
print("wrote notebook")
