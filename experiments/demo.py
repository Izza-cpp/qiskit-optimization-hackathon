import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
"""Short live demo for the video (about 30-40 s): exact answers, then one XY-QAOA run on model C.

Run from the repository root:   python experiments/demo.py
"""
import logging
import time

logging.getLogger().setLevel(logging.ERROR)

from src.classical_solver import solve_exact
from src.evaluation import evaluate_run
from src.objective import make_formulation
from src.problem import default_problem
from src.qaoa_solver import QaoaConfig, run_qaoa

problem = default_problem()
print("=" * 70)
print("Medical team deployment: 10 locations, 3 teams, 120 possible deployments")
print("=" * 70)

print("\n[1] EXACT classical optimum (enumerates all 120 deployments)")
for model in ("A", "B", "C"):
    exact = solve_exact(problem, make_formulation(model))
    names = ", ".join(problem.names[i] for i in exact.chosen_set)
    print(f"    Model {model}: {names:<12s} objective F* = {exact.best_value:.4f}")

form = make_formulation("C")
exact = solve_exact(problem, form)
print("\n[2] QAOA on Qiskit Aer (model C, XY-ring mixer + Dicke start, depth p=3, 2048 shots)")
t = time.time()
result = run_qaoa(problem, form, QaoaConfig(depth=3, mixer="xy", seed=0))
row = evaluate_run(problem, form, exact, result, "C")
print(f"    finished in {time.time() - t:.1f} s")
print(f"    feasible samples (exactly 3 teams): {row['feasibility_rate'] * 100:.1f}%")
print(f"    P(sample = exact optimum):          {row['optimal_probability'] * 100:.1f}%")
print(f"    random feasible sampling would get: {100 / exact.num_candidates:.1f}%")

n = problem.n
tally = {}
for bits, count in result.counts.items():
    sel = tuple(sorted(i for i in range(n) if bits[n - 1 - i] == "1"))
    if len(sel) == problem.num_teams:
        tally[sel] = tally.get(sel, 0) + count
total = sum(result.counts.values())
print("\n    Top 5 deployments sampled by QAOA:")
for sel, count in sorted(tally.items(), key=lambda kv: -kv[1])[:5]:
    names = ", ".join(problem.names[i] for i in sel)
    mark = "  <-- exact optimum" if sel == exact.chosen_set else ""
    print(f"      {names:<12s} {count / total * 100:5.1f}%   F = {exact.all_values[sel]:.4f}{mark}")

print("\n[3] Honest summary: QAOA is enriched over random sampling, but it does NOT beat")
print("    exact enumeration or simple classical heuristics at this size.")

