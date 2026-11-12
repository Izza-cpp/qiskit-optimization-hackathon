"""Draws docs/architecture.png from the real module structure. Run from the repo root:
    python experiments/make_architecture.py
Only components that exist in this repository are drawn."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parents[1] / "docs" / "architecture.png"
CLASSICAL, QUANTUM, DATA = "#DCE9F5", "#FBE3B8", "#E3F2E1"
RED = "#B00020"

fig, ax = plt.subplots(figsize=(15, 9))
ax.set_xlim(0, 15.2); ax.set_ylim(0, 9); ax.axis("off")


def box(x, y, w, h, title, sub, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12", fc=color, ec="#333", lw=1.3))
    ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="center", fontsize=10.5, weight="bold")
    ax.text(x + w / 2, y + (h - 0.55) / 2, sub, ha="center", va="center", fontsize=8.2, linespacing=1.35)


def line(points, color="#222", arrow=True, lw=1.6):
    xs, ys = zip(*points)
    ax.plot(xs[:-1] if arrow else xs, ys[:-1] if arrow else ys, color=color, lw=lw, solid_capstyle="round")
    if arrow:
        ax.annotate("", xy=points[-1], xytext=points[-2], arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))
        ax.plot(*zip(points[-2], points[-1]), color=color, lw=lw)


ax.text(7.6, 8.65, "Quantum-assisted medical-team deployment: architecture (as implemented)", ha="center",
        fontsize=14, weight="bold")

# Row A: modelling (y 6.6-7.9)
box(0.3, 6.6, 3.4, 1.3, "Input data", "src/problem.py\n10 locations (SYNTHETIC): patients,\ntravel time, outbreak risk, km", DATA)
box(4.3, 6.6, 3.4, 1.3, "Objective + constraint", "src/objective.py, src/constraints.py\nF(x), models A / B / C;  sum x_i = 3", CLASSICAL)
box(8.3, 6.6, 3.4, 1.3, "QUBO -> Ising Hamiltonian", "src/quantum_model.py\nH = -F + lambda (sum x - K)^2\nx = (1 - Z)/2 ;  10 qubits", CLASSICAL)
box(12.2, 6.6, 2.7, 1.3, "Exact baseline", "src/classical_solver.py\nall C(10,3) = 120 deployments\n(no quantum code used)", CLASSICAL)
line([(3.7, 7.25), (4.3, 7.25)]); line([(7.7, 7.25), (8.3, 7.25)])
line([(2.0, 7.9), (2.0, 8.35), (13.55, 8.35), (13.55, 7.9)])      # data also feeds the exact baseline
ax.text(7.8, 8.43, "same data", ha="center", fontsize=8, color="#444")

# Row B: quantum loop (y 3.8-5.1)
box(0.3, 3.8, 3.1, 1.3, "QAOA circuit", "build_qaoa_circuit(), p layers\ncost: RZ + RZZ\nmixer: RX  |  XY ring (Dicke start)", QUANTUM)
box(3.9, 3.8, 3.1, 1.3, "Aer simulator", "qiskit_aer.AerSimulator\n(statevector method)\nshots -> measurement counts", QUANTUM)
box(7.5, 3.8, 3.1, 1.3, "Decode + score", "bitstring -> x (little-endian)\nsampled energy E(x)\nfeasibility check", CLASSICAL)
box(11.1, 3.8, 3.1, 1.3, "COBYLA optimiser", "src/qaoa_solver.py\nproposes (gamma, beta)\nINTERP warm start across depths", CLASSICAL)
for a, b in ((3.4, 3.9), (7.0, 7.5), (10.6, 11.1)):
    line([(a, 4.45), (b, 4.45)])
line([(10.0, 6.6), (10.0, 5.95), (1.2, 5.95), (1.2, 5.1)])
ax.text(5.6, 6.05, "cost Hamiltonian H_C", ha="center", fontsize=8.5)
line([(12.65, 5.1), (12.65, 5.5), (2.6, 5.5), (2.6, 5.1)], color=RED)
ax.text(7.8, 5.57, "classical feedback loop: new (gamma, beta) -> run the circuit again", ha="center", fontsize=8.5, color=RED, weight="bold")

# Row C: evaluation (y 0.9-2.2)
box(8.0, 0.9, 5.2, 1.3, "Evaluation", "src/evaluation.py\nfeasibility rate, P(exact optimum), optimality gap;\nbest measured vs best feasible vs exact", CLASSICAL)
box(2.6, 0.9, 4.6, 1.3, "Experiments + results", "experiments/run_experiments.py\n-> experiments/results/*.csv, *.png\n(all report numbers come from here)", DATA)
line([(9.05, 3.8), (9.05, 2.2)])
ax.text(9.15, 3.0, "final counts", fontsize=8, ha="left")
line([(14.55, 6.6), (14.55, 1.55), (13.2, 1.55)])
ax.text(14.45, 3.0, "exact optimum\n(reference)", fontsize=8, ha="right", va="center")
line([(8.0, 1.55), (7.2, 1.55)])

# legend
for x, c, t in ((0.3, QUANTUM, "quantum circuit / simulated execution"), (4.7, CLASSICAL, "classical processing"),
                (7.7, DATA, "data / outputs")):
    ax.add_patch(Rectangle((x, 0.18), 0.35, 0.22, fc=c, ec="#333"))
    ax.text(x + 0.5, 0.29, t, va="center", fontsize=8.5)
OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT, dpi=170, bbox_inches="tight")
print("wrote", OUT)
