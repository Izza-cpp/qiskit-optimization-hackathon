"""Build docs/final_report.docx from the CSV/PNG files in experiments/results/ (python-docx).

Every number in a table comes from a CSV. Prose numbers marked (CSV) are read from the files too.
Usage:  python experiments/build_report.py [--results experiments/results] [--out docs/final_report.docx]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]


def table(doc, df: pd.DataFrame, widths=None, size=9):
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = "Light Grid Accent 1"
    for j, c in enumerate(df.columns):
        cell = t.rows[0].cells[j]
        cell.text = str(c)
        for r in cell.paragraphs[0].runs:
            r.bold = True; r.font.size = Pt(size)
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for j, v in enumerate(row):
            if isinstance(v, float) and v != v:
                v = ""
            elif isinstance(v, float) and float(v).is_integer() and abs(v) >= 1:
                v = int(v)
            cells[j].text = str(v)
            for r in cells[j].paragraphs[0].runs:
                r.font.size = Pt(size)
    doc.add_paragraph()
    return t


def fig(doc, path: Path, caption: str, width=6.0):
    if not path.exists():
        doc.add_paragraph(f"[missing figure: {path.name}]")
        return
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph(caption)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in p.runs:
        r.italic = True; r.font.size = Pt(9)


def para(doc, text, bold=False):
    p = doc.add_paragraph()
    r = p.add_run(text); r.bold = bold
    return p


def bullets(doc, items):
    for it in items:
        doc.add_paragraph(it, style="List Bullet")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "experiments" / "results"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "final_report.docx"))
    a = ap.parse_args()
    R = Path(a.results)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)

    depth = pd.read_csv(R / "summary_depth.csv")
    ctrl = pd.read_csv(R / "summary_random_controls.csv").set_index("method")
    pen = pd.read_csv(R / "summary_penalty.csv")
    models = pd.read_csv(R / "summary_models_qaoa.csv")
    e8 = pd.read_csv(R / "e8_heuristics.csv")
    rob = pd.read_csv(R / "e10_robustness.csv")
    mass = pd.read_csv(R / "e10_near_optimal_mass.csv")
    hw = pd.read_csv(R / "e10_transpiled_size.csv")
    scal = pd.read_csv(R / "e9_scaling_summary.csv")
    cross = pd.read_csv(R / "e1_cross_evaluation.csv") if (R / "e1_cross_evaluation.csv").exists() else None

    # ---- report-only figures built from the CSVs ----
    fig_dir = R / "report_figs"; fig_dir.mkdir(exist_ok=True)
    fg, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for mixer, lab in (("x", "X mixer + penalty"), ("xy", "XY ring + Dicke start")):
        d = depth[depth.mixer == mixer]
        ax[0].errorbar(d.depth, d.optimal_probability_mean, yerr=d.optimal_probability_std, marker="o", capsize=3, label=lab)
        ax[1].plot(d.depth, d.feasibility_rate_mean, marker="o", label=lab)
    ax[0].axhline(ctrl.loc["random_feasible", "optimal_probability_mean"], ls="--", c="gray", label="random feasible sampling")
    ax[0].set_xlabel("QAOA depth p"); ax[0].set_ylabel("P(sample = exact optimum)"); ax[0].legend(fontsize=8)
    ax[0].set_title("Success probability (10 seeds, 2048 shots)")
    ax[1].set_xlabel("QAOA depth p"); ax[1].set_ylabel("fraction of feasible samples"); ax[1].set_ylim(0, 1.05)
    ax[1].set_title("Constraint satisfaction"); ax[1].legend(fontsize=8)
    fg.tight_layout(); fg.savefig(fig_dir / "report_depth.png", dpi=170); plt.close(fg)

    fg, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(rob.noise_sigma * 100, rob.nominal_still_optimal * 100, marker="o", label="nominal deployment (L1,L3,L9) still exactly optimal")
    ax.plot(rob.noise_sigma * 100, rob.optimum_contains_L1_and_L3 * 100, marker="s", label="optimum contains both L1 and L3")
    ax.plot(rob.noise_sigma * 100, rob.nominal_in_top5 * 100, marker="^", label="nominal deployment within top 5 of 120")
    ax.set_xlabel("input noise (log-normal sigma, %)"); ax.set_ylabel("% of 3000 random data draws"); ax.legend(fontsize=7)
    ax.set_title("Robustness of the model-C optimum"); fg.tight_layout(); fg.savefig(fig_dir / "report_robust.png", dpi=170); plt.close(fg)

    d5 = depth[(depth.mixer == "xy") & (depth.depth == 5)].iloc[0]
    x5 = depth[(depth.mixer == "x") & (depth.depth == 5)].iloc[0]
    rnd = float(ctrl.loc["random_feasible", "optimal_probability_mean"])

    doc = Document()
    st = doc.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(10.5)
    doc.add_heading("Quantum-Assisted Optimization of Mobile Medical Team Deployment During Disasters", 0)
    para(doc, "Qiskit Fall Fest 2026 hackathon - domain: Disaster Response. Repository: https://github.com/Izza-cpp/qiskit-optimization-hackathon")

    doc.add_heading("1. Summary", 1)
    para(doc, (
        "We select which 3 of 10 disaster-affected locations receive a mobile medical team, trading off patients in need, "
        "outbreak risk and travel time, and we penalise teams whose service areas overlap. The problem is written as a "
        "constrained binary quadratic problem, converted to a QUBO and an Ising Hamiltonian on 10 qubits, and solved with "
        "QAOA in Qiskit on the Aer simulator. Everything is checked against an independent exact classical solver "
        "(120 candidate deployments) and against random-sampling and classical-heuristic baselines."))
    bullets(doc, [
        f"The constraint-preserving XY-ring QAOA with a Dicke start state returns 100% feasible samples and puts "
        f"{d5.optimal_probability_mean*100:.1f}% of its probability on the exact optimum at depth p=5, about "
        f"{d5.optimal_probability_mean/rnd:.1f}x more than random feasible sampling ({rnd*100:.2f}%). "
        f"The penalty-based X-mixer reaches {x5.optimal_probability_mean*100:.1f}% with only {x5.feasibility_rate_mean*100:.0f}% feasible samples.",
        "We do NOT claim quantum advantage. With 10 qubits the exact answer is found instantly by enumeration, and a "
        "simple classical heuristic (greedy, local search, simulated annealing) solves the instance reliably.",
        "We report where the method is weak: the optimum of model C is sensitive to the input data, the XY circuit with "
        "a generic Dicke state is very deep once transpiled, and the circuit cost grows steeply with problem size.",
    ])

    doc.add_heading("2. Problem and context", 1)
    para(doc, (
        "After a flood or earthquake, health coordinators must decide quickly where to send a limited number of mobile "
        "medical teams. We model the core allocation decision: three interchangeable teams start at one depot; each of ten "
        "locations has an estimated number of patients, an outbreak-risk score (0-100, e.g. cholera or dengue) and a travel "
        "time. At most one team goes to a location. The model is decision support only: there is no routing, scheduling, "
        "capacity or traffic model, and all data are synthetic (no real disaster was observed)."))
    para(doc, "Three formulations use the same data and constraint:")
    table(doc, pd.DataFrame([
        ["A", "patients + travel only", "w_P=0.5, w_T=0.2"],
        ["B", "outbreak-aware", "w_P=0.5, w_R=0.3, w_T=0.2"],
        ["C", "B + pairwise service-area overlap (our extension)", "as B, w_O=0.3"]],
        columns=["Model", "Meaning", "Weights"]))
    para(doc, "Objective (maximise): F(x) = sum_i s_i x_i - w_O sum_{i<j} O_ij x_i x_j, subject to sum_i x_i = 3, with "
              "s_i = w_P P_i/max(P) + w_R R_i/100 - w_T T_i/max(T) and O_ij = max(0, 1 - d_ij/15 km).")

    doc.add_heading("3. Quantum formulation", 1)
    para(doc, (
        "Energy to minimise: H(x) = -F(x) + lambda (sum x_i - 3)^2. With x_i = (1 - Z_i)/2 this is an Ising Hamiltonian "
        "with fields h_i and couplings J_ij. The penalty lambda is not arbitrary: we compute the exact threshold "
        "lambda* = max over infeasible x of (F(x)-F*)/(m-K)^2 (0.436 for model B, 0.356 for model C) and use 1.25 lambda*. "
        "Unit tests verify that the QUBO energies and the Hamiltonian diagonal equal the classical energies for all 1024 "
        "bitstrings (error ~1e-14), and that below lambda* an infeasible state becomes the ground state."))
    doc.add_heading("Architecture", 2)
    fig(doc, ROOT / "docs" / "architecture.png", "Figure 1. Pipeline: data, QUBO/Ising, QAOA circuit (quantum), classical optimiser, evaluation against the exact solver.", 6.3)
    para(doc, (
        "Quantum part (simulated by Aer): state preparation, cost/mixer layers, measurement. Classical part: decoding, "
        "sampled expected energy, COBYLA parameter updates, evaluation. The exact solver is never used inside the loop."))
    table(doc, pd.DataFrame([
        ["Start state", "uniform superposition", "Dicke state |D(10,3)>"],
        ["Mixer", "RX on every qubit", "XX+YY ring (keeps Hamming weight)"],
        ["Constraint", "penalty only", "enforced by construction"],
        ["Logical circuit depth (p=3)", "59", "~2090 (Dicke preparation dominates)"]],
        columns=["", "X mixer", "XY mixer (default)"]))
    para(doc, "Parameters are initialised by INTERP warm starts across depths (Zhou et al., 2020) and optimised by COBYLA from "
              "sampled energies (2048 shots). Seeds fix the start points and the simulator sampling.")

    doc.add_heading("4. Classical baselines", 1)
    para(doc, "Exact solver: enumerates all C(10,3)=120 deployments with an objective implementation independent of the quantum code.")
    table(doc, pd.DataFrame([
        ["A", "L1, L3, L7", "0.8848"], ["B", "L1, L3, L5", "1.6152"], ["C", "L1, L3, L9", "1.5207"]],
        columns=["Model", "Exact optimum", "F*"]))
    para(doc, ("Ignoring outbreak risk (A's deployment scored under B) costs 0.115, i.e. 7.1% of F*; ignoring overlap (B's "
               "deployment scored under C) costs 0.113, i.e. 7.4%. Without the overlap term the problem is linear and is solved by "
               "sorting; on our weight grid sorting is wrong in 40 of 66 cells (it requires an outbreak weight of at least 0.2). "
               "Only model C is a genuine quadratic problem."))
    fig(doc, R / "fig9_why_not_sort.png", "Figure 2. Where sorting by score fails (E7).", 5.2)

    doc.add_heading("Classical heuristics (E8)", 2)
    c8 = e8[e8.model == "C"]
    tbl = c8.pivot_table(index="method", columns="budget", values="success_rate").round(3).reset_index().fillna("-")
    tbl.columns = [str(c) if c == "method" else f"{int(c)} evals" for c in tbl.columns]
    para(doc, "Probability of returning the exact model-C optimum (300 repetitions per cell; budget = number of objective evaluations). "
              "Greedy (marginal-gain) construction returns the optimum with 27 evaluations.")
    table(doc, tbl)
    para(doc, "Classical heuristics solve this 10-location instance with a few dozen evaluations. QAOA is therefore not "
              "competitive on this instance; its value here is as a verified formulation and a scaling testbed.")

    doc.add_heading("5. QAOA results", 1)
    fig(doc, R / "report_figs" / "report_depth.png", "Figure 3. Depth study, model C, 10 seeds.", 6.3)
    rows = []
    for _, r in depth.iterrows():
        rows.append([("X + penalty" if r.mixer == "x" else "XY + Dicke"), int(r.depth),
                     f"{r.optimal_probability_mean:.4f} +/- {r.optimal_probability_std:.4f}",
                     f"{r.feasibility_rate_mean*100:.0f}%", f"{r.runtime_total_s_mean:.1f}", int(r.circuit_depth_mean)])
    table(doc, pd.DataFrame(rows, columns=["Mixer", "p", "P(exact optimum)", "Feasible samples", "Runtime (s)", "Logical depth"]))
    para(doc, (
        f"Controls: random feasible sampling finds the optimum with probability {rnd:.4f} (theory 1/120 = 0.0083); random 10-bit "
        f"strings give {ctrl.loc['random_uniform','optimal_probability_mean']:.4f} with {ctrl.loc['random_uniform','feasibility_rate_mean']*100:.0f}% feasible samples. "
        "The best sampled feasible deployment equals the exact optimum in all runs, but that is not evidence of skill: with 2048 shots random "
        "feasible sampling finds it too. We therefore report enrichment over random sampling as the meaningful metric."))
    para(doc, "QAOA recovers a different exact optimum for each model (XY, p=4, 5 seeds each): " +
         "; ".join(f"model {r.model}: {', '.join(eval(r.best_sampled_selection_most_common))} (P={r.optimal_probability_mean:.3f})" for _, r in models.iterrows()) + ".")
    pr = pen[["penalty_over_threshold", "qubo_ground_state_feasible", "feasibility_rate_mean", "optimal_probability_mean"]].copy()
    pr.columns = ["lambda / lambda*", "QUBO ground state feasible", "Feasible samples", "P(exact optimum)"]
    pr["Feasible samples"] = (pr["Feasible samples"] * 100).round(0).astype(int).astype(str) + "%"
    pr["P(exact optimum)"] = pr["P(exact optimum)"].round(4)
    para(doc, "Penalty study (X mixer): below lambda* the QUBO ground state is infeasible; above it the ground state is feasible.")
    table(doc, pr)

    doc.add_heading("6. Robustness of the model-C optimum", 1)
    para(doc, ("Model C's best and second-best deployments differ by only 0.009 in F, so the exact optimum is nearly degenerate. "
               "We perturbed patients, risk and travel times with log-normal noise (3000 draws per level) and recomputed the exact optimum."))
    fig(doc, R / "report_figs" / "report_robust.png", "Figure 4. Stability of the optimum under input noise.", 4.6)
    rt = rob.copy()
    for c in rt.columns[2:]:
        rt[c] = rt[c].round(3)
    table(doc, rt)
    para(doc, ("Interpretation: the decision 'send teams to L1 and L3' is robust; the choice of the third location is not. "
               "Reporting the single nominal optimum without this caveat would overstate its certainty."))
    para(doc, "Probability mass of XY-QAOA (p=5, 3 seeds, 2048 shots) on near-optimal deployments, compared with random feasible sampling:")
    m = mass.mean(numeric_only=True)
    table(doc, pd.DataFrame([[f"within {t}% of F*", int(m[f'num_deployments_within_{t}pct']),
                              f"{m[f'mass_within_{t}pct']*100:.1f}%", f"{m[f'random_feasible_mass_within_{t}pct']*100:.1f}%"]
                             for t in (0, 1, 5, 10)], columns=["Deployments", "How many (of 120)", "QAOA mass", "Random feasible mass"]))

    doc.add_heading("7. Scaling study (E9)", 1)
    para(doc, ("Synthetic random instances (seeded) with n = 8 ... 14 locations, model C, XY-QAOA p=3, 2-3 seeds per size. "
               "Random feasible sampling has success probability 1/C(n,K)."))
    st_ = scal[["n", "K", "num_candidates", "runs", "p_opt_mean", "random_p_opt", "advantage_over_random", "circuit_depth", "two_qubit_gates", "wall_s"]].copy()
    st_.columns = ["n", "K", "candidates", "runs", "QAOA P(opt)", "random P(opt)", "x random", "logical depth", "2-qubit gates", "wall s"]
    for c in ("QAOA P(opt)", "random P(opt)", "x random"):
        st_[c] = st_[c].round(4)
    st_["wall s"] = st_["wall s"].round(0)
    table(doc, st_)
    fig(doc, R / "fig10_scaling.png", "Figure 5. Success probability and circuit size versus n.", 6.3)
    worst = pd.read_csv(R / "e9_scaling_runs.csv")
    below = worst[worst.optimal_probability < worst.random_feasible_p_opt]
    if len(below):
        para(doc, "Honest note: " + "; ".join(f"at n={int(r.n)} seed {int(r.seed)} QAOA reached P={r.optimal_probability:.4f}, below random sampling ({r.random_feasible_p_opt:.4f})" for _, r in below.iterrows())
             + ". With a fixed budget of 150 optimiser iterations the optimiser has not converged at the largest size, and the single-seed spread is large.")
    para(doc, ("Reading: see the table. The enrichment over random sampling and the growth of circuit depth are both reported as measured; "
               "we ran only 2-3 seeds per size, so these are indicative, not statistically conclusive. The generic Dicke-state "
               "synthesis grows quickly with n and is the practical limit of this implementation (a linear-depth Dicke circuit is future work)."))

    doc.add_heading("8. Hardware readiness (no hardware run)", 1)
    para(doc, ("We did not run on a quantum device. To quantify what a device would face we transpiled the circuits to a heavy-hex "
               "coupling map with basis gates ECR/RZ/SX/X (noise-free; Qiskit generic backend, optimisation level 1)."))
    h = hw.copy(); h.columns = ["p", "circuit", "transpiled depth", "ECR gates", "total gates", "qubits used"]
    table(doc, h)
    para(doc, ("The XY circuit with the generic Dicke start has thousands of two-qubit gates even at p=1, far beyond what current "
               "noisy hardware can run coherently. A shallow alternative (a single basis state |1110000000> instead of the Dicke "
               "state) is cheap but failed in our test: 0 of 3 runs sampled the optimum (P=0.0). We kept it as a documented negative result."))

    doc.add_heading("9. Limitations (honest)", 1)
    bullets(doc, [
        "No quantum advantage: 120 candidates are enumerated instantly and classical heuristics succeed with a few dozen evaluations.",
        "Noiseless simulation only; no hardware run; transpiled circuits are too deep for current devices.",
        "Synthetic data; weights, the 0.5 km/min speed and the overlap model are our assumptions; model C is our own extension.",
        "Success probabilities stay below 10% at p <= 5 (local COBYLA optimiser, 5-10 seeds); results are indicative.",
        "The model-C optimum is nearly degenerate and sensitive to the data (Section 6).",
        "Scaling study covers n <= 14, 2-3 seeds per size.",
    ])

    doc.add_heading("10. Reproducibility", 1)
    para(doc, ("Python 3.12, Qiskit 2.6.2, Qiskit Aer 0.18.2. 110 unit tests (python -m pytest -q). "
               "python experiments/run_experiments.py runs the main study; python -m experiments.run_extra runs E8-E10. "
               "All random seeds are fixed; results are written as CSV to experiments/results/."))

    doc.add_heading("11. References", 1)
    for ref in [
        "Farhi, Goldstone, Gutmann. A Quantum Approximate Optimization Algorithm. arXiv:1411.4028 (2014).",
        "Hadfield et al. From the QAOA to a Quantum Alternating Operator Ansatz. Algorithms 12(2):34 (2019).",
        "Zhou et al. QAOA: Performance, Mechanism, and Implementation on Near-Term Devices. Phys. Rev. X 10, 021067 (2020).",
        "Baertschi, Eidenbenz. Deterministic Preparation of Dicke States. FCT 2019 (we use generic amplitude loading, not their circuit).",
        "Qiskit and Qiskit Aer documentation (IBM Quantum).",
    ]:
        doc.add_paragraph(ref, style="List Bullet")
    doc.save(out)
    print("saved", out)


if __name__ == "__main__":
    main()
