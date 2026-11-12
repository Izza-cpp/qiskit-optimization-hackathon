"""Reproducible experiments. Run from the repository root:

    python experiments/run_experiments.py            # full study (~6-8 min on 2 cores)
    python experiments/run_experiments.py --quick    # smoke test (~1-2 min)

Every number and figure written to experiments/results/ is produced by this script from real
Aer-simulator runs and exact classical enumeration. Nothing is typed in by hand.

Experiments
  E1  exact baseline for models A, B, C (and what each model's optimum scores under the others)
  E2  weight sensitivity (classical, exact): how often does the optimal deployment change?
  E3  QAOA depth study, model C, both mixers (X+penalty vs XY+Dicke), many seeds, random controls
  E4  shot-count study
  E5  penalty-strength study (X mixer): feasibility vs lambda / lambda*
  E7  'why not just sort?': exact optimum vs sort-and-take-top-K over overlap and risk weights
  E6  can QAOA recover the DIFFERENT optima of models A, B and C?
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")   # one thread per worker process

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import qiskit
import qiskit_aer
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classical_solver import solve_exact                                   # noqa: E402
from src.evaluation import aggregate, evaluate_run, random_baseline             # noqa: E402
from src.objective import (DEFAULT_OVERLAP_WEIGHT, DEFAULT_WEIGHTS_AB, make_formulation,   # noqa: E402
                           objective_breakdown, objective_value)
from src.problem import Weights, default_problem                                # noqa: E402
from src.qaoa_solver import QaoaConfig, run_qaoa_progressive                    # noqa: E402
from src.quantum_model import (all_selections, default_penalty, exact_min_penalty,   # noqa: E402
                               verify_model_consistency)

OUT = ROOT / "experiments" / "results"
log = logging.getLogger("experiments")

# Okabe-Ito colour-blind-safe palette
BLUE, ORANGE, GREEN, RED, GREY, SKY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#666666", "#56B4E9"
LABEL = {"xy": "XY mixer + Dicke start", "x": "X mixer + penalty"}
COLOR = {"xy": BLUE, "x": ORANGE}
PLOT_DEFAULTS = {"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
                 "axes.grid": True, "grid.alpha": 0.25, "font.size": 10}


# ----------------------------------------------------------------------------------------------
# worker (runs in a subprocess; must be a top-level function)
# ----------------------------------------------------------------------------------------------
def worker(task: dict) -> list:
    """Run one progressive QAOA chain and return evaluation rows (+ traces for trace seeds)."""
    problem = default_problem()
    form = make_formulation(task["model"])
    classical = solve_exact(problem, form)
    lam_star = exact_min_penalty(problem, form)
    mult = task.get("penalty_mult")
    penalty = mult * lam_star if mult is not None else default_penalty(problem, form)
    cfg = QaoaConfig(depth=task["depth"], mixer=task["mixer"], shots=task["shots"], seed=task["seed"],
                     penalty=penalty)
    stages = run_qaoa_progressive(problem, form, cfg)
    if task.get("last_only"):
        stages = stages[-1:]
    ground_feasible = verify_model_consistency(problem, form, penalty)["ground_state_is_feasible"]
    rows = []
    for res in stages:
        row = evaluate_run(problem, form, classical, res, model_label=task["model"])
        row.update({"experiment": task["experiment"], "penalty_lambda_star": lam_star,
                    "penalty_over_threshold": penalty / lam_star, "qubo_ground_state_feasible": ground_feasible})
        if task.get("trace"):
            row["_energy_history"] = res.energy_history
            row["_counts"] = res.counts
            row["_ideal_probabilities"] = res.ideal_probabilities.tolist()
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------------------------
# classical experiments
# ----------------------------------------------------------------------------------------------
def e1_exact(problem) -> pd.DataFrame:
    rows, optima = [], {}
    for m in "ABC":
        f = make_formulation(m)
        r = solve_exact(problem, f)
        optima[m] = r.chosen_set
        x = np.zeros(problem.n, dtype=int)
        x[list(r.chosen_set)] = 1
        bd = objective_breakdown(problem, f, x)
        ranking = r.ranking
        rows.append({
            "model": m, "weights_patients": f.weights.patients, "weights_outbreak": f.outbreak_weight,
            "weights_travel": f.weights.travel, "weights_overlap": f.weights.overlap,
            "optimal_selection": " ".join(bd["selected"]), "candidates_evaluated": r.num_candidates,
            "optimal_F": r.best_value, "second_best_F": ranking[1][1],
            "gap_to_second": r.best_value - ranking[1][1], "ties": r.has_tie,
            "patients_covered": bd["patients_covered"], "outbreak_risk_sum": bd["outbreak_risk_sum"],
            "total_travel_min": bd["total_travel_min"], "overlap_sum": bd["overlap_sum"],
        })
    df = pd.DataFrame(rows)
    # cross-evaluation: score every model's optimum under every objective
    cross = []
    for m_opt, sel in optima.items():
        x = np.zeros(problem.n, dtype=int)
        x[list(sel)] = 1
        for m_eval in "ABC":
            fe = make_formulation(m_eval)
            best = solve_exact(problem, fe).best_value
            val = objective_value(problem, fe, x)
            cross.append({"deployment_optimal_for": m_opt, "scored_under": m_eval,
                          "selection": " ".join(problem.names[i] for i in sel), "F": val,
                          "F_optimal_under_that_model": best, "regret": best - val})
    pd.DataFrame(cross).to_csv(OUT / "e1_cross_evaluation.csv", index=False)
    # full ranking of all 120 deployments for each model
    allrows = []
    for m in "ABC":
        r = solve_exact(problem, make_formulation(m))
        for rank, (sel, val) in enumerate(r.ranking, 1):
            allrows.append({"model": m, "rank": rank, "selection": " ".join(problem.names[i] for i in sel), "F": val})
    pd.DataFrame(allrows).to_csv(OUT / "e1_all_120_deployments.csv", index=False)
    df.to_csv(OUT / "e1_exact_baseline.csv", index=False)
    return df


def e2_sensitivity(problem, step: float = 0.1) -> pd.DataFrame:
    """Exact optimum over a grid of weights on the simplex w_P + w_R + w_T = 1."""
    ticks = np.round(np.arange(0, 1 + 1e-9, step), 10)
    rows = []
    for model in ("A", "B", "C"):
        for wp, wr in product(ticks, ticks):
            wt = round(1 - wp - wr, 10)
            if wt < -1e-12:
                continue
            wt = max(wt, 0.0)
            if model == "A" and wr != 0:
                continue                       # A has no outbreak term: only the w_R = 0 line is meaningful
            base = Weights(patients=float(wp), outbreak=float(wr), travel=float(wt))
            f = make_formulation(model, base, overlap=DEFAULT_OVERLAP_WEIGHT)
            r = solve_exact(problem, f)
            rows.append({"model": model, "w_patients": wp, "w_outbreak": wr, "w_travel": wt,
                         "optimal_selection": " ".join(problem.names[i] for i in r.chosen_set),
                         "optimal_F": r.best_value, "tie": r.has_tie})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "e2_weight_sensitivity.csv", index=False)
    return df


def e7_why_not_sort(problem) -> pd.DataFrame:
    """Does the optimum differ from 'sort by score and take the top K'? Sweep overlap weight and risk weight.

    Sorting ignores the pairwise overlap term, so it is exact only when w_O = 0. This experiment measures
    how far (regret under the full objective) the sorted answer is from the exact optimum.
    """
    from src.objective import location_scores
    rows = []
    for w_r in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        for mu in np.round(np.arange(0.0, 1.0001, 0.1), 10):
            base = Weights(patients=round(0.8 - w_r, 10), outbreak=w_r, travel=0.2)
            model = "C" if mu > 0 else "B"
            f = make_formulation(model, base, overlap=float(mu))
            exact = solve_exact(problem, f)
            top = tuple(sorted(int(i) for i in np.argsort(-location_scores(problem, f))[:problem.num_teams]))
            x = np.zeros(problem.n, dtype=int)
            x[list(top)] = 1
            f_sort = objective_value(problem, f, x)
            rows.append({"w_outbreak": w_r, "w_patients": base.patients, "overlap_weight": float(mu),
                         "exact_optimum": " ".join(problem.names[i] for i in exact.chosen_set),
                         "sorted_top_k": " ".join(problem.names[i] for i in top),
                         "sorting_is_wrong": top != exact.chosen_set, "F_exact": exact.best_value,
                         "F_sorted": f_sort, "regret_of_sorting": exact.best_value - f_sort})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "e7_why_not_sort.csv", index=False)
    return df


def fig_why_not_sort(df: pd.DataFrame) -> None:
    plt.rcParams.update(PLOT_DEFAULTS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    piv = df.pivot(index="w_outbreak", columns="overlap_weight", values="sorting_is_wrong").astype(float)
    im = axes[0].imshow(piv.values, origin="lower", cmap="Blues", vmin=0, vmax=1.3, aspect="auto")
    axes[0].set_xticks(range(len(piv.columns)))
    axes[0].set_xticklabels([f"{c:.1f}" for c in piv.columns])
    axes[0].set_yticks(range(len(piv.index)))
    axes[0].set_yticklabels([f"{c:.1f}" for c in piv.index])
    axes[0].set_xlabel("overlap weight $w_O$")
    axes[0].set_ylabel("outbreak weight $w_R$")
    axes[0].set_title("dark = 'sort and take top 3' gives a WRONG deployment")
    axes[0].grid(False)
    d = df[np.isclose(df.w_outbreak, 0.3)]
    axes[1].plot(d.overlap_weight, d.regret_of_sorting, marker="o", color=RED)
    axes[1].set_xlabel("overlap weight $w_O$")
    axes[1].set_ylabel("F(exact) - F(sorted answer)")
    axes[1].set_title("regret of sorting (weights 0.5 / 0.3 / 0.2)")
    fig.tight_layout()
    fig.savefig(OUT / "fig9_why_not_sort.png")
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------------
def fig_map(problem, e1: pd.DataFrame) -> None:
    plt.rcParams.update(PLOT_DEFAULTS)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    xy = problem.coordinates_km
    ax.scatter([0], [0], marker="s", s=140, color="black", zorder=5)
    ax.annotate("Depot", (0, 0), textcoords="offset points", xytext=(8, -12), fontsize=9)
    sizes = 40 + 6 * problem.patients
    ax.scatter(xy[:, 0], xy[:, 1], s=sizes, c=problem.outbreak_risk, cmap="YlOrRd", edgecolor="black",
               vmin=0, vmax=100, zorder=3)
    for i, nm in enumerate(problem.names):
        ax.annotate(nm, xy[i], textcoords="offset points", xytext=(7, 7), fontsize=9)
    cols = {"A": SKY, "B": GREEN, "C": BLUE}
    handles = []
    for m, off in (("A", 1.0), ("B", 1.0), ("C", 1.0)):
        sel = e1.loc[e1.model == m, "optimal_selection"].iloc[0].split()
        idx = [problem.names.index(s) for s in sel]
        ax.scatter(xy[idx, 0], xy[idx, 1], s=sizes[idx] + 330 * {"A": 1.5, "B": 1.0, "C": 0.55}[m], facecolors="none",
                   edgecolors=cols[m], linewidths=2.6, zorder=2)
        handles.append(Line2D([0], [0], marker="o", mfc="none", mec=cols[m], mew=2.4, ms=11, ls="",
                              label=f"Model {m} optimum: {', '.join(sel)}"))
    ax.set_aspect("equal")
    ax.set_xlabel("x (km from depot)")
    ax.set_ylabel("y (km from depot)")
    ax.set_title("Synthetic instance: marker size = patients, colour = outbreak risk\nrings = exact optimum of each model")
    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_instance_and_optima.png")
    plt.close(fig)


def fig_landscape(problem) -> None:
    plt.rcParams.update(PLOT_DEFAULTS)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for m, c in (("A", SKY), ("B", GREEN), ("C", BLUE)):
        vals = [v for _, v in solve_exact(problem, make_formulation(m)).ranking]
        ax.plot(range(1, 121), vals, color=c, lw=1.8, label=f"Model {m}")
        ax.scatter([1], [vals[0]], color=c, zorder=4)
    ax.set_xlabel("rank of the deployment (1 = exact optimum)")
    ax.set_ylabel("objective F (higher is better)")
    ax.set_title("All 120 feasible deployments, ranked (exact enumeration)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig2_exact_landscape.png")
    plt.close(fig)


def fig_sensitivity(sens: pd.DataFrame) -> None:
    plt.rcParams.update(PLOT_DEFAULTS)
    models = [m for m in ("B", "C")]
    all_sets = sorted(sens[sens.model.isin(models)].optimal_selection.unique())
    palette = [BLUE, ORANGE, GREEN, RED, SKY, "#CC79A7", "#F0E442", GREY, "#000000", "#8C564B", "#17BECF", "#BCBD22"]
    cmap = {s: palette[i % len(palette)] for i, s in enumerate(all_sets)}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5), sharey=True)
    for ax, m in zip(axes, models):
        d = sens[sens.model == m]
        for s, g in d.groupby("optimal_selection"):
            ax.scatter(g.w_outbreak, g.w_patients, s=230, marker="s", color=cmap[s], edgecolor="white",
                       label=s.replace(" ", ","))
        ax.set_xlabel("outbreak weight $w_R$  (travel weight = 1 - $w_P$ - $w_R$)")
        ax.set_title(f"Model {m}: exact optimum over the weight simplex ({d.optimal_selection.nunique()} distinct)")
        ax.legend(fontsize=7, loc="upper right", ncol=2, title="deployment", title_fontsize=7)
        ax.set_xlim(-0.06, 1.06)
        ax.set_ylim(-0.06, 1.06)
    axes[0].set_ylabel("patient weight $w_P$")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_weight_sensitivity.png")
    plt.close(fig)


def _band(ax, df, col, mixer, **kw):
    g = df[df.mixer == mixer].groupby("depth")[col].agg(["mean", "std"]).fillna(0)
    ax.errorbar(g.index, g["mean"], yerr=g["std"], marker="o", color=COLOR[mixer], capsize=3, label=LABEL[mixer], **kw)


def fig_depth(depth_df: pd.DataFrame, random_row: dict) -> None:
    plt.rcParams.update(PLOT_DEFAULTS)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    for m in ("x", "xy"):
        _band(axes[0], depth_df, "ideal_optimal_probability", m)
        _band(axes[1], depth_df, "ideal_feasibility", m)
        _band(axes[2], depth_df, "optimality_gap_range", m)
        g = depth_df[depth_df.mixer == m].groupby("depth")["found_exact_optimum"].mean()
        axes[3].plot(g.index, g.values, marker="o", color=COLOR[m], label=LABEL[m])
    axes[0].axhline(1 / 120, color=GREY, ls="--", label="uniform over 120 feasible (1/120)")
    axes[1].axhline(120 / 1024, color=GREY, ls="--", label="uniform over 1024 bitstrings")
    axes[0].set_title("P(exact optimum), ideal circuit")
    axes[1].set_title("P(feasible), ideal circuit")
    axes[2].set_title("range-normalised gap of best feasible sample\n(0 = optimum found)")
    axes[3].set_title("fraction of runs whose samples\ncontain the exact optimum")
    for a in axes:
        a.set_xlabel("QAOA depth p")
        a.set_xticks(sorted(depth_df.depth.unique()))
    axes[0].set_ylabel("mean ± std over seeds")
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    axes[3].set_ylim(-0.05, 1.05)
    fig.suptitle("Depth study (model C, 2048 shots)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_depth_study.png", bbox_inches="tight")
    plt.close(fig)


def fig_shots(shots_df: pd.DataFrame) -> None:
    plt.rcParams.update(PLOT_DEFAULTS)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for m in ("x", "xy"):
        d = shots_df[shots_df.mixer == m]
        for ax, col in zip(axes, ("optimal_probability", "feasibility_rate", "optimality_gap_range")):
            g = d.groupby("shots")[col].agg(["mean", "std"]).fillna(0)
            ax.errorbar(g.index, g["mean"], yerr=g["std"], marker="o", capsize=3, color=COLOR[m], label=LABEL[m])
    for ax, t in zip(axes, ("sampled P(exact optimum)", "sampled feasibility rate", "range-normalised gap")):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("shots (optimisation AND final sampling)")
        ax.set_title(t)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Shot-count study (model C, depth {int(shots_df.depth.max())})", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_shots_study.png", bbox_inches="tight")
    plt.close(fig)


def fig_penalty(pen_df: pd.DataFrame) -> None:
    plt.rcParams.update(PLOT_DEFAULTS)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, col, t in zip(axes, ("feasibility_rate", "optimal_probability", "found_exact_optimum"),
                          ("sampled feasibility rate", "sampled P(exact optimum)", "runs whose best FEASIBLE sample is the optimum")):
        g = pen_df.groupby("penalty_over_threshold")[col].agg(["mean", "std"]).fillna(0)
        ax.errorbar(g.index, g["mean"], yerr=g["std"], marker="o", capsize=3, color=ORANGE)
        ax.axvline(1.0, color=RED, ls="--", label=r"$\lambda^*$ (exact threshold)")
        ax.set_xscale("log", base=2)
        ax.set_xlabel(r"penalty $\lambda$ / $\lambda^*$")
        ax.set_title(t)
    axes[0].legend(fontsize=8)
    fig.suptitle(r"Penalty-strength study (X mixer, model C): below $\lambda^*$ the QUBO ground state is INFEASIBLE", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_penalty_study.png", bbox_inches="tight")
    plt.close(fig)


def fig_convergence(traces: dict) -> None:
    plt.rcParams.update(PLOT_DEFAULTS)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=False)
    for ax, m in zip(axes, ("x", "xy")):
        for (mixer, depth), hist in sorted(traces.items()):
            if mixer == m:
                ax.plot(hist, lw=1.4, label=f"p = {depth}")
        ax.set_title(f"{LABEL[m]}: sampled $\\langle H_C \\rangle$ per COBYLA evaluation")
        ax.set_xlabel("circuit-evaluation count within the stage")
        ax.set_ylabel("sampled energy (lower is better)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig7_convergence.png")
    plt.close(fig)


def fig_distribution(problem, trace_rows: dict, optimal_index: int, energies_feasible: np.ndarray) -> None:
    """Output distribution of the final circuits: ideal (exact) vs sampled, top states."""
    plt.rcParams.update(PLOT_DEFAULTS)
    xs = all_selections(problem.n)
    feasible = xs.sum(axis=1) == problem.num_teams
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    for ax, m in zip(axes, ("x", "xy")):
        row = trace_rows[m]
        probs = np.array(row["_ideal_probabilities"])
        n_shots = sum(row["_counts"].values())
        sampled = np.zeros(1024)
        for b, c in row["_counts"].items():
            sampled[int(b, 2)] += c / n_shots
        top = np.argsort(-probs)[:14]
        pos = np.arange(len(top))
        colors = [GREEN if k == optimal_index else (BLUE if feasible[k] else RED) for k in top]
        ax.bar(pos - 0.2, probs[top], width=0.4, color=colors, label="ideal (exact statevector)")
        ax.bar(pos + 0.2, sampled[top], width=0.4, color=colors, alpha=0.45, hatch="//", label="sampled counts")
        ax.set_xticks(pos)
        ax.set_xticklabels(["".join(problem.names[i][1:] + "," for i in np.flatnonzero(xs[k]))[:-1] for k in top],
                           rotation=60, ha="right", fontsize=8)
        ax.set_ylabel("probability")
        ax.set_title(f"{LABEL[m]}, p={row['depth']}: 14 most likely outputs\n"
                     f"green = exact optimum, blue = feasible, red = infeasible")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig8_output_distribution.png")
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small smoke-test configuration")
    ap.add_argument("--classical-only", action="store_true", help="run only the exact/classical experiments (E1, E2, E7)")
    ap.add_argument("--workers", type=int, default=max(1, min(os.cpu_count() or 1, 4)))
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    log.setLevel(logging.INFO)
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    cfg = dict(
        depth_max=2 if args.quick else 5, depth_seeds=range(2 if args.quick else 10),
        shots_list=[256, 1024] if args.quick else [128, 512, 2048, 8192], shots_depth=2 if args.quick else 3,
        shots_seeds=range(2 if args.quick else 5),
        pen_mults=[0.5, 1.0, 2.0] if args.quick else [0.25, 0.5, 0.8, 1.0, 1.25, 2.0, 4.0],
        pen_seeds=range(2 if args.quick else 5), pen_depth=2 if args.quick else 3,
        cmp_seeds=range(2 if args.quick else 5), cmp_depth=2 if args.quick else 4,
        base_shots=1024 if args.quick else 2048,
    )

    problem = default_problem()
    log.info("E1 exact baseline")
    e1 = e1_exact(problem)
    print(e1[["model", "optimal_selection", "optimal_F", "gap_to_second", "patients_covered",
              "outbreak_risk_sum", "total_travel_min", "overlap_sum"]].to_string(index=False))
    log.info("E2 weight sensitivity (classical)")
    sens = e2_sensitivity(problem)
    fig_map(problem, e1)
    fig_landscape(problem)
    fig_sensitivity(sens)
    log.info("E7 why not just sort?")
    e7 = e7_why_not_sort(problem)
    fig_why_not_sort(e7)
    print(e7[e7.w_outbreak == 0.3][["overlap_weight", "exact_optimum", "sorted_top_k", "sorting_is_wrong", "regret_of_sorting"]].round(4).to_string(index=False))
    if args.classical_only:
        return

    # QAOA task list ------------------------------------------------------------------------
    tasks = []
    for mixer, seed in product(("x", "xy"), cfg["depth_seeds"]):
        tasks.append(dict(experiment="E3_depth", model="C", mixer=mixer, depth=cfg["depth_max"], shots=cfg["base_shots"],
                          seed=seed, trace=(seed == 0)))
    for mixer, shots, seed in product(("x", "xy"), cfg["shots_list"], cfg["shots_seeds"]):
        tasks.append(dict(experiment="E4_shots", model="C", mixer=mixer, depth=cfg["shots_depth"], shots=shots, seed=seed,
                          last_only=True))
    for mult, seed in product(cfg["pen_mults"], cfg["pen_seeds"]):
        tasks.append(dict(experiment="E5_penalty", model="C", mixer="x", depth=cfg["pen_depth"], shots=cfg["base_shots"],
                          seed=seed, penalty_mult=mult, last_only=True))
    for model, seed in product("ABC", cfg["cmp_seeds"]):
        tasks.append(dict(experiment="E6_models", model=model, mixer="xy", depth=cfg["cmp_depth"], shots=cfg["base_shots"],
                          seed=seed, last_only=True))
    log.info("running %d QAOA chains on %d worker processes", len(tasks), args.workers)

    rows = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for out in pool.map(worker, tasks, chunksize=1):
            rows.extend(out)
            done += 1
            if done % 5 == 0 or done == len(tasks):
                log.info("  %d/%d chains finished (%.0fs elapsed)", done, len(tasks), time.perf_counter() - t_start)

    # separate traces from flat rows
    traces, trace_rows = {}, {}
    flat = []
    for r in rows:
        if "_energy_history" in r:
            traces[(r["mixer"], r["depth"])] = r.pop("_energy_history")
            if r["depth"] == cfg["depth_max"]:
                trace_rows[r["mixer"]] = dict(depth=r["depth"], _counts=r.pop("_counts"),
                                              _ideal_probabilities=r.pop("_ideal_probabilities"))
            else:
                r.pop("_counts"); r.pop("_ideal_probabilities")
        flat.append(r)
    runs = pd.DataFrame(flat)
    runs.to_csv(OUT / "runs_all.csv", index=False)

    # random controls (NOT quantum) with the same sample budget -----------------------------
    f_c = make_formulation("C")
    c_c = solve_exact(problem, f_c)
    ctrl = []
    for seed in cfg["depth_seeds"]:
        ctrl.append(random_baseline(problem, f_c, c_c, cfg["base_shots"], seed, feasible_only=False))
        ctrl.append(random_baseline(problem, f_c, c_c, cfg["base_shots"], seed, feasible_only=True))
    ctrl_df = pd.DataFrame(ctrl)
    ctrl_df.to_csv(OUT / "random_controls.csv", index=False)
    ctrl_summary = aggregate(ctrl, ["method"], ["feasibility_rate", "optimal_probability", "optimality_gap_range"]) \
        if len(ctrl) else None
    ctrl_summary.to_csv(OUT / "summary_random_controls.csv", index=False)

    # summaries + figures --------------------------------------------------------------------
    metrics = ["feasibility_rate", "optimal_probability", "ideal_optimal_probability", "ideal_feasibility",
               "optimality_gap_abs", "optimality_gap_range", "best_feasible_objective", "runtime_total_s",
               "evaluations_cumulative", "circuit_depth", "two_qubit_gates"]
    e3 = runs[runs.experiment == "E3_depth"]
    aggregate(e3.to_dict("records"), ["mixer", "depth"], metrics).to_csv(OUT / "summary_depth.csv", index=False)
    e4 = runs[runs.experiment == "E4_shots"]
    aggregate(e4.to_dict("records"), ["mixer", "shots"], metrics).to_csv(OUT / "summary_shots.csv", index=False)
    e5 = runs[runs.experiment == "E5_penalty"]
    aggregate(e5.to_dict("records"), ["penalty_over_threshold", "qubo_ground_state_feasible"], metrics) \
        .to_csv(OUT / "summary_penalty.csv", index=False)
    e6 = runs[runs.experiment == "E6_models"]
    agg6 = aggregate(e6.to_dict("records"), ["model"], metrics)
    agg6["exact_optimum"] = agg6.model.map(lambda m: " ".join(problem.names[i] for i in solve_exact(problem, make_formulation(m)).chosen_set))
    agg6["best_sampled_selection_most_common"] = agg6.model.map(
        lambda m: e6[e6.model == m].best_feasible_selection.astype(str).mode().iloc[0])
    agg6.to_csv(OUT / "summary_models_qaoa.csv", index=False)

    fig_depth(e3, ctrl_summary.to_dict("records"))
    fig_shots(e4)
    fig_penalty(e5)
    fig_convergence(traces)
    if {"x", "xy"} <= set(trace_rows):
        opt_index = sum(1 << i for i in c_c.chosen_set)
        fig_distribution(problem, trace_rows, opt_index, None)

    meta = {
        "quick": args.quick, "config": {k: (list(v) if isinstance(v, range) else v) for k, v in cfg.items()},
        "python": platform.python_version(), "platform": platform.platform(), "qiskit": qiskit.__version__,
        "qiskit_aer": qiskit_aer.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
        "pandas": pd.__version__, "matplotlib": matplotlib.__version__,
        "weights_models_A_B": {"patients": DEFAULT_WEIGHTS_AB.patients, "outbreak": DEFAULT_WEIGHTS_AB.outbreak,
                               "travel": DEFAULT_WEIGHTS_AB.travel},
        "overlap_weight_model_C": DEFAULT_OVERLAP_WEIGHT,
        "optimizer": "scipy COBYLA, INTERP warm starts", "backend": "qiskit_aer.AerSimulator(method='statevector')",
        "num_qaoa_chains": len(tasks), "num_result_rows": len(runs),
        "runtime_note": "runtime_total_s = transpile + optimisation loop + final sampling (+ shallower INTERP stages); "
                        "excludes exact-statevector diagnostics. Measured with workers running in parallel.",
        "wall_clock_seconds": round(time.perf_counter() - t_start, 1),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    log.info("done in %.0fs; results in %s", time.perf_counter() - t_start, OUT)


if __name__ == "__main__":
    main()
