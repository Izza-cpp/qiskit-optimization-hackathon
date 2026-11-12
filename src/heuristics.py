"""Classical heuristics used as additional baselines (greedy, local search, simulated annealing).

They see only the objective F(x); they use no quantum code and no exact solution. Each returns the
best selection found and the number of objective evaluations used, so they can be compared with
QAOA and random sampling at a given evaluation budget.
"""
from __future__ import annotations

import math
from typing import Callable, Tuple

import numpy as np

Selection = Tuple[int, ...]


def _value(fn: Callable[[Selection], float], sel: Selection, counter: list) -> float:
    counter[0] += 1
    return fn(tuple(sorted(sel)))


def greedy(n: int, k: int, fn: Callable[[Selection], float]) -> Tuple[Selection, int]:
    """Add, K times, the location with the largest marginal objective gain (deterministic)."""
    used = [0]
    chosen: list = []
    for _ in range(k):
        best, best_val = None, -math.inf
        for i in range(n):
            if i in chosen:
                continue
            # marginal gain of adding i to a partial set: evaluate the partial set (size < K ok for fn)
            val = _value(fn, tuple(chosen + [i]), used)
            if val > best_val:
                best, best_val = i, val
        chosen.append(best)
    return tuple(sorted(chosen)), used[0]


def random_search(n: int, k: int, fn, rng: np.random.Generator, budget: int) -> Tuple[Selection, int]:
    used = [0]
    best, best_val = None, -math.inf
    for _ in range(budget):
        sel = tuple(sorted(rng.choice(n, size=k, replace=False).tolist()))
        v = _value(fn, sel, used)
        if v > best_val:
            best, best_val = sel, v
    return best, used[0]


def local_search(n: int, k: int, fn, rng: np.random.Generator, budget: int) -> Tuple[Selection, int]:
    """Random-restart first-improvement 1-swap hill climbing until the budget is spent."""
    used = [0]
    best, best_val = None, -math.inf
    while used[0] < budget:
        cur = tuple(sorted(rng.choice(n, size=k, replace=False).tolist()))
        cur_val = _value(fn, cur, used)
        improved = True
        while improved and used[0] < budget:
            improved = False
            outs = rng.permutation(list(cur))
            ins = rng.permutation([i for i in range(n) if i not in cur])
            for o in outs:
                for i in ins:
                    if used[0] >= budget:
                        break
                    cand = tuple(sorted([j for j in cur if j != o] + [int(i)]))
                    v = _value(fn, cand, used)
                    if v > cur_val + 1e-12:
                        cur, cur_val, improved = cand, v, True
                        break
                if improved:
                    break
        if cur_val > best_val:
            best, best_val = cur, cur_val
    return best, used[0]


def simulated_annealing(n: int, k: int, fn, rng: np.random.Generator, budget: int,
                        t_start: float = 0.2, t_end: float = 0.005) -> Tuple[Selection, int]:
    """Simulated annealing with 1-swap moves and a geometric cooling schedule."""
    used = [0]
    cur = tuple(sorted(rng.choice(n, size=k, replace=False).tolist()))
    cur_val = _value(fn, cur, used)
    best, best_val = cur, cur_val
    steps = max(1, budget - 1)
    for step in range(steps):
        temp = t_start * (t_end / t_start) ** (step / steps)
        o = int(rng.choice(list(cur)))
        i = int(rng.choice([j for j in range(n) if j not in cur]))
        cand = tuple(sorted([j for j in cur if j != o] + [i]))
        v = _value(fn, cand, used)
        if v >= cur_val or rng.random() < math.exp((v - cur_val) / temp):
            cur, cur_val = cand, v
            if v > best_val:
                best, best_val = cand, v
    return best, used[0]
