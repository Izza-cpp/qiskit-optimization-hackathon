"""Synthetic instance generator for the SCALING study (not used by the main n=10 experiments).

Instances are random but seeded, so every size is reproducible. They are SYNTHETIC, like the
default dataset: locations are scattered in a disc around the depot, travel time = distance/speed.
"""
from __future__ import annotations

import numpy as np

from .problem import MedicalDeploymentProblem, TRAVEL_SPEED_KM_PER_MIN


def random_problem(n: int, num_teams: int, seed: int, radius_km: float = 30.0) -> MedicalDeploymentProblem:
    """Random synthetic instance with ``n`` locations (= n qubits) and ``num_teams`` teams."""
    if not (0 < num_teams < n):
        raise ValueError("need 0 < num_teams < n")
    rng = np.random.default_rng(seed)
    r = radius_km * np.sqrt(rng.uniform(0.05, 1.0, n))
    a = rng.uniform(0.0, 2.0 * np.pi, n)
    coords = np.column_stack([r * np.cos(a), r * np.sin(a)])
    travel = np.linalg.norm(coords, axis=1) / TRAVEL_SPEED_KM_PER_MIN
    return MedicalDeploymentProblem(
        names=tuple(f"L{i}" for i in range(1, n + 1)),
        patients=np.round(rng.uniform(30, 120, n)),
        travel_minutes=np.round(travel, 1),
        outbreak_risk=np.round(rng.uniform(20, 95, n)),
        coordinates_km=np.round(coords, 2),
        num_teams=num_teams,
        description=f"synthetic random instance n={n}, K={num_teams}, seed={seed}",
    )
