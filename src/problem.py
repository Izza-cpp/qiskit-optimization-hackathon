"""Problem instance: ten disaster-affected locations, three medical teams.

All input data live HERE and nowhere else. Other modules import
``default_problem()`` instead of re-typing numbers.

DATA PROVENANCE: the default dataset is SYNTHETIC. It is not an observation of
any real disaster. It was designed (see README, "Dataset") so that the three
formulations compared in this project select different locations; otherwise the
comparison would show nothing.

MODEL ASSUMPTIONS (these are modelling choices, not facts about the world):
  1. The three teams are interchangeable deployment units.
  2. All teams start at ONE central depot; travel time is depot -> location.
  3. At most one team per location (x_i is binary).
  4. No routing, no scheduling, no traffic model, no team-specific skills.
  5. Location coordinates (km) are synthetic. Travel time = distance / speed,
     so coordinates and travel times are consistent by construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

NUM_LOCATIONS = 10
NUM_TEAMS = 3
RISK_SCALE = 100.0           # outbreak risk is a 0-100 score
TRAVEL_SPEED_KM_PER_MIN = 0.5  # 30 km/h effective speed in disaster conditions
OVERLAP_REFERENCE_KM = 15.0  # service-area overlap vanishes beyond this distance


@dataclass(frozen=True)
class Weights:
    """Objective weights. ``patients + outbreak + travel`` must equal 1.

    ``overlap`` is the weight of the OPTIONAL pairwise service-overlap penalty
    (model C). It is NOT part of the simplex constraint: it is a separate
    penalty coefficient on a normalised [0, 1] quantity.
    """

    patients: float = 0.5
    outbreak: float = 0.3
    travel: float = 0.2
    overlap: float = 0.0

    def __post_init__(self) -> None:
        for name in ("patients", "outbreak", "travel", "overlap"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"weight '{name}' must be finite and >= 0, got {value}")
        total = self.patients + self.outbreak + self.travel
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"patients + outbreak + travel must equal 1, got {total}")

    def with_(self, **kwargs: float) -> "Weights":
        """Return a copy with some weights replaced (validated again)."""
        values = {"patients": self.patients, "outbreak": self.outbreak,
                  "travel": self.travel, "overlap": self.overlap}
        values.update(kwargs)
        return Weights(**values)


@dataclass(frozen=True)
class MedicalDeploymentProblem:
    """A small, validated deployment instance."""

    names: Tuple[str, ...]
    patients: np.ndarray          # estimated patients needing care (people)
    travel_minutes: np.ndarray    # depot -> location (minutes)
    outbreak_risk: np.ndarray     # 0-100 score (synthetic)
    coordinates_km: np.ndarray    # shape (n, 2), depot at (0, 0)
    num_teams: int = NUM_TEAMS
    description: str = ""
    extras: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.names)
        for label in ("patients", "travel_minutes", "outbreak_risk"):
            arr = np.asarray(getattr(self, label), dtype=float)
            if arr.shape != (n,):
                raise ValueError(f"{label} must have exactly {n} entries, got shape {arr.shape}")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{label} contains non-finite values")
            if np.any(arr < 0):
                raise ValueError(f"{label} must be non-negative")
            object.__setattr__(self, label, arr)
        coords = np.asarray(self.coordinates_km, dtype=float)
        if coords.shape != (n, 2):
            raise ValueError(f"coordinates_km must have shape ({n}, 2), got {coords.shape}")
        object.__setattr__(self, "coordinates_km", coords)
        if np.max(self.patients) <= 0 or np.max(self.travel_minutes) <= 0:
            raise ValueError("patients and travel_minutes need a positive maximum (used for normalisation)")
        if np.any(self.outbreak_risk > RISK_SCALE):
            raise ValueError(f"outbreak_risk must be <= {RISK_SCALE}")
        if len(set(self.names)) != n:
            raise ValueError("location names must be unique")
        if not (0 < self.num_teams <= n):
            raise ValueError("num_teams must satisfy 0 < num_teams <= number of locations")

    @property
    def n(self) -> int:
        return len(self.names)

    def overlap_matrix(self) -> np.ndarray:
        """Pairwise service-area overlap O_ij in [0, 1], zero diagonal.

        O_ij = max(0, 1 - d_ij / OVERLAP_REFERENCE_KM): two locations at the same
        point overlap fully (1); locations >= 15 km apart do not overlap (0).
        """
        diff = self.coordinates_km[:, None, :] - self.coordinates_km[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        overlap = np.clip(1.0 - dist / OVERLAP_REFERENCE_KM, 0.0, None)
        np.fill_diagonal(overlap, 0.0)
        return overlap

    def table(self):
        """Dataset as a pandas DataFrame (for reports / notebook)."""
        import pandas as pd
        return pd.DataFrame({
            "location": self.names, "qubit": range(self.n),
            "patients": self.patients, "travel_min": self.travel_minutes,
            "outbreak_risk": self.outbreak_risk,
            "x_km": self.coordinates_km[:, 0].round(2), "y_km": self.coordinates_km[:, 1].round(2),
        })


# angles (degrees, from the depot) chosen so that the best linear locations
# L1, L3 and L5 sit close together => pairwise overlap matters in model C.
_DEFAULT_ANGLES_DEG = (10.0, 95.0, 25.0, 170.0, 40.0, 250.0, 300.0, 200.0, 60.0, 330.0)


def default_problem() -> MedicalDeploymentProblem:
    """The SYNTHETIC ten-location instance used throughout the project."""
    names = tuple(f"L{i}" for i in range(1, NUM_LOCATIONS + 1))
    patients = np.array([80, 55, 110, 35, 70, 45, 100, 30, 65, 50], dtype=float)
    travel = np.array([20, 35, 50, 15, 25, 40, 60, 10, 30, 45], dtype=float)
    risk = np.array([90, 40, 85, 30, 75, 95, 30, 20, 80, 65], dtype=float)
    radius = TRAVEL_SPEED_KM_PER_MIN * travel
    theta = np.radians(_DEFAULT_ANGLES_DEG)
    coords = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    return MedicalDeploymentProblem(
        names=names, patients=patients, travel_minutes=travel, outbreak_risk=risk,
        coordinates_km=coords, num_teams=NUM_TEAMS,
        description="SYNTHETIC instance: 10 locations, 3 interchangeable teams, 1 depot.",
    )
