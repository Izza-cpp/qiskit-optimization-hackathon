"""Cardinality constraint (exactly K locations selected) and bitstring decoding.

BIT ORDERING (the classic Qiskit trap)
--------------------------------------
Qiskit prints measurement strings little-endian: the RIGHTMOST character is
qubit 0. We define:   qubit i  <->  decision variable x_i  <->  location L(i+1).
So for the 10-bit string ``b`` returned by Qiskit:  x_i = int(b[n - 1 - i]).
``bitstring_to_selection`` does this and is unit-tested.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def validate_binary(x: Sequence[int], n: int | None = None) -> np.ndarray:
    """Return x as an int array, raising ValueError unless it is a 0/1 vector."""
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if n is not None and arr.shape[0] != n:
        raise ValueError(f"x must have length {n}, got {arr.shape[0]}")
    if not np.all((arr == 0) | (arr == 1)):
        raise ValueError("x must contain only 0 and 1")
    return arr.astype(int)


def count_selected(x: Sequence[int]) -> int:
    """Number of selected locations (Hamming weight)."""
    return int(validate_binary(x).sum())


def is_feasible(x: Sequence[int], num_teams: int) -> bool:
    """True iff exactly ``num_teams`` locations are selected."""
    return count_selected(x) == num_teams


def cardinality_violation(x: Sequence[int], num_teams: int) -> int:
    """Signed violation: (selected - num_teams). Zero means feasible."""
    return count_selected(x) - num_teams


def cardinality_penalty(x: Sequence[int], num_teams: int, strength: float = 1.0) -> float:
    """lambda * (sum_i x_i - K)^2  -- the quadratic penalty used in the QUBO."""
    if strength < 0:
        raise ValueError("penalty strength must be >= 0")
    return float(strength * cardinality_violation(x, num_teams) ** 2)


def expanded_penalty_coefficients(n: int, num_teams: int, strength: float = 1.0):
    """Coefficients of the binary expansion of lambda*(sum x - K)^2.

    Using x_i^2 = x_i:
        lambda*(sum x - K)^2 = lambda*K^2
                               + lambda*(1 - 2K) * sum_i x_i
                               + 2*lambda * sum_{i<j} x_i x_j
    For K = 3 this is lambda*(9 - 5 sum x_i + 2 sum_{i<j} x_i x_j).
    Returns (constant, linear_per_variable, pair_coefficient).
    """
    lam, k = strength, num_teams
    return lam * k * k, lam * (1 - 2 * k), 2 * lam


def bitstring_to_selection(bitstring: str, n: int) -> Tuple[int, ...]:
    """Decode a Qiskit measurement string into x = (x_0, ..., x_{n-1}).

    Qiskit is little-endian, so x_i = bitstring[n-1-i].
    Spaces (from multiple classical registers) are removed first.
    """
    b = bitstring.replace(" ", "")
    if len(b) != n or any(c not in "01" for c in b):
        raise ValueError(f"expected a {n}-bit string of 0/1, got {bitstring!r}")
    return tuple(int(b[n - 1 - i]) for i in range(n))


def selection_to_bitstring(x: Sequence[int]) -> str:
    """Inverse of :func:`bitstring_to_selection` (little-endian Qiskit string)."""
    arr = validate_binary(x)
    return "".join(str(int(v)) for v in arr[::-1])


def selection_to_locations(x: Sequence[int], names: Sequence[str]) -> List[str]:
    """Names of the selected locations."""
    arr = validate_binary(x, len(names))
    return [names[i] for i in np.flatnonzero(arr)]
