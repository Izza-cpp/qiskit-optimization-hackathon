import itertools

import numpy as np
import pytest

from src.constraints import (bitstring_to_selection, cardinality_penalty, cardinality_violation, count_selected,
                             expanded_penalty_coefficients, is_feasible, selection_to_bitstring,
                             selection_to_locations, validate_binary)


@pytest.mark.parametrize("ones,feasible", [(0, False), (1, False), (2, False), (3, True), (4, False), (10, False)])
def test_feasibility_by_count(ones, feasible):
    x = [1] * ones + [0] * (10 - ones)
    assert count_selected(x) == ones
    assert is_feasible(x, 3) is feasible


@pytest.mark.parametrize("ones,expected", [(0, 9), (2, 1), (3, 0), (4, 1), (5, 4), (10, 49)])
def test_penalty_values(ones, expected):
    x = [1] * ones + [0] * (10 - ones)
    assert cardinality_penalty(x, 3) == expected
    assert cardinality_penalty(x, 3, strength=2.5) == 2.5 * expected
    assert cardinality_violation(x, 3) == ones - 3


def test_expansion_equals_square_for_all_1024_states():
    const, lin, pair = expanded_penalty_coefficients(10, 3, strength=1.0)
    assert (const, lin, pair) == (9, -5, 2)
    for x in itertools.product([0, 1], repeat=10):
        xs = np.array(x)
        expanded = const + lin * xs.sum() + pair * sum(xs[i] * xs[j] for i in range(10) for j in range(i + 1, 10))
        assert expanded == (xs.sum() - 3) ** 2


def test_invalid_selection_rejected():
    for bad in ([2, 0, 1], [0.5, 1, 1], [[1, 0], [0, 1]], [-1, 1, 1]):
        with pytest.raises(ValueError):
            validate_binary(bad)
    with pytest.raises(ValueError):
        validate_binary([1, 0, 1], n=10)
    with pytest.raises(ValueError):
        cardinality_penalty([1, 1, 1], 3, strength=-1)


def test_bitstring_is_little_endian():
    # Qiskit string "0000000001": rightmost char is qubit 0 -> location L1
    assert bitstring_to_selection("0000000001", 10) == (1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    assert bitstring_to_selection("1000000000", 10) == (0, 0, 0, 0, 0, 0, 0, 0, 0, 1)
    assert bitstring_to_selection("0000010101", 10)[:5] == (1, 0, 1, 0, 1)   # L1, L3, L5


def test_bitstring_roundtrip_and_errors():
    for x in itertools.islice(itertools.product([0, 1], repeat=10), 0, 1024, 37):
        assert bitstring_to_selection(selection_to_bitstring(x), 10) == x
    with pytest.raises(ValueError):
        bitstring_to_selection("101", 10)
    with pytest.raises(ValueError):
        bitstring_to_selection("00000000a1", 10)
    assert bitstring_to_selection("0000 000001", 10)[0] == 1   # register separator tolerated


def test_locations_from_selection():
    names = [f"L{i}" for i in range(1, 11)]
    assert selection_to_locations([1, 0, 1, 0, 1, 0, 0, 0, 0, 0], names) == ["L1", "L3", "L5"]
