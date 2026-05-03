"""Tests for :func:`experiments.epistemic_eval.methods._base.derive_member_seed`.

If :func:`derive_member_seed` implementation changes, update both this
test and the docstring example in ``methods/_base.py``.
"""

from __future__ import annotations

from experiments.epistemic_eval.methods._base import derive_member_seed


def test_derive_member_seed_known_values() -> None:
    """Hard-coded values verified by the reference computation."""
    assert derive_member_seed(0, 0) == 226078449
    assert derive_member_seed(0, 1) == 1561542571
    assert derive_member_seed(1, 0) == 1605203515


def test_derive_member_seed_31_bit_positive() -> None:
    """Output must fit in a non-negative 31-bit signed integer."""
    for root in [0, 1, 7, 99, 12345]:
        for member in range(8):
            value = derive_member_seed(root, member)
            assert value >= 0
            assert value < (1 << 31)


def test_derive_member_seed_distinct_per_member() -> None:
    """Members at the same root seed must hash to distinct integers."""
    seen = {derive_member_seed(0, m) for m in range(16)}
    assert len(seen) == 16


def test_derive_member_seed_distinct_per_root() -> None:
    """Different roots at the same member must hash to distinct integers."""
    seen = {derive_member_seed(r, 0) for r in range(16)}
    assert len(seen) == 16
