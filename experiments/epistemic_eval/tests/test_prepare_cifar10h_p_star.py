"""Smoke tests for ``prepare_cifar10h_p_star.py``.

Synthetic 3-image x 4-class fixture (does NOT touch the real
CIFAR-10H counts file) verifies the schema, normalisation, and
idempotence behaviour of the prep script.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments.epistemic_eval.scripts import prepare_cifar10h_p_star


@pytest.fixture
def synthetic_counts(tmp_path: Path) -> Path:
    """Write a synthetic ``(3, 4) int64`` counts file and return its path.

    Row sums chosen to be different (4, 8, 16) so a bug that reuses
    a single normalising constant would surface as non-row-stochastic
    output.
    """
    counts = np.array(
        [
            [1, 1, 1, 1],     # uniform
            [4, 2, 1, 1],     # peaked on class 0
            [0, 0, 0, 16],    # one-hot on class 3
        ],
        dtype=np.int64,
    )
    path = tmp_path / "counts.npy"
    np.save(path, counts)
    return path


def test_writes_p_star_with_expected_schema(tmp_path: Path, synthetic_counts: Path) -> None:
    output_path = tmp_path / "p_star.npz"
    rc = prepare_cifar10h_p_star.main(
        [
            "--counts-path",
            str(synthetic_counts),
            "--output-path",
            str(output_path),
        ]
    )
    assert rc == 0
    assert output_path.is_file()

    blob = np.load(output_path, allow_pickle=False)
    assert set(blob.files) == {"p_star", "support", "indices"}

    p_star = blob["p_star"]
    support = blob["support"]
    indices = blob["indices"]

    assert p_star.shape == (3, 4)
    assert p_star.dtype == np.float32
    assert support.shape == (4,)
    assert support.dtype == np.int64
    assert indices.shape == (3,)
    assert indices.dtype == np.int64
    np.testing.assert_array_equal(support, np.arange(4, dtype=np.int64))
    np.testing.assert_array_equal(indices, np.arange(3, dtype=np.int64))


def test_p_star_rows_sum_to_one(tmp_path: Path, synthetic_counts: Path) -> None:
    output_path = tmp_path / "p_star.npz"
    prepare_cifar10h_p_star.main(
        [
            "--counts-path",
            str(synthetic_counts),
            "--output-path",
            str(output_path),
        ]
    )
    p_star = np.load(output_path, allow_pickle=False)["p_star"]
    np.testing.assert_allclose(
        p_star.sum(axis=1, dtype=np.float64),
        np.ones(3, dtype=np.float64),
        atol=1e-6,
    )


def test_p_star_values_match_normalised_counts(tmp_path: Path, synthetic_counts: Path) -> None:
    """Verify p_star[i, k] == counts[i, k] / counts[i, :].sum()."""
    output_path = tmp_path / "p_star.npz"
    prepare_cifar10h_p_star.main(
        [
            "--counts-path",
            str(synthetic_counts),
            "--output-path",
            str(output_path),
        ]
    )
    p_star = np.load(output_path, allow_pickle=False)["p_star"]
    expected = np.array(
        [
            [0.25, 0.25, 0.25, 0.25],
            [4 / 8, 2 / 8, 1 / 8, 1 / 8],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(p_star, expected, atol=1e-6)


def test_idempotent_skip_when_file_exists(
    tmp_path: Path, synthetic_counts: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_path = tmp_path / "p_star.npz"
    prepare_cifar10h_p_star.main(
        [
            "--counts-path",
            str(synthetic_counts),
            "--output-path",
            str(output_path),
        ]
    )
    capsys.readouterr()  # discard first-run "wrote ..." stdout
    mtime_before = output_path.stat().st_mtime_ns

    rc = prepare_cifar10h_p_star.main(
        [
            "--counts-path",
            str(synthetic_counts),
            "--output-path",
            str(output_path),
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "skipped" in captured.out
    # File is not rewritten on the idempotent path.
    assert output_path.stat().st_mtime_ns == mtime_before


def test_force_overwrites_even_if_match(
    tmp_path: Path, synthetic_counts: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_path = tmp_path / "p_star.npz"
    prepare_cifar10h_p_star.main(
        [
            "--counts-path",
            str(synthetic_counts),
            "--output-path",
            str(output_path),
        ]
    )
    capsys.readouterr()

    rc = prepare_cifar10h_p_star.main(
        [
            "--counts-path",
            str(synthetic_counts),
            "--output-path",
            str(output_path),
            "--force",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "skipped" not in captured.out
    assert "wrote" in captured.out


def test_missing_counts_file_returns_nonzero(tmp_path: Path) -> None:
    output_path = tmp_path / "p_star.npz"
    rc = prepare_cifar10h_p_star.main(
        [
            "--counts-path",
            str(tmp_path / "does_not_exist.npy"),
            "--output-path",
            str(output_path),
        ]
    )
    assert rc == 1
    assert not output_path.exists()


def test_zero_row_sum_raises(tmp_path: Path) -> None:
    """A row of zero counts cannot be normalised; build_p_star must raise."""
    counts = np.array(
        [
            [1, 1, 1, 1],
            [0, 0, 0, 0],
        ],
        dtype=np.int64,
    )
    with pytest.raises(ValueError, match="zero-sum row"):
        prepare_cifar10h_p_star.build_p_star(counts)


def test_negative_count_raises(tmp_path: Path) -> None:
    counts = np.array(
        [
            [1, 1, 1, 1],
            [-1, 2, 0, 0],
        ],
        dtype=np.int64,
    )
    with pytest.raises(ValueError, match="non-negative"):
        prepare_cifar10h_p_star.build_p_star(counts)


def test_one_d_counts_raises() -> None:
    counts = np.array([1, 2, 3, 4], dtype=np.int64)
    with pytest.raises(ValueError, match="2-D"):
        prepare_cifar10h_p_star.build_p_star(counts)
