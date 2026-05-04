#!/usr/bin/env python3
"""Build the per-seed ``p_star.npz`` sidecar for a DCIC dataset.

Reads a DCIC dataset config (``configs/datasets/<name>.yaml``) and a
seed, instantiates the loader, derives the test fold via
:func:`experiments.epistemic_eval.datasets.dcic.test_fold_for_seed`,
filters ``targets`` to the test-fold images, and emits

    data/<DatasetName>/p_star_seed<N>.npz

with::

    p_star:  (N_test, K) float32  -- row-stochastic conditional p*(y | x).
    support: (K,)        int64    -- arange(K).
    indices: (N_test,)   int64    -- per-row dataset index in the parent
                                     loader (i.e. the index returned by
                                     :class:`DCICDataset` so downstream
                                     scripts can align predictions and p*
                                     by row).

p* construction per ``decisions.md`` -> "p*(y | x) construction":
the per-image vote distribution is exactly what ``DCICDataset.targets[i]``
already exposes (row-normalised vote histogram), so this script just
filters and saves; no math.

Class ordering: the sidecar's column index ``k`` corresponds to the
``k``-th label in lexicographic order of the dataset's label set.
The wrapper class (:class:`experiments.epistemic_eval.datasets.dcic._DeterministicDCIC`)
re-keys ``label_mappings`` deterministically so the same column
ordering is reconstructed in train / extract / oracle, even across
processes with different ``PYTHONHASHSEED``.

Idempotent by default: if ``p_star_seed<N>.npz`` already exists with
the correct shape AND its values match what we would produce now to
``atol=1e-12``, the script prints ``skipped: <path> exists`` and
exits 0. Pass ``--force`` to overwrite unconditionally.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import yaml

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.epistemic_eval.datasets.dcic import (  # noqa: E402
    _build_loader,
    _data_root_from_config,
    _dataset_name_from_config,
    split_test_train_indices,
)


def build_p_star(
    dataset_config: dict,
    seed: int,
) -> dict[str, np.ndarray]:
    """Compute ``p_star`` over the seed-derived test fold.

    Args:
        dataset_config: Resolved DCIC dataset config (the experiment
            config block under ``dataset:`` or a standalone YAML).
        seed: Run seed; the test fold is derived per
            :func:`experiments.epistemic_eval.datasets.dcic.test_fold_for_seed`.

    Returns:
        Dict with keys ``p_star`` ``(N_test, K) float32``, ``support``
        ``(K,) int64``, ``indices`` ``(N_test,) int64``.

    Raises:
        ValueError: If the loaded ``targets`` do not sum to 1 within
            tolerance, or if the test fold is empty.
    """
    data_root = _data_root_from_config(dataset_config)
    dataset_name = _dataset_name_from_config(dataset_config)
    loader = _build_loader(dataset_name, data_root, transform=None)
    _fold, _train_indices, test_indices = split_test_train_indices(loader, seed)
    p_rows = [loader.targets[i].detach().cpu().numpy() for i in test_indices]
    p_star = np.stack(p_rows, axis=0).astype(np.float32, copy=False)
    n, k = p_star.shape
    if n == 0:
        msg = (
            f"empty test fold for {dataset_name!r} at seed {seed}; "
            f"check the on-disk fold structure."
        )
        raise ValueError(msg)
    row_sums = p_star.sum(axis=1, dtype=np.float64)
    drift = float(np.max(np.abs(row_sums - 1.0)))
    if drift > 1.0e-5:
        msg = (
            f"p_star rows deviate from 1.0 by {drift:.3e} (atol=1e-5); "
            f"underlying DCIC ``targets`` should be row-stochastic by "
            f"construction. Check the loader."
        )
        raise ValueError(msg)
    return {
        "p_star": p_star,
        "support": np.arange(k, dtype=np.int64),
        "indices": np.asarray(test_indices, dtype=np.int64),
    }


def _matches_existing(output_path: Path, candidate: dict[str, np.ndarray]) -> bool:
    """Return True iff ``output_path`` exists and its arrays match ``candidate``."""
    if not output_path.is_file():
        return False
    try:
        cached = np.load(output_path, allow_pickle=False)
    except (OSError, ValueError):
        return False
    required = {"p_star", "support", "indices"}
    if not required.issubset(cached.files):
        return False
    if cached["p_star"].shape != candidate["p_star"].shape:
        return False
    if not np.array_equal(cached["support"], candidate["support"]):
        return False
    if not np.array_equal(cached["indices"], candidate["indices"]):
        return False
    return bool(
        np.allclose(cached["p_star"], candidate["p_star"], atol=1e-12, rtol=0.0)
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point. See module docstring."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-config",
        type=Path,
        required=True,
        help="Path to a DCIC dataset config (configs/datasets/<name>.yaml).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Run seed; the test fold is derived per dcic.test_fold_for_seed.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Override; defaults to "
            "<loader_kwargs.root>/<dcic_name>/p_star_seed<N>.npz."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the sidecar unconditionally (skip the idempotence check).",
    )
    args = parser.parse_args(argv)

    config_path: Path = args.dataset_config
    if not config_path.is_file():
        print(f"dataset config not found at {config_path}", file=sys.stderr)
        return 1
    dataset_config = yaml.safe_load(config_path.read_text())

    candidate = build_p_star(dataset_config, int(args.seed))

    if args.output_path is not None:
        output_path = args.output_path
    else:
        data_root = _data_root_from_config(dataset_config)
        dataset_name = _dataset_name_from_config(dataset_config)
        output_path = data_root / dataset_name / f"p_star_seed{int(args.seed)}.npz"

    if not args.force and _matches_existing(output_path, candidate):
        print(f"skipped: {output_path} exists")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        p_star=candidate["p_star"],
        support=candidate["support"],
        indices=candidate["indices"],
    )
    n, k = candidate["p_star"].shape
    sidecar = output_path.with_suffix(".meta.json")
    sidecar.write_text(
        json.dumps(
            {
                "dataset_name": _dataset_name_from_config(dataset_config),
                "seed": int(args.seed),
                "n_test_points": int(n),
                "num_classes": int(k),
            },
            indent=2,
        )
    )
    print(f"wrote {output_path} (n={n}, k={k}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
