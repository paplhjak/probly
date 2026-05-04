"""Test fixtures local to ``experiments/epistemic_eval/tests``.

Adds the experiment's ``scripts/`` directory to ``sys.path`` so that
test modules can import the script files as bare names (e.g.
``import imagenet_real_adapter``) without polluting probly's import
space.

Also exposes :func:`make_synthetic_dcic_fixture`, used by the DCIC
adapter / prep-script tests to build a tiny on-disk DCIC dataset
without needing the multi-GB Zenodo release.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))


def make_synthetic_dcic_fixture(
    base_dir: Path,
    *,
    dataset_name: str = "Plankton",
    num_classes: int = 4,
    images_per_fold: int = 3,
    num_folds: int = 5,
    image_size: int = 32,
    annotations_per_image: int = 5,
    seed: int = 0,
    fold_prefix: str = "part",
) -> Path:
    """Create a synthetic DCIC-format dataset on disk.

    Builds ``base_dir/<dataset_name>/annotations.json`` plus tiny
    random PNG images at ``base_dir/<dataset_name>/<fold>/<filename>``
    so a :class:`probly.datasets.torch.DCICDataset` instance can load
    them. Used by the DCIC test suite to exercise the adapter's
    fold-filtering, p* construction, and provider builders without
    needing the real Zenodo-hosted DCIC archives.

    The default ``fold_prefix="part"`` matches the actual on-disk
    naming of the DCIC release (per Oleg). Tests that want to verify
    the adapter is fold-prefix-agnostic can pass ``fold_prefix="fold"``.

    Args:
        base_dir: Parent directory the DCIC loader will receive as
            ``root`` (the loader appends ``<dataset_name>/`` itself).
        dataset_name: Canonical DCIC dataset name (must match a key
            in :data:`experiments.epistemic_eval.datasets.dcic.DCIC_LOADERS`).
        num_classes: Number of classes in the synthetic dataset.
        images_per_fold: Number of images per fold.
        num_folds: Number of folds.
        image_size: Side length of the random RGB images written to
            disk; small to keep tests fast.
        annotations_per_image: Number of votes per image (the
            soft-label vector is the row-normalised vote histogram).
        seed: Seed for the per-image label sampling so the fixture is
            deterministic.
        fold_prefix: Prefix for fold directory names; defaults to
            ``"part"`` (matching the DCIC release).

    Returns:
        ``base_dir`` (the loader's ``root`` arg).
    """
    rng = np.random.default_rng(int(seed))
    dataset_root = Path(base_dir) / dataset_name
    dataset_root.mkdir(parents=True, exist_ok=True)
    annotations: list[dict[str, Any]] = []
    for fold_idx in range(int(num_folds)):
        fold = f"{fold_prefix}{fold_idx + 1}"
        (dataset_root / fold).mkdir(exist_ok=True)
        for image_idx in range(int(images_per_fold)):
            filename = f"img_{fold_idx:02d}_{image_idx:02d}.png"
            relative_path = f"{dataset_name}/{fold}/{filename}"
            absolute_path = Path(base_dir) / relative_path
            arr = (rng.random((int(image_size), int(image_size), 3)) * 255).astype(
                np.uint8
            )
            Image.fromarray(arr).save(absolute_path)
            entry: dict[str, Any] = {
                "annotations": [
                    {
                        "image_path": relative_path,
                        "class_label": int(rng.integers(0, int(num_classes))),
                    }
                    for _ in range(int(annotations_per_image))
                ]
            }
            annotations.append(entry)
    (dataset_root / "annotations.json").write_text(json.dumps(annotations))
    return Path(base_dir)
