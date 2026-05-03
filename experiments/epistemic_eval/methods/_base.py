"""Shared helpers for the per-method UQ wrappers.

This module centralises four concerns shared by every method wrapper
under ``experiments/epistemic_eval/methods/``:

1. :class:`FeatureProvider` -- the data abstraction that yields either
   ``(images, labels)`` (full-network mode) or ``(features, labels)``
   (linear-probe mode) so wrappers stay agnostic of which dataset
   strategy is in play.
2. :func:`derive_member_seed` -- per-ensemble-member seed derivation
   via ``blake2b``, namespaced so future methods (LLLA, etc.) can pick
   their own seed sub-streams without colliding.
3. :func:`hash_config` / :func:`make_run_id` -- canonical run-id and
   config-hash helpers used by the orchestrator and the cache layer.
4. :func:`setup_determinism` -- the single seeding entry point used at
   the top of every ``fit`` and ``extract``.

The :class:`MethodHandle` :class:`typing.Protocol` documents the
duck-typed contract every per-method ``Handle`` dataclass satisfies.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Iterator, Literal, Protocol, runtime_checkable

import numpy as np
import torch


def derive_member_seed(root_seed: int, member_idx: int) -> int:
    """Derive a per-ensemble-member seed via blake2b.

    Collision-free across "seed types" -- when LLLA samples its own
    sub-streams in Task 5b, the namespace prefix prevents overlap.
    Inspectable: the formula is explicit and has no arithmetic
    order dependency.

    Worked example (verified at module import time via assert):
        derive_member_seed(0, 0) == 226078449
        derive_member_seed(0, 1) == 1561542571
        derive_member_seed(1, 0) == 1605203515

    Args:
        root_seed: The run's root seed (from the experiment config).
        member_idx: The 0-based ensemble member index.

    Returns:
        A 31-bit positive integer suitable for ``torch.manual_seed``
        and ``numpy.random.default_rng``.
    """
    h = hashlib.blake2b(
        f"ensemble_member:{root_seed}:{member_idx}".encode(),
        digest_size=4,
    ).digest()
    return int.from_bytes(h, "big") & 0x7FFFFFFF  # 31-bit positive int


# Sanity check: keep the docstring example honest. Mutating the formula
# without updating the docstring or the test in
# `tests/test_seed_derivation.py` will trip these asserts.
assert derive_member_seed(0, 0) == 226078449  # noqa: S101
assert derive_member_seed(0, 1) == 1561542571  # noqa: S101
assert derive_member_seed(1, 0) == 1605203515  # noqa: S101


@runtime_checkable
class MethodHandle(Protocol):
    """Protocol every per-method handle satisfies.

    Each method module's ``Handle`` dataclass implements this
    informally -- duck-typed.

    Attributes:
        method_config: Resolved method config (a YAML-loaded dict).
        dataset_config: Resolved dataset config (a YAML-loaded dict).
        seed: The run's root seed.
    """

    method_config: dict[str, Any]
    dataset_config: dict[str, Any]
    seed: int


@dataclass
class FeatureProvider:
    """Yields ``(features_or_images, labels)`` per batch.

    The ``mode`` field tells consumers which path the data takes:

    * ``"full_network"``: yields ``(images, labels)``, where ``images``
      is a ``(B, C, H, W)`` tensor and ``labels`` is a
      ``(B, ...)`` tensor of per-image targets (soft p* for our
      datasets).
    * ``"linear_probe"``: yields ``(features, labels)``, where
      ``features`` is a ``(B, feature_dim)`` tensor of cached backbone
      features.

    This abstraction lets the method wrappers be agnostic to whether
    they're consuming images or features.

    The provider is also the only object that owns dataset traceability:
    ``indices`` returns the dataset row index for the i-th yielded
    sample, so extraction can write traceable per-row predictions.

    Attributes:
        batches: An iterable of ``(x, y)`` tensor pairs. ``x`` is
            shaped ``(B, ...)`` and ``y`` is shaped ``(B, ...)``.
        mode: ``"full_network"`` or ``"linear_probe"``.
        feature_dim: Feature dimensionality for ``linear_probe`` mode;
            ``None`` for ``full_network``.
        n_classes: Output label dimensionality (``K``).
        indices: Per-sample dataset indices for traceability. Length
            equals the total number of samples produced by iterating
            ``batches``.
    """

    batches: list[tuple[torch.Tensor, torch.Tensor]]
    mode: Literal["full_network", "linear_probe"]
    n_classes: int
    indices: np.ndarray
    feature_dim: int | None = None

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Iterate over ``(x, y)`` batches in order."""
        return iter(self.batches)

    def __len__(self) -> int:
        """Return the number of batches."""
        return len(self.batches)

    @property
    def n_samples(self) -> int:
        """Return the total number of samples across all batches."""
        return int(self.indices.shape[0])


def hash_config(config: dict[str, Any]) -> str:
    """Return a stable 16-hex-char hash of a YAML-loaded config dict.

    Uses a JSON canonical-form serialisation (sorted keys, no
    whitespace, ``default=str`` for path-likes) so that two semantically
    equal configs produce the same hash regardless of Python dict
    insertion order.

    Args:
        config: A YAML-loaded config dict.

    Returns:
        16-character lower-case hex hash.
    """
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.blake2b(canonical.encode("utf-8"), digest_size=8).hexdigest()


def make_run_id(method: str, dataset: str, seed: int, exp: str = "main") -> str:
    """Compose a run-id directory name.

    Format: ``<YYYYMMDD>_<exp>_<method>_<dataset>_seed<N>``.

    Args:
        method: Method name (e.g. ``"mc_dropout"``).
        dataset: Dataset name (e.g. ``"cifar10h"``).
        seed: Root seed integer.
        exp: Free-form experiment tag (default ``"main"``).

    Returns:
        The composed run-id string.
    """
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"{today}_{exp}_{method}_{dataset}_seed{seed}"


def setup_determinism(seed: int) -> None:
    """Set the global RNGs for the current process.

    Sets ``torch.manual_seed``, ``numpy.random.default_rng`` (used by
    callers via assignment), and toggles
    ``torch.use_deterministic_algorithms(True, warn_only=True)``.

    Args:
        seed: Root seed for this process.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.default_rng(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


__all__ = [
    "FeatureProvider",
    "MethodHandle",
    "derive_member_seed",
    "hash_config",
    "make_run_id",
    "setup_determinism",
]
