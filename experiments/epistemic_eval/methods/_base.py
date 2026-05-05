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
from typing import Any, Callable, Iterator, Literal, Protocol, runtime_checkable

import numpy as np
import torch
from torch import nn


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


def build_optimizer(
    model: nn.Module,
    *,
    optimizer_name: str,
    lr: float,
    momentum: float,
    weight_decay: float,
    nesterov: bool,
) -> torch.optim.Optimizer:
    """Construct the locked SGD/AdamW optimizer over ``model``'s trainable params.

    Centralises the optimizer dispatch shared by every from-scratch UQ
    training path (ensemble per-member, evidential, ddu) and by the
    CIFAR-10H basecls trainer. ``"sgd"`` is the default across UQ
    methods; ``"adamw"`` is opt-in via the dataset config's
    ``training.method_overrides.optimizer`` for the linear-probe
    (APPA-REAL) regime where SGD-cosine underfits a small head over
    cached features.

    Args:
        model: The module whose ``requires_grad`` parameters become the
            optimizer's param group.
        optimizer_name: ``"sgd"`` or ``"adamw"``.
        lr: Initial learning rate.
        momentum: SGD momentum (ignored by AdamW).
        weight_decay: L2 / decoupled weight decay.
        nesterov: SGD Nesterov flag (ignored by AdamW).

    Returns:
        The constructed :class:`torch.optim.Optimizer`.

    Raises:
        ValueError: If ``optimizer_name`` is neither ``"sgd"`` nor
            ``"adamw"``.
    """
    name = str(optimizer_name).lower()
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    msg = f"unknown optimizer {optimizer_name!r}; expected 'sgd' (default) or 'adamw'."
    raise ValueError(msg)


def train_with_val_tracking(
    model: nn.Module,
    train_provider: FeatureProvider,
    val_provider: FeatureProvider | None,
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epochs: int,
    compute_loss: Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor],
    progress_prefix: str = "",
) -> nn.Module:
    """Train ``model`` for ``epochs``, keeping the best-val-loss state if a val provider is given.

    Centralises the per-epoch training/eval/best-state bookkeeping
    that previously lived inline in the three from-scratch UQ
    wrappers (``ensemble._train_member``,
    ``evidential_classification._train_evidential_model``,
    ``ddu._train_ddu_classifier``). All three suffered from the same
    failure mode on CIFAR-10H (pre-fix): with frozen augmentation the
    train loss bottomed out around epoch 118 then climbed back to
    ``ln(K)`` (uniform output) by epoch 200, and the *final-epoch*
    weights were saved -- meaning the cached classifier was
    effectively a constant predictor. Tracking ``best_val_loss`` and
    loading that state at the end shields against any future
    divergence-after-best-epoch trajectory.

    Per-epoch flow:

    1. ``model.train()``; iterate ``train_provider``; call
       ``compute_loss(model, x, y)`` per batch; backprop; step the
       optimizer.
    2. If ``val_provider`` is given, ``model.eval()``; iterate it
       under ``torch.no_grad()``; aggregate the same
       ``compute_loss`` value; if the mean is a new best, snapshot
       ``model.state_dict()`` to CPU.
    3. Print one progress line.
    4. ``scheduler.step()``.

    On exit, if a best state was captured, ``model.load_state_dict``
    restores it. Otherwise ``model`` retains its final-epoch state
    (the no-val-provider fallback, used for paths that have no held-
    out split).

    Args:
        model: Module trained in place; final state matches the
            best-val-loss epoch when val tracking is on.
        train_provider: Yields ``(x, y)`` batches for training.
        val_provider: Yields ``(x, y)`` batches for validation;
            ``None`` disables val tracking.
        optimizer: Pre-built optimizer over ``model.parameters()``.
        scheduler: Pre-built LR scheduler.
        epochs: Number of epochs.
        compute_loss: Closure that computes the per-batch loss.
            Called as ``compute_loss(model, x, y)`` and must return a
            scalar ``torch.Tensor``. The closure performs the forward
            pass and is the single point where method-specific
            forward semantics (logits CE for ensemble/ddu, evidential
            CE on Dirichlet alphas for evidential) plug in.
        progress_prefix: String prepended to each per-epoch log line
            (e.g. ``"member 3/5 "``); empty by default.

    Returns:
        ``model`` with the best-val-loss state loaded (when val
        tracking is on) or its final-epoch state (when not).
    """
    device = next(model.parameters()).device
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(epochs):
        # Read the LR active during the epoch we're about to run before
        # stepping the scheduler; ``get_last_lr()`` returns the most
        # recently set LR (the initial LR before any ``.step()`` calls).
        lr_now = float(scheduler.get_last_lr()[0])

        model.train()
        train_loss_sum = 0.0
        n_train_batches = 0
        for x, y in train_provider:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            loss = compute_loss(model, x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach().item())
            n_train_batches += 1
        train_loss = train_loss_sum / max(n_train_batches, 1)

        val_loss: float | None = None
        if val_provider is not None:
            model.eval()
            val_loss_sum = 0.0
            n_val_batches = 0
            with torch.no_grad():
                for x, y in val_provider:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    loss = compute_loss(model, x, y)
                    val_loss_sum += float(loss.item())
                    n_val_batches += 1
            val_loss = val_loss_sum / max(n_val_batches, 1)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }

        if val_loss is not None:
            print(
                f"{progress_prefix}epoch {epoch + 1}/{epochs}: "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"lr={lr_now:.4f}",
                flush=True,
            )
        else:
            print(
                f"{progress_prefix}epoch {epoch + 1}/{epochs}: "
                f"train_loss={train_loss:.4f} lr={lr_now:.4f}",
                flush=True,
            )

        scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


__all__ = [
    "FeatureProvider",
    "MethodHandle",
    "build_optimizer",
    "derive_member_seed",
    "hash_config",
    "make_run_id",
    "setup_determinism",
    "train_with_val_tracking",
]
