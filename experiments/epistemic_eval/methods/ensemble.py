"""Deep Ensembles wrapper for the epistemic-eval pipeline.

For paper reproducibility, all results are reported as mean +/- std
across 5 seeds. CUDA non-determinism (atomic ops on GPU) may cause
sub-1e-4 logit drift across runs; this is well below inter-seed
variance and does not affect any reported metric. Bit-identical
reproducibility holds only on CPU with deterministic algorithms
enabled.

Three modes:

* ``linear_probe`` (APPA-REAL): each member is a freshly initialised
  :class:`probly.method.head.MlpHead`, trained on the cached features
  from the frozen ResNet-101 backbone. Per-member seeds are derived
  from the root seed via :func:`_base.derive_member_seed`.
* ``full_network`` from scratch (CIFAR-10H): each member is a fresh
  classifier (e.g. CIFAR ``ResNet18``) trained from scratch on the
  training split. Per-member seeds are derived from the root seed.
* ``full_network`` from pretrained checkpoints (ImageNet-ReaL): per
  the locked decision option (a) (see ``decisions.md`` "ImageNet-ReaL
  ensemble construction"), the dataset config supplies a list
  ``ensemble_classifier_paths`` of ``N`` independently pretrained
  classifiers; we load each into the handle's ``state_dicts`` list.
  No training; this is Lakshminarayanan et al. 2017's standard form.

The wrapper bypasses probly's :class:`Sampler` because we want raw
pre-softmax logits cached on disk, not a
:class:`CategoricalDistribution` (see the Bypass-of-Sampler comment
in :func:`extract`).

Each member runs in ``model.eval()`` at extraction time -- per
``decisions.md`` "APPA-REAL backbone strategy", dropout is disabled
for ensemble members so each one is deterministic post-training.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ._base import (
    FeatureProvider,
    build_optimizer,
    derive_member_seed,
    setup_determinism,
    train_with_val_tracking,
)


@dataclass
class EnsembleHandle:
    """Per-run state produced by :func:`fit`.

    Attributes:
        state_dicts: One state-dict per ensemble member (CPU tensors).
        head_factory_args: Args used by the head factory in
            ``linear_probe`` mode; ``None`` in ``full_network`` mode.
        member_seeds: Per-member seeds derived from ``seed`` via
            :func:`_base.derive_member_seed`.
        method_config: Resolved method config dict.
        dataset_config: Resolved dataset config dict.
        seed: Root seed for this run.
    """

    state_dicts: list[dict[str, torch.Tensor]] = field(default_factory=list)
    head_factory_args: dict[str, Any] | None = None
    member_seeds: list[int] = field(default_factory=list)
    method_config: dict[str, Any] = field(default_factory=dict)
    dataset_config: dict[str, Any] = field(default_factory=dict)
    seed: int = 0


def _required(config: dict[str, Any], key: str) -> Any:  # noqa: ANN401
    """Read ``key`` from ``config`` or raise loudly if missing."""
    if key not in config:
        msg = f"missing required config key '{key}'"
        raise KeyError(msg)
    return config[key]


#: Fixed RNG seed used to partition the train pool into per-member folds.
#: Independent of the run seed so seeds 0..4 of an experiment grid all
#: see the same fold structure (and only differ in member init); this
#: separates "fold-rotation diversity" from "init diversity" cleanly.
_FOLD_PARTITION_SEED: int = 0


def _make_member_subset_provider(
    provider: FeatureProvider,
    member_idx: int,
    n_members: int,
) -> FeatureProvider:
    """Return a FeatureProvider whose batches drop the member_idx-th fold.

    Concatenates the provider's existing batches into one ``(X, Y)``
    pair, partitions the rows into ``n_members`` deterministic folds via
    a fixed-seed permutation (see :data:`_FOLD_PARTITION_SEED`), and
    rebuilds batches over the ``n_members - 1`` folds that don't include
    ``member_idx``.

    Each member therefore trains on ``(n_members - 1) / n_members`` of
    the train pool with a different fold dropped per member; pairs of
    members share ``(n_members - 2) / n_members`` of their training
    rows. Used by the linear-probe APPA-REAL ensemble path to inject
    data-side diversity in addition to init-side diversity (per
    :doc:`decisions.md` ``-> "APPA-REAL ensemble train rotation"``).

    The test set is never touched -- ``data_provider`` here is the
    train provider; the test-time provider is built separately by
    :mod:`extract_uncertainties` and is identical for every member.

    Args:
        provider: Train-time :class:`FeatureProvider`.
        member_idx: 0-based ensemble member index.
        n_members: Total ensemble size.

    Returns:
        A new :class:`FeatureProvider` whose ``batches`` cover the
        non-dropped folds. ``mode``, ``feature_dim``, ``n_classes`` are
        carried through. ``indices`` is the corresponding subset.
    """
    if not provider.batches:
        return provider
    if n_members <= 1:
        return provider
    all_x = torch.cat([b[0] for b in provider.batches], dim=0)
    all_y = torch.cat([b[1] for b in provider.batches], dim=0)
    n_total = int(all_x.shape[0])
    rng = np.random.default_rng(_FOLD_PARTITION_SEED)
    perm = rng.permutation(n_total)
    fold_size = max(n_total // n_members, 1)
    fold_of = np.zeros(n_total, dtype=np.int64)
    for pos, idx in enumerate(perm):
        # Cap the last fold at n_members - 1 so the residue lands there
        # rather than overflowing past n_members - 1.
        fold_of[int(idx)] = min(pos // fold_size, n_members - 1)
    keep = np.flatnonzero(fold_of != int(member_idx))
    subset_x = all_x[keep]
    subset_y = all_y[keep]
    if provider.batches:
        batch_size = int(provider.batches[0][0].shape[0])
    else:
        batch_size = 128
    new_batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for start in range(0, len(keep), batch_size):
        stop = min(start + batch_size, len(keep))
        new_batches.append((subset_x[start:stop], subset_y[start:stop]))
    if int(provider.indices.shape[0]) == n_total:
        new_indices = provider.indices[keep]
    else:
        new_indices = np.arange(len(keep), dtype=np.int64)
    return FeatureProvider(
        batches=new_batches,
        mode=provider.mode,
        feature_dim=provider.feature_dim,
        n_classes=provider.n_classes,
        indices=new_indices,
    )


def _soft_label_ce(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Soft-label cross-entropy on logit-emitting classifiers.

    ``y`` is a row-stochastic ``(B, K)`` target (one-hot for
    CIFAR-10/DCIC int labels, or a dense ``p*`` for soft-labelled
    datasets); ``model(x)`` returns logits ``(B, K)``. Equivalent to
    :func:`torch.nn.functional.cross_entropy` for one-hot ``y`` and to
    KL plus entropy for general ``y``.
    """
    logits = model(x)
    log_probs = torch.log_softmax(logits, dim=1)
    return -(y * log_probs).sum(dim=1).mean()


def _train_member(
    model: nn.Module,
    data_provider: FeatureProvider,
    method_config: dict[str, Any],
    seed: int,
    *,
    val_provider: FeatureProvider | None = None,
    member_idx: int = 0,
    n_members: int = 1,
) -> nn.Module:
    """Train a single ensemble member with cross-entropy on soft labels.

    Runs on GPU when one is available; falls back to CPU otherwise.
    Per-member training of a ResNet-18-class model on CIFAR-10 train
    is ~50x faster on a consumer GPU than on CPU.

    Optimisation recipe (locked across all four UQ methods, mirrors
    :mod:`scripts.train_classifier`'s basecls path):
    SGD with momentum (Nesterov configurable) and a cosine learning-
    rate schedule with ``T_max == epochs``.

    When ``val_provider`` is given, val_loss is computed at the end
    of every epoch and the model state from the lowest-val-loss epoch
    is restored before returning. This shields against the
    divergence-after-best-epoch trajectory observed on CIFAR-10H
    pre-fix (loss reached 2.07 at epoch 118 then climbed back to
    ``ln(10)`` by epoch 200).

    Per-epoch progress is printed to stdout in the format
    ``member <idx>/<n> epoch <e>/<E>: train_loss=<float>
    [val_loss=<float>] lr=<float>`` so SLURM logs surface training
    progress and silent-no-op regressions are detectable at a glance.
    """
    setup_determinism(seed)
    epochs = int(method_config.get("epochs", 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = build_optimizer(
        model,
        optimizer_name=str(method_config.get("optimizer", "sgd")).lower(),
        lr=float(method_config.get("lr", 1.0e-3)),
        momentum=float(method_config.get("momentum", 0.9)),
        weight_decay=float(method_config.get("weight_decay", 0.0)),
        nesterov=bool(method_config.get("nesterov", False)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )
    return train_with_val_tracking(
        model,
        data_provider,
        val_provider,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        compute_loss=_soft_label_ce,
        progress_prefix=f"member {member_idx + 1}/{n_members} ",
    )


def fit(
    method_config: dict[str, Any],
    dataset_config: dict[str, Any],
    data_provider: FeatureProvider,
    model_factory: Callable[[], nn.Module],
    seed: int,
    *,
    val_data_provider: FeatureProvider | None = None,
) -> EnsembleHandle:
    """Train (or load) ``n_members`` ensemble members.

    ImageNet-ReaL path: if ``dataset_config['ensemble_classifier_paths']``
    is set and ``extraction_mode == 'full_network'``, each member is
    loaded from disk -- no training, per ``decisions.md``
    "ImageNet-ReaL ensemble construction" option (a).

    Otherwise (CIFAR-10H full-network from scratch, or APPA-REAL
    linear-probe) each member is trained from scratch with a per-member
    seed derived from ``seed`` via :func:`_base.derive_member_seed`.

    Args:
        method_config: Resolved method config dict.
        dataset_config: Resolved dataset config dict.
        data_provider: Training data provider.
        model_factory: Zero-argument callable returning a fresh
            :class:`torch.nn.Module`. Called once per member.
        seed: Root seed for this run.

    Returns:
        The trained :class:`EnsembleHandle`.

    Raises:
        ValueError: If ``n_members > len(ensemble_classifier_paths)``
            on the ImageNet-ReaL path, per the locked failure mode.
    """
    setup_determinism(seed)
    n_members = int(_required(method_config, "n_members"))
    member_seeds = [derive_member_seed(seed, i) for i in range(n_members)]

    raw_paths = dataset_config.get("ensemble_classifier_paths")
    extraction_mode = dataset_config.get("extraction_mode")
    use_pretrained = raw_paths is not None and extraction_mode == "full_network"

    state_dicts: list[dict[str, torch.Tensor]] = []
    if use_pretrained:
        paths: list[str] = list(raw_paths) if raw_paths is not None else []
        if n_members > len(paths):
            msg = (
                f"ensemble.n_members={n_members} but ensemble_classifier_paths has "
                f"only {len(paths)} entries. Provide N classifier paths in the "
                f"dataset config or reduce n_members."
            )
            raise ValueError(msg)
        for path in paths[:n_members]:
            blob = torch.load(Path(path), map_location="cpu", weights_only=False)
            # Accept either a raw state_dict or a {"state_dict": ...} blob.
            if isinstance(blob, dict) and "state_dict" in blob and isinstance(blob["state_dict"], dict):
                sd = blob["state_dict"]
            else:
                sd = blob
            state_dicts.append({k: v.detach().cpu() for k, v in sd.items()})
    else:
        # Per-member train rotation (off by default for back-compat
        # with CIFAR-10H/DCIC; opted in by the APPA-REAL dataset config
        # via ``training.method_overrides.per_member_train_rotation: true``).
        # When on, each member sees a different (n_members-1)/n_members
        # subset of the train pool; this gives ensembles a data-side
        # source of diversity beyond random init, which is necessary
        # in regimes where heads otherwise converge to a shared basin
        # (frozen-features APPA-REAL).
        per_member_rotation = bool(
            method_config.get("per_member_train_rotation", False)
        )
        for member_idx, member_seed in enumerate(member_seeds):
            setup_determinism(member_seed)
            model = model_factory()
            member_provider = (
                _make_member_subset_provider(data_provider, member_idx, n_members)
                if per_member_rotation
                else data_provider
            )
            trained = _train_member(
                model,
                member_provider,
                method_config,
                member_seed,
                val_provider=val_data_provider,
                member_idx=member_idx,
                n_members=n_members,
            )
            state_dicts.append({k: v.detach().cpu() for k, v in trained.state_dict().items()})

    head_factory_args = method_config.get("head_factory_args")
    return EnsembleHandle(
        state_dicts=state_dicts,
        head_factory_args=dict(head_factory_args) if head_factory_args is not None else None,
        member_seeds=member_seeds,
        method_config=dict(method_config),
        dataset_config=dict(dataset_config),
        seed=int(seed),
    )


def save(handle: EnsembleHandle, path: Path) -> None:
    """Persist a handle to ``path`` via ``torch.save``.

    Args:
        handle: The handle to save.
        path: Destination file (parent directory must exist).
    """
    torch.save(asdict(handle), path)


def load(path: Path) -> EnsembleHandle:
    """Reload a handle previously written by :func:`save`.

    Args:
        path: Source file produced by :func:`save`.

    Returns:
        The reconstructed :class:`EnsembleHandle`.
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return EnsembleHandle(**blob)


def extract(
    handle: EnsembleHandle,
    data_provider: FeatureProvider,
    n_samples: int | None = None,  # noqa: ARG001
    *,
    model_factory: Callable[[], nn.Module],
) -> dict[str, np.ndarray]:
    """Run forward passes for every ensemble member.

    Returns a dict with::

        {"logits": np.ndarray of shape (N, K, n_members) float32,
         "indices": np.ndarray of shape (N,) int64}

    The ``n_samples`` argument is ignored for ensembles (``S`` is fixed
    to ``n_members``); it's part of the signature for symmetry with
    :func:`mc_dropout.extract`.

    Bypass-of-Sampler note:

    .. code-block::

        # probly's Sampler converts to CategoricalDistribution via
        # create_categorical_distribution_from_logits
        # (src/probly/predictor/_common.py:195-200). We cache raw
        # pre-softmax logits so the loss-specific decompositions
        # (Task 6.5) can produce a (N, K, S) tensor that downstream
        # code is free to interpret. We therefore run forward passes
        # manually instead of using Sampler.predict().

    Args:
        handle: A trained handle produced by :func:`fit`.
        data_provider: Data provider for the test split.
        n_samples: Ignored; kept for API symmetry with MC-Dropout.
        model_factory: Zero-argument callable returning the same
            architecture used at ``fit`` time. Called once per member.

    Returns:
        Dict with ``"logits"`` (``(N, K, n_members)`` float32) and
        ``"indices"`` (``(N,)`` int64).
    """
    setup_determinism(handle.seed)
    n = data_provider.n_samples
    k = data_provider.n_classes
    n_members = len(handle.state_dicts)
    logits_out = np.zeros((n, k, n_members), dtype=np.float32)

    # probly's Sampler converts to CategoricalDistribution via
    # create_categorical_distribution_from_logits
    # (src/probly/predictor/_common.py:195-200). We cache raw
    # pre-softmax logits so the loss-specific decompositions
    # (Task 6.5) can produce a (N, K, S) tensor that downstream
    # code is free to interpret. We therefore run forward passes
    # manually instead of using Sampler.predict().
    # Run inference on GPU when one is available; falls back to CPU
    # otherwise. Each ensemble member's forward pass is independent,
    # so we move the per-member model to GPU once and stream batches
    # to it.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for member_idx, sd in enumerate(handle.state_dicts):
            model = model_factory()
            model.load_state_dict(sd)
            model = model.to(device)
            # Per decisions.md "APPA-REAL backbone strategy": dropout
            # is disabled at inference for ensemble members.
            model.eval()
            offset = 0
            for x, _ in data_provider:
                x = x.to(device, non_blocking=True)
                out = model(x).detach().cpu().to(torch.float32).numpy()
                bsz = out.shape[0]
                logits_out[offset : offset + bsz, :, member_idx] = out
                offset += bsz
    return {
        "logits": logits_out,
        "indices": data_provider.indices.astype(np.int64, copy=False),
    }


__all__ = ["EnsembleHandle", "extract", "fit", "load", "save"]
