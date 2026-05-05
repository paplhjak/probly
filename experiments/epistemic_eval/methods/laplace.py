"""Laplace approximation wrapper for the epistemic-eval pipeline.

Linear-probe regime only (APPA-REAL). The pipeline:

1. ``fit()`` builds a fresh :class:`probly.method.head.MlpHead` (with
   ``head_dropout_p`` overridden to 0 by default so the MAP point is
   a clean cross-entropy MAP), trains it on the soft-label provider
   for ``epochs`` steps, then wraps it with
   :class:`laplace.Laplace` and fits the posterior precision matrix.
2. ``optimize_prior_precision`` is called to tune the prior precision
   via type-II marginal-likelihood (the "marglik" method on a separate
   internal grid). This is the standard Daxberger 2021 recipe.
3. ``extract()`` calls ``la.predictive_samples(x, pred_type="glm",
   n_samples=S)`` to draw ``S`` softmax-probability samples per test
   input. We log them so the cached array fits the existing
   ``output_schema: "logits_nks"`` contract -- ``softmax(log p) == p``,
   so downstream BMA computation in ``compute_metrics`` is unchanged.

The wrapper bypasses probly's :class:`Sampler` for the same reason
:mod:`mc_dropout` does: we want raw per-sample tensors cached on
disk in the loss-agnostic ``(N, K, S)`` shape that the existing
decomposition modules consume.

Backend: we force the ``AsdlGGN`` backend because the default
``CurvlinopsGGN`` requires the upstream curvlinops 3.0+ ``_base``
internal API, which conflicts with our pinned numpy 2.x. ASDL works
fine for the small linear-probe head and is what the Daxberger paper
uses for its small-scale experiments.

Reproducibility: per-sample stochasticity at extraction time comes
from PyTorch's global RNG inside ``predictive_samples`` (samples
weights from :math:`\\mathcal{N}(\\theta_{\\text{MAP}}, H^{-1})`).
We seed the global RNG once at the top of :func:`fit` and
:func:`extract` via :func:`_base.setup_determinism`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ._base import FeatureProvider, setup_determinism


@dataclass
class LaplaceHandle:
    """Per-run state produced by :func:`fit`.

    Attributes:
        state_dict: MAP head weights (CPU tensors).
        laplace_state_dict: Serialised state of the fitted
            :class:`laplace.Laplace` object (posterior precision,
            mean, prior precision, etc.); produced by
            ``Laplace.state_dict()`` on the CPU. Restored at extract
            time via ``Laplace.load_state_dict``.
        head_factory_args: Args needed to reconstruct the head
            architecture at extract time.
        method_config: Resolved method config dict.
        dataset_config: Resolved dataset config dict.
        seed: Root seed for this run.
    """

    state_dict: dict[str, torch.Tensor] = field(default_factory=dict)
    laplace_state_dict: dict[str, Any] = field(default_factory=dict)
    head_factory_args: dict[str, Any] | None = None
    method_config: dict[str, Any] = field(default_factory=dict)
    dataset_config: dict[str, Any] = field(default_factory=dict)
    seed: int = 0


def _required(config: dict[str, Any], key: str) -> Any:  # noqa: ANN401
    """Read ``key`` from ``config`` or raise loudly if missing."""
    if key not in config:
        msg = f"missing required config key '{key}'"
        raise KeyError(msg)
    return config[key]


def _train_map_head(
    model: nn.Module,
    data_provider: FeatureProvider,
    method_config: dict[str, Any],
    seed: int,
) -> nn.Module:
    """Train the head's MAP point with soft-label cross-entropy.

    AdamW + plain cross-entropy on soft targets, mirroring the
    :func:`mc_dropout._train_mc_dropout_model` recipe so the MAP
    point is comparable with what MCD trains. The Laplace posterior
    is then centred on this MAP.
    """
    setup_determinism(seed)
    epochs = int(method_config.get("epochs", 50))
    lr = float(method_config.get("lr", 1.0e-3))
    weight_decay = float(method_config.get("weight_decay", 0.0))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for x, y in data_provider:
            logits = model(x)
            log_probs = torch.log_softmax(logits, dim=1)
            loss = -(y * log_probs).sum(dim=1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().item())
            n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)
        print(
            f"epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f}",
            flush=True,
        )
    return model


def _build_laplace_loader(
    data_provider: FeatureProvider,
) -> torch.utils.data.DataLoader[Any]:
    """Materialise the data_provider into a DataLoader of ``(x, hard_y)``.

    ``laplace-torch``'s ``BaseLaplace.fit`` expects a DataLoader that
    yields ``(features, integer_labels)`` for classification. Our
    soft-label provider yields ``(features, p*)``. We argmax each row
    to derive a hard label -- the standard reduction used by every
    Laplace-on-soft-labels tutorial and by ``laplace-torch`` itself
    when given Categorical-style targets.

    The MAP point already sits at the soft-label CE optimum (we
    trained it that way in :func:`_train_map_head`); the only thing
    we lose by argmaxing for the Hessian step is some second-order
    information about label smoothness. For APPA-REAL's Gaussian-
    spread votes this is a mild approximation.
    """
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for x, y_soft in data_provider:
        xs.append(x.detach())
        ys.append(y_soft.detach().argmax(dim=1).long())
    x_cat = torch.cat(xs, dim=0)
    y_cat = torch.cat(ys, dim=0)
    dataset = torch.utils.data.TensorDataset(x_cat, y_cat)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )


def _build_laplace(
    model: nn.Module,
    *,
    subset_of_weights: str = "last_layer",
    hessian_structure: str = "full",
) -> Any:  # noqa: ANN401
    """Construct a Laplace object using the ASDL backend.

    Defaults are LLLA-style (``subset_of_weights='last_layer'``,
    ``hessian_structure='full'``): only the last linear layer's
    weights get a Gaussian posterior, the rest stay at MAP. This
    keeps the Hessian a manageable ``last_layer_params x
    last_layer_params`` matrix (~13k x 13k for our 128-hidden
    101-class APPA-REAL head, ~680 MB float32). With
    ``subset_of_weights='all'`` and the same head, a full Hessian
    would be 275k x 275k floats = 300 GB and OOM the GPU.

    Forced ASDL because the default curvlinops backend requires
    curvlinops>=3 with a ``_base`` internal that conflicts with our
    pinned numpy 2.x; pinning curvlinops<3 downgrades numpy and
    breaks probly. ASDL is the second-most-mature backend in
    ``laplace-torch`` and is the recommended default for small
    linear-probe heads.

    Args:
        model: The MAP head to wrap.
        subset_of_weights: ``"last_layer"`` (default; LLLA) or
            ``"all"`` (full-head Laplace, requires Kron/diag for
            large heads).
        hessian_structure: ``"full"`` (default with ``last_layer``),
            ``"kron"``, ``"diag"``, or ``"lowrank"``.
    """
    from laplace import Laplace  # noqa: PLC0415
    from laplace.curvature.asdl import AsdlGGN  # noqa: PLC0415

    return Laplace(
        model,
        likelihood="classification",
        subset_of_weights=subset_of_weights,
        hessian_structure=hessian_structure,
        backend=AsdlGGN,
    )


def fit(
    method_config: dict[str, Any],
    dataset_config: dict[str, Any],
    data_provider: FeatureProvider,
    model_factory: Callable[[], nn.Module],
    seed: int,
    *,
    val_data_provider: FeatureProvider | None = None,  # noqa: ARG001
) -> LaplaceHandle:
    """Train the head MAP, fit Laplace, persist as a handle.

    Linear-probe only: ``model_factory()`` returns a fresh
    :class:`MlpHead`; we expect the method config to set
    ``head_dropout_p: 0.0`` so the MAP head has no dropout in the
    architecture (Laplace samples weights, not activations -- having
    a dropout layer in the head would conflate two stochasticity
    sources at extract time).

    Args:
        method_config: Resolved method config dict. Reads
            ``epochs``, ``lr``, ``weight_decay``,
            ``optimize_prior_precision_method`` (default
            ``"marglik"``).
        dataset_config: Resolved dataset config dict.
        data_provider: Linear-probe provider yielding
            ``(features, p*)`` batches over the train split.
        model_factory: Zero-argument callable returning a fresh head.
        seed: Root seed for this run.

    Returns:
        The trained :class:`LaplaceHandle`.
    """
    if data_provider.mode != "linear_probe":
        msg = (
            f"laplace wrapper currently only supports linear_probe mode; "
            f"got data_provider.mode={data_provider.mode!r}. "
            f"Extending to full_network requires deciding which subset of "
            f"weights to apply Laplace to (last-layer vs all)."
        )
        raise NotImplementedError(msg)

    base_model = model_factory()
    setup_determinism(seed)
    trained = _train_map_head(base_model, data_provider, method_config, seed)
    map_state = {k: v.detach().cpu() for k, v in trained.state_dict().items()}

    # Move to the device Laplace will fit on (laplace-torch infers
    # device from the model's parameters).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained.to(device)

    laplace_loader = _build_laplace_loader(data_provider)
    subset_of_weights = str(method_config.get("subset_of_weights", "last_layer"))
    hessian_structure = str(method_config.get("hessian_structure", "full"))
    la = _build_laplace(
        trained,
        subset_of_weights=subset_of_weights,
        hessian_structure=hessian_structure,
    )
    # Move loader tensors to device on each yield. laplace-torch
    # iterates the loader internally; we wrap to add .to(device).
    class _DeviceLoader:
        def __init__(self, base: Any, dev: torch.device) -> None:
            self.base = base
            self.dev = dev

        def __iter__(self) -> Any:
            for x, y in self.base:
                yield x.to(self.dev, non_blocking=True), y.to(
                    self.dev, non_blocking=True
                )

        def __len__(self) -> int:
            return len(self.base)

        @property
        def dataset(self) -> Any:
            return self.base.dataset

    la.fit(_DeviceLoader(laplace_loader, device))

    # Type-II MLE for the prior precision; the standard Daxberger
    # 2021 recipe. ``method='marglik'`` uses analytical marginal
    # likelihood -- closed form for a Gaussian posterior over
    # parameters -- and is the most stable choice for a small head.
    method = str(method_config.get("optimize_prior_precision_method", "marglik"))
    la.optimize_prior_precision(method=method)

    # Serialise via the official laplace-torch state_dict API so all
    # internal bookkeeping (n_data, mean, posterior precision, prior
    # precision, sigma_noise, etc.) round-trips without us tracking
    # individual fields by hand. Keeps the wrapper robust to future
    # internal changes in laplace-torch.
    laplace_sd_raw = la.state_dict()
    laplace_sd = {
        k: (v.detach().cpu() if hasattr(v, "detach") else v)
        for k, v in laplace_sd_raw.items()
    }
    head_factory_args = method_config.get("head_factory_args")
    return LaplaceHandle(
        state_dict=map_state,
        laplace_state_dict=laplace_sd,
        head_factory_args=dict(head_factory_args) if head_factory_args is not None else None,
        method_config=dict(method_config),
        dataset_config=dict(dataset_config),
        seed=int(seed),
    )


def save(handle: LaplaceHandle, path: Path) -> None:
    """Persist a handle to ``path`` via ``torch.save``."""
    torch.save(asdict(handle), path)


def load(path: Path) -> LaplaceHandle:
    """Reload a handle previously written by :func:`save`."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return LaplaceHandle(**blob)


def _rehydrate_laplace(
    handle: LaplaceHandle,
    head: nn.Module,
) -> Any:  # noqa: ANN401
    """Reconstruct a fitted ``Laplace`` from a handle via ``load_state_dict``.

    Builds a fresh Laplace shell with the same backend / subset /
    hessian-structure as fit-time, then restores the posterior
    precision, mean, and prior precision via the official
    ``laplace-torch`` state-dict API.
    """
    subset_of_weights = str(handle.method_config.get("subset_of_weights", "last_layer"))
    hessian_structure = str(handle.method_config.get("hessian_structure", "full"))
    la = _build_laplace(
        head,
        subset_of_weights=subset_of_weights,
        hessian_structure=hessian_structure,
    )
    device = next(head.parameters()).device
    state = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in handle.laplace_state_dict.items()
    }
    la.load_state_dict(state)
    return la


def extract(
    handle: LaplaceHandle,
    data_provider: FeatureProvider,
    n_samples: int,
    *,
    model_factory: Callable[[], nn.Module],
) -> dict[str, np.ndarray]:
    """Sample ``S = n_samples`` parameter draws and forward each.

    Returns ``{"logits": (N, K, S) float32, "indices": (N,) int64}``.
    The cached array is ``log(predictive_samples)`` -- not pre-softmax
    logits in the usual sense, but ``softmax(log p) == p`` so
    downstream BMA / decomposition code consumes them identically.

    Args:
        handle: A trained :class:`LaplaceHandle`.
        data_provider: Linear-probe test provider.
        n_samples: ``S`` -- number of posterior samples.
        model_factory: Fresh head builder.

    Returns:
        Dict with ``"logits"`` ``(N, K, S)`` float32 and ``"indices"``
        ``(N,)`` int64.
    """
    setup_determinism(handle.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    head = model_factory()
    head.load_state_dict(handle.state_dict)
    head.to(device).eval()

    la = _rehydrate_laplace(handle, head)
    pred_type = str(handle.method_config.get("pred_type", "glm"))

    n = data_provider.n_samples
    k = data_provider.n_classes
    out = np.zeros((n, k, n_samples), dtype=np.float32)

    log_eps = float(np.finfo(np.float32).tiny)
    offset = 0
    with torch.no_grad():
        for x, _ in data_provider:
            x = x.to(device, non_blocking=True)
            # predictive_samples returns (S, B, K) softmax probs
            samples = la.predictive_samples(
                x, pred_type=pred_type, n_samples=n_samples
            )
            # Log-prob; clip to avoid -inf for any underflowed entries.
            log_samples = torch.log(samples.clamp_min(log_eps))
            # Reorder (S, B, K) -> (B, K, S) to match logits_nks.
            log_samples = log_samples.permute(1, 2, 0).detach().cpu().numpy()
            bsz = log_samples.shape[0]
            out[offset : offset + bsz, :, :] = log_samples.astype(np.float32)
            offset += bsz

    return {
        "logits": out,
        "indices": data_provider.indices.astype(np.int64, copy=False),
    }


__all__ = ["LaplaceHandle", "extract", "fit", "load", "save"]
