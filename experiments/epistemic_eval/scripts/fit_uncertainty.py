#!/usr/bin/env python3
"""Stage 2b: fit a UQ method, save its handle.

Reads the cached classifier (``classifier.pth`` from
``train_classifier.py``) for full-network mode or the cached features
(``data/appa_real_features/<split>.npz`` from ``extract_features.py``)
for linear-probe mode. Imports the method module via
``importlib.import_module(method_config['method_module'])`` -- no
method names are hard-coded here. Adding a new method (e.g. LLLA in
Task 5b) requires only a new ``methods/llla.py`` and a new method
config; this script does not change.

Outputs:

* ``runs/<run_id>/method.pth``: handle blob via
  ``methods/<name>.save``.
* ``runs/<run_id>/config.yaml``: resolved merged (method + dataset)
  config.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
import yaml

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.epistemic_eval.methods._base import (  # noqa: E402
    FeatureProvider,
    make_run_id,
    setup_determinism,
)
from probly.method.head import MlpHead  # noqa: E402

#: UQ methods that train (or otherwise iterate the training set) at
#: ``fit`` time when run on a from-scratch full-network dataset (no
#: ``ensemble_classifier_paths`` set in the dataset config). These
#: methods need a real CIFAR-10 train provider, not the metadata-only
#: stub that the load-pretrained branch uses.
#:
#: Membership rationale (one entry per method that needs it):
#:   * ``ensemble``: trains N independent members from scratch with
#:     derived seeds (Lakshminarayanan et al. 2017).
#:   * ``evidential``: trains a base classifier with the soft-label
#:     evidential cross-entropy (Sensoy et al. 2018).
#:   * ``ddu``: Phase A trains the spectral-norm-restricted
#:     classifier; Phase B walks the train set to fit the GMM
#:     density head (Mukhoti et al. 2023).
#:
#: Mc_dropout is intentionally absent: in full-network mode it loads
#: a pretrained classifier and only applies probly's dropout
#: transformation, with ``epochs = 0`` short-circuiting any
#: training-loop iteration. New methods that need real train data at
#: fit time must be added here AND wired through their respective
#: wrapper.
_FROM_SCRATCH_FULL_NETWORK_METHODS: frozenset[str] = frozenset({
    "ensemble",
    "evidential",
    "ddu",
})


def _resolve_callable(dotted: str) -> Any:  # noqa: ANN401
    """Resolve a dotted path like ``a.b:c`` or ``a.b.c`` to a callable."""
    if ":" in dotted:
        module_path, attr = dotted.split(":", 1)
    else:
        module_path, _, attr = dotted.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def _build_full_network_factory(
    dataset_config: dict[str, Any],
) -> tuple[Any, dict[str, Any] | None]:
    """Build a ``model_factory`` callable for ``full_network`` mode.

    Returns ``(factory, head_factory_args)``. ``head_factory_args`` is
    ``None`` because the architecture is the full classifier.

    DCIC datasets (``family == "dcic"``) take a dedicated path that
    constructs a torchvision ``resnet18`` with ImageNet-pretrained
    weights and a fresh K-class linear head, mirroring Oleg's
    reference :class:`ImageEntmaxClassifier` recipe. Other datasets
    use the dataset-config-driven architecture spec.
    """
    if dataset_config.get("family") == "dcic":
        from experiments.epistemic_eval.datasets.dcic import (  # noqa: PLC0415
            make_resnet18_factory,
        )

        classifier_cfg = dataset_config.get("classifier", {})
        num_classes = int(classifier_cfg.get("num_classes", 0))
        pretrained = bool(classifier_cfg.get("pretrained", True))
        return make_resnet18_factory(
            num_classes=num_classes, pretrained=pretrained
        ), None

    classifier_cfg = dataset_config.get("classifier", {})
    arch = classifier_cfg["architecture"]
    cls_or_fn = _resolve_callable(arch)
    num_classes = int(classifier_cfg.get("num_classes", 0))

    def factory() -> nn.Module:
        # Many torchvision factories take num_classes; ResNet18 from
        # probly_benchmark takes none. Try with kwargs first, fall back.
        try:
            return cls_or_fn(num_classes=num_classes)  # type: ignore[call-arg]
        except TypeError:
            return cls_or_fn()

    return factory, None


def _build_head_factory(
    dataset_config: dict[str, Any],
    method_config: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Build a ``model_factory`` callable for ``linear_probe`` mode.

    Returns ``(factory, head_factory_args)``. ``head_factory_args``
    is the kwargs dict used by the factory; saved into the handle so
    extraction can reconstruct the architecture.
    """
    backbone_cfg = dataset_config.get("backbone", {})
    head_cfg = dataset_config.get("head", {})
    in_features = int(backbone_cfg.get("feature_dim", 0))
    num_classes = int(dataset_config.get("metadata", {}).get("num_classes", 0))
    if num_classes == 0:
        # Fall back to head config or len(age_support); the test
        # suite passes num_classes via dataset_config['num_classes'].
        num_classes = int(dataset_config.get("num_classes", 0))
    hidden = int(head_cfg.get("hidden", 256))
    dropout_p = float(method_config.get("head_dropout_p", 0.1))
    args: dict[str, Any] = {
        "in_features": in_features,
        "num_classes": num_classes,
        "hidden": hidden,
        "dropout_p": dropout_p,
    }

    def factory() -> nn.Module:
        return MlpHead(
            in_features=int(args["in_features"]),
            num_classes=int(args["num_classes"]),
            hidden=int(args["hidden"]),
            dropout_p=float(args["dropout_p"]),
        )

    return factory, args


def _make_full_network_cifar10h_train_provider(
    dataset_config: dict[str, Any],
    batch_size: int = 128,
) -> FeatureProvider:
    """Build a real CIFAR-10H training provider for from-scratch ensembles.

    Used by the ensemble path that trains each member from scratch
    (no ``ensemble_classifier_paths`` and no pretrained checkpoint
    in the dataset config). Loads CIFAR-10 train via
    :class:`CIFAR10NoMD5`, one-hot encodes the integer labels so they
    plug into the soft-label cross-entropy loss in
    :func:`ensemble._train_member`, and returns a
    :class:`FeatureProvider` over the full 50k training set.
    """
    from torchvision import transforms as T  # noqa: PLC0415

    from experiments.epistemic_eval.datasets.cifar10_canonical import (  # noqa: PLC0415
        CIFAR10NoMD5,
    )

    norm = dataset_config["training"]["normalization"]
    crop = dataset_config["training"]["augmentation"].get("random_crop", {})
    transforms_list: list[Any] = []
    if crop:
        transforms_list.append(T.RandomCrop(int(crop["size"]), padding=int(crop.get("padding", 0))))
    if dataset_config["training"]["augmentation"].get("horizontal_flip"):
        transforms_list.append(T.RandomHorizontalFlip())
    transforms_list.extend([
        T.ToTensor(),
        T.Normalize(tuple(float(x) for x in norm["mean"]), tuple(float(x) for x in norm["std"])),
    ])
    transform = T.Compose(transforms_list)

    cifar_root = dataset_config.get("loader_kwargs", {}).get("root", "data/cifar10h")
    train_ds = CIFAR10NoMD5(root=cifar_root, train=True, transform=transform)
    n = len(train_ds)
    num_classes = int(dataset_config.get("classifier", {}).get("num_classes", 10))
    indices = np.arange(n, dtype=np.int64)

    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        xs: list[torch.Tensor] = []
        ys: list[int] = []
        for i in range(start, stop):
            x, y = train_ds[i]
            xs.append(x)
            ys.append(int(y))
        x_batch = torch.stack(xs, dim=0)
        # One-hot encode int labels into a (B, K) float tensor so the
        # soft-label cross-entropy in ensemble._train_member /
        # mc_dropout._train_mc_dropout_model
        # (``loss = -(y * log_probs).sum(dim=1).mean()``) gets the
        # right input dtype + shape.
        y_batch = torch.zeros(len(ys), num_classes, dtype=torch.float32)
        y_batch[torch.arange(len(ys)), torch.tensor(ys, dtype=torch.long)] = 1.0
        batches.append((x_batch, y_batch))

    return FeatureProvider(
        batches=batches,
        mode="full_network",
        feature_dim=None,
        n_classes=num_classes,
        indices=indices,
    )


def _make_full_network_stub_provider(
    dataset_config: dict[str, Any],
) -> FeatureProvider:
    """Build a metadata-only stub provider for the load-pretrained path.

    Full-network MC-Dropout in Task 5's flow is "load classifier,
    apply ``probly.method.dropout()``, save handle" -- no training,
    so :func:`mc_dropout.fit` runs zero epochs and never iterates the
    provider. Same for :func:`ensemble.fit` when
    ``ensemble_classifier_paths`` is set: it loads ``N`` state_dicts
    from disk and never iterates.

    Real test-time iteration over CIFAR-10 / ImageNet test images
    happens at extract time, on the cluster. The stub here exposes the
    correct ``n_classes`` metadata for handle bookkeeping and zero
    batches for the (unused) training loop.
    """
    classifier_cfg = dataset_config.get("classifier", {})
    num_classes = int(classifier_cfg.get("num_classes", 0)) or int(
        dataset_config.get("metadata", {}).get("num_classes", 0)
    )
    return FeatureProvider(
        batches=[],
        mode="full_network",
        feature_dim=None,
        n_classes=max(num_classes, 1),
        indices=np.zeros(0, dtype=np.int64),
    )


def _resolve_classifier_state_dict_path(
    dataset_config: dict[str, Any],
    seed: int,
    explicit: Path | None,
) -> Path:
    """Resolve the path to stage 1's ``classifier.pth``.

    Resolution order: explicit ``--classifier-state-dict-path`` arg ->
    ``dataset_config['classifier']['pretrained_classifier_path']`` ->
    default at ``runs/<basecls_run_id>/classifier.pth`` where
    ``basecls_run_id`` is :func:`make_run_id` with ``method='basecls'``.
    """
    if explicit is not None:
        return Path(explicit)
    classifier_cfg = dataset_config.get("classifier", {})
    pretrained = classifier_cfg.get("pretrained_classifier_path")
    if pretrained:
        return Path(pretrained)
    return (
        _REPO_ROOT
        / "experiments/epistemic_eval/runs"
        / make_run_id(method="basecls", dataset=str(dataset_config["name"]), seed=seed)
        / "classifier.pth"
    )


def _load_classifier_state_dict(path: Path) -> dict[str, torch.Tensor]:
    """Load a classifier state_dict from disk.

    Accepts either a raw ``state_dict`` or a ``{"state_dict": ...}``
    blob (the convention used by :mod:`train_classifier`).
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "state_dict" in blob and isinstance(blob["state_dict"], dict):
        return {k: v.detach().cpu() for k, v in blob["state_dict"].items()}
    return {k: v.detach().cpu() for k, v in blob.items()}


def _wrap_factory_with_state_dict(
    base_factory: Any,  # noqa: ANN401
    state_dict: dict[str, torch.Tensor],
) -> Any:  # noqa: ANN401
    """Return a factory that calls ``base_factory()`` and loads ``state_dict``.

    Ensemble's :func:`fit` calls the factory once per member; each call
    must produce a *fresh* nn.Module with the loaded weights, so
    members don't share parameters via Python aliasing. ``load_state_dict``
    deep-copies the tensors into the new module.
    """

    def factory() -> nn.Module:
        model = base_factory()
        model.load_state_dict(state_dict)
        return model

    return factory


def _make_linear_probe_provider(
    dataset_config: dict[str, Any],
    cache_dir: Path,
    split: str,
    batch_size: int = 128,
) -> FeatureProvider:
    """Build a feature-mode provider from cached ``.npz`` features.

    When the cache file carries a ``targets`` array (the contract
    extract_features.py writes for APPA-REAL: row-stochastic ``p*``
    per image), use it. Old caches that pre-date the targets contract
    (and the synthetic test fixtures that don't yield labels) fall
    back to uniform-stub labels — those callers are training-recipe
    tests that don't measure loss anyway.
    """
    cache_file = cache_dir / f"{split}.npz"
    if not cache_file.exists():
        msg = f"feature cache not found at {cache_file}; run extract_features.py first."
        raise FileNotFoundError(msg)
    blob = np.load(cache_file, allow_pickle=True)
    features = np.asarray(blob["features"], dtype=np.float32)
    n = features.shape[0]
    feature_dim = int(features.shape[1])
    num_classes = int(dataset_config.get("metadata", {}).get("num_classes", 0))
    if "targets" in blob.files:
        labels = np.asarray(blob["targets"], dtype=np.float32)
        if num_classes <= 0:
            num_classes = int(labels.shape[1])
    else:
        labels = np.full(
            (n, max(num_classes, 1)), 1.0 / max(num_classes, 1), dtype=np.float32
        )
    indices = np.arange(n, dtype=np.int64)
    batches = []
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        x = torch.from_numpy(features[start:stop])
        y = torch.from_numpy(labels[start:stop])
        batches.append((x, y))
    return FeatureProvider(
        batches=batches,
        mode="linear_probe",
        feature_dim=feature_dim,
        n_classes=max(num_classes, 1),
        indices=indices,
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point. See module docstring."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-config", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        default=_REPO_ROOT / "data" / "appa_real_features",
    )
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument(
        "--classifier-state-dict-path",
        type=Path,
        default=None,
        help=(
            "Override the auto-resolved path to stage 1's classifier.pth "
            "(full-network mode only). Default: "
            "runs/<basecls_<dataset>_seed<N>>/classifier.pth."
        ),
    )
    args = parser.parse_args(argv)

    method_config = yaml.safe_load(args.method_config.read_text())
    dataset_config = yaml.safe_load(args.dataset_config.read_text())

    setup_determinism(args.seed)

    method_module = importlib.import_module(method_config["method_module"])
    extraction_mode = dataset_config.get("extraction_mode", "full_network")

    method_config_with_args = dict(method_config)

    # Dataset-driven recipe override: a dataset config can declare
    # ``training.method_overrides`` (e.g. DCIC datasets with the
    # AdamW fine-tune recipe vs CIFAR-10H's SGD-cosine from-scratch
    # recipe). This is the documented bridge between cross-method
    # recipe uniformity (the locked recipe in the method YAMLs) and
    # cross-dataset recipe variation (a 224x224 ImageNet-pretrained
    # backbone needs different hyperparameters than a 32x32
    # from-scratch CIFAR ResNet). Unknown keys are passed through;
    # the wrappers ignore anything they don't read.
    method_overrides = (
        dataset_config.get("training", {}).get("method_overrides", {}) or {}
    )
    if method_overrides:
        method_config_with_args.update(dict(method_overrides))

    if extraction_mode == "linear_probe":
        factory, head_factory_args = _build_head_factory(dataset_config, method_config)
        provider = _make_linear_probe_provider(
            dataset_config, args.feature_cache_dir, args.train_split
        )
        if head_factory_args is not None:
            method_config_with_args["head_factory_args"] = head_factory_args
    else:
        # Full-network: three sub-paths, dispatched on whether the
        # method trains from scratch and whether per-member
        # checkpoints are pre-supplied.
        #
        # 1. ``ensemble`` + ``ensemble_classifier_paths`` set
        #    (ImageNet-ReaL): :func:`ensemble.fit` loads the N
        #    pretrained checkpoints itself; we hand it a stub
        #    provider and a base factory, no training happens here.
        # 2. Method ∈ ``_FROM_SCRATCH_FULL_NETWORK_METHODS`` AND no
        #    ``ensemble_classifier_paths`` (CIFAR-10H from-scratch
        #    training): build a real CIFAR-10H training provider so
        #    the wrapper's fit() can iterate real data. Pass an
        #    unwrapped base factory (each ``model_factory()`` call
        #    yields a freshly initialised model). Currently only
        #    CIFAR-10H is wired; other datasets raise loudly.
        # 3. ``mc_dropout`` (and any other "wrap a pretrained
        #    classifier" method not in the from-scratch set): load
        #    stage-1's classifier, wrap the factory with its
        #    state_dict, and short-circuit training via
        #    ``epochs = 0``.
        method_name = method_config.get("name")
        base_factory, _ = _build_full_network_factory(dataset_config)

        if method_name == "ensemble" and dataset_config.get("ensemble_classifier_paths"):
            # Path 1.
            provider = _make_full_network_stub_provider(dataset_config)
            factory = base_factory
        elif method_name in _FROM_SCRATCH_FULL_NETWORK_METHODS:
            # Path 2: from-scratch full-network training (currently
            # ensemble / evidential / ddu). All three iterate the
            # train provider in their fit() (ensemble: per-member
            # training; evidential: classifier training under the
            # evidential CE loss; ddu: Phase A classifier training
            # + Phase B GMM density-head fit on encoder features).
            # Wired for CIFAR-10H (dedicated train provider) and for
            # the DCIC family (per-seed test fold drops out, train
            # is the union of the other 4 folds). ImageNet-ReaL is
            # expected to use the ensemble_classifier_paths path
            # with one of the frozen ensemble checkpoint sets.
            dataset_name = dataset_config.get("name")
            if dataset_name == "cifar10h":
                provider = _make_full_network_cifar10h_train_provider(dataset_config)
            elif dataset_config.get("family") == "dcic":
                from experiments.epistemic_eval.datasets.dcic import (  # noqa: PLC0415
                    build_full_train_provider,
                )

                training_block = dataset_config.get("training", {}) or {}
                provider = build_full_train_provider(
                    dataset_config,
                    seed=args.seed,
                    batch_size=int(training_block.get("batch_size", 32)),
                    num_workers=int(training_block.get("num_workers", 0)),
                    augment=True,
                )
                # Guard against the config/data num_classes mismatch
                # that bit MiceBone (DCIC README said 4, the data has
                # 3). The wrapper's training loop multiplies (B, K)
                # soft labels by (B, K) log-probs; a mismatch crashes
                # at the first batch. The loader's n_classes is
                # derived from the unique labels in annotations.json.
                declared_num_classes = int(
                    dataset_config.get("classifier", {}).get("num_classes", 0)
                )
                if int(provider.n_classes) != declared_num_classes:
                    msg = (
                        f"DCIC dataset {dataset_name!r}: classifier.num_classes="
                        f"{declared_num_classes} disagrees with the loader's "
                        f"n_classes={provider.n_classes} (unique labels in "
                        f"annotations.json). Fix the dataset YAML so the head "
                        f"matches the data."
                    )
                    raise ValueError(msg)
            else:
                msg = (
                    f"from-scratch full-network training for method "
                    f"{method_name!r} is not wired for dataset "
                    f"{dataset_name!r}. Either provide "
                    f"`ensemble_classifier_paths` in the dataset config "
                    f"or extend fit_uncertainty.py with a per-dataset "
                    f"training provider."
                )
                raise NotImplementedError(msg)
            factory = base_factory
        else:
            # Path 3.
            provider = _make_full_network_stub_provider(dataset_config)
            classifier_path = _resolve_classifier_state_dict_path(
                dataset_config, args.seed, args.classifier_state_dict_path
            )
            if not classifier_path.exists():
                print(
                    f"missing stage-1 classifier at {classifier_path}; run "
                    "train_classifier.py first or pass "
                    "--classifier-state-dict-path.",
                    file=sys.stderr,
                )
                return 1
            state_dict = _load_classifier_state_dict(classifier_path)
            factory = _wrap_factory_with_state_dict(base_factory, state_dict)
            # Skip training: the load-pretrained flow just needs to wrap
            # the loaded model with probly.method.dropout (MC-Dropout)
            # and persist the handle. epochs=0 short-circuits the
            # training loop in mc_dropout._train_mc_dropout_model.
            if method_name == "mc_dropout":
                method_config_with_args["epochs"] = 0

    handle = method_module.fit(
        method_config=method_config_with_args,
        dataset_config=dataset_config,
        data_provider=provider,
        model_factory=factory,
        seed=args.seed,
    )

    run_dir = args.run_dir or (
        _REPO_ROOT / "experiments/epistemic_eval/runs"
        / make_run_id(method_config["name"], dataset_config["name"], args.seed)
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    method_module.save(handle, run_dir / "method.pth")

    merged = {"method": method_config, "dataset": dataset_config, "seed": args.seed}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(merged, sort_keys=True))

    print(f"method handle written to {run_dir / 'method.pth'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
