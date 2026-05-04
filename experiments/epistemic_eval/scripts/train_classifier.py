#!/usr/bin/env python3
"""Stage 1: train (or stage) the base classifier for a dataset.

For CIFAR-10H this trains a :func:`probly_benchmark.resnet.ResNet18`
from scratch on the CIFAR-10 train split per the dataset config's
``training:`` block.

For ImageNet-ReaL this **never** trains from scratch. If
``classifier.pretrained_classifier_path`` is set in the dataset
config, the file is copied (or symlinked when possible) into the run
directory as ``classifier.pth``. If it is null the script errors with
the documented message and exit code 1.

A ``--smoke-test-synthetic`` flag bypasses the dataset and uses a
tiny synthetic dataset (32 samples, 4 classes, random images) with a
small CNN that imports cleanly without torchvision data. This path
exists only for the end-to-end smoke test.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import yaml

# Ensure the methods package is importable when this script is invoked
# directly (e.g. from a smoke-test subprocess where the experiment
# package isn't on sys.path).
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.epistemic_eval.methods._base import (  # noqa: E402
    hash_config,
    make_run_id,
    setup_determinism,
)


def _git_commit() -> str:
    """Return the current git HEAD short SHA, or ``"unknown"`` on failure."""
    result = subprocess.run(  # noqa: S603
        ["git", "-C", str(_REPO_ROOT), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip()


def _write_meta(run_dir: Path, config_hash: str) -> None:
    """Write the standard ``meta.json`` to ``run_dir``."""
    meta = {
        "git_commit": _git_commit(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(),
        "run_id_hash": config_hash,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))


class _TinySyntheticCNN(nn.Module):
    """A 4-class CNN sized for the synthetic smoke test.

    Two conv layers + global average pool + linear head. Imports
    cleanly without torchvision; training runs in seconds on CPU.
    """

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


def _make_synthetic_dataset(
    n: int = 32,
    num_classes: int = 4,
    image_size: int = 8,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a tiny ``(images, soft_labels)`` tensor pair."""
    g = torch.Generator().manual_seed(seed)
    images = torch.randn(n, 3, image_size, image_size, generator=g)
    labels = torch.randint(0, num_classes, (n,), generator=g)
    soft = torch.zeros(n, num_classes)
    soft[torch.arange(n), labels] = 1.0
    return images, soft


def _train_smoke_test(run_dir: Path, seed: int) -> None:
    """Train the synthetic-CNN and write outputs.

    Used only by ``--smoke-test-synthetic`` callers (in particular the
    end-to-end pytest smoke test).
    """
    setup_determinism(seed)
    images, soft = _make_synthetic_dataset(seed=seed)
    model = _TinySyntheticCNN(num_classes=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-2)
    log_rows = []
    epochs = 2
    batch_size = 8
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(images.shape[0])
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, images.shape[0], batch_size):
            idx = perm[i : i + batch_size]
            x, y = images[idx], soft[idx]
            logits = model(x)
            log_probs = torch.log_softmax(logits, dim=1)
            loss = -(y * log_probs).sum(dim=1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().item())
            n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)
        # Trivial val on the same data; this is a smoke test.
        with torch.no_grad():
            model.eval()
            val_logits = model(images)
            val_log_probs = torch.log_softmax(val_logits, dim=1)
            val_loss = float(-(soft * val_log_probs).sum(dim=1).mean().item())
            train_acc = float((val_logits.argmax(dim=1) == soft.argmax(dim=1)).float().mean().item())
        log_rows.append((epoch, train_loss, val_loss, train_acc, train_acc))

    torch.save(model.state_dict(), run_dir / "classifier.pth")
    log_path = run_dir / "training_log.csv"
    log_path.write_text(
        "epoch,train_loss,val_loss,train_acc,val_acc\n"
        + "\n".join(f"{e},{tl:.6f},{vl:.6f},{ta:.6f},{va:.6f}" for e, tl, vl, ta, va in log_rows)
        + "\n"
    )


def _stage_pretrained(
    pretrained_path: Path, run_dir: Path
) -> None:
    """Stage a pretrained classifier checkpoint into the run dir.

    Symlinks where possible; copies on failure (e.g. cross-device
    mounts or filesystems that disallow symlinks).
    """
    target = run_dir / "classifier.pth"
    if target.exists() or target.is_symlink():
        target.unlink()
    abs_src = pretrained_path.resolve()
    if not abs_src.exists():
        msg = (
            f"pretrained_classifier_path points at {abs_src} which does not exist. "
            f"Provide a valid file."
        )
        raise FileNotFoundError(msg)
    if hasattr(os, "symlink"):
        os.symlink(abs_src, target)
    else:  # pragma: no cover - exercised only on platforms without symlinks
        shutil.copyfile(abs_src, target)


def _resolve_dataset_run_id(config: dict[str, Any], seed: int) -> str:
    """Compose a run id from ``config['name']`` and ``seed``."""
    return make_run_id(method="basecls", dataset=str(config["name"]), seed=seed)


def _read_cifar10h_training_hparams(config: dict[str, Any]) -> dict[str, Any]:
    """Extract the training hyperparameters from a CIFAR-10H config.

    Validates declared values against the small set of options the
    training loop currently supports (SGD + cosine annealing). Raises
    on unknown values rather than silently falling back -- per the
    project convention every hyperparameter must come from config.
    """
    training = config["training"]
    optimizer_name = str(training["optimizer"])
    if optimizer_name != "sgd":
        msg = f"unknown optimizer {optimizer_name!r}; only 'sgd' is supported."
        raise ValueError(msg)
    schedule = str(training["schedule"])
    if schedule != "cosine":
        msg = f"unknown schedule {schedule!r}; only 'cosine' is supported."
        raise ValueError(msg)
    aug = training["augmentation"]
    norm = training["normalization"]
    return {
        "lr": float(training["lr"]),
        "momentum": float(training["momentum"]),
        "weight_decay": float(training["weight_decay"]),
        "nesterov": bool(training["nesterov"]),
        "batch_size": int(training["batch_size"]),
        "epochs": int(training["epochs"]),
        "crop_size": int(aug["random_crop"]["size"]),
        "crop_padding": int(aug["random_crop"]["padding"]),
        "horizontal_flip": bool(aug["horizontal_flip"]),
        "mean": tuple(float(x) for x in norm["mean"]),
        "std": tuple(float(x) for x in norm["std"]),
    }


def _train_cifar10h_run(
    config: dict[str, Any],
    run_dir: Path,
    seed: int,
) -> None:
    """Train a ``ResNet18`` from scratch on CIFAR-10 train per the config.

    Reads SGD/cosine/200-epoch hyperparameters from
    ``config['training']``; uses
    :class:`probly_benchmark.resnet.ResNet18` as the classifier.
    Loads CIFAR-10 train via ``torchvision.datasets.CIFAR10`` with
    ``download=True`` (cached at ``loader_kwargs['root']``); splits
    50k train into a 45k train / 5k val partition deterministically by
    ``seed``. Trains, prints epoch/loss/val-acc per epoch, and writes:

    * ``classifier.pth`` -- the model state_dict (CPU tensors).
    * ``training_log.csv`` -- ``epoch,train_loss,val_loss,train_acc,val_acc``.

    Idempotent: if ``classifier.pth`` exists with a matching
    ``classifier.config_hash`` sidecar, the function logs a "skipped"
    message and returns without retraining.

    Determinism follows the project convention: torch / numpy global
    seeds + ``torch.use_deterministic_algorithms(True, warn_only=True)``
    via :func:`_base.setup_determinism`.

    Args:
        config: Resolved CIFAR-10H dataset config (from YAML).
        run_dir: Output directory; must already exist.
        seed: Root seed for the run.

    Raises:
        ValueError: On unknown optimizer or schedule, or if the locked
            ``num_classes`` does not match the ``ResNet18`` architecture
            (which is hardcoded to 10).
    """
    # Late imports: keep torchvision and the heavy ResNet18 module
    # out of the import path of the synthetic-only smoke test. The
    # canonical pickled CIFAR-10 batches at
    # ``data/cifar10h/cifar-10-batches-py/`` are produced from the
    # raw PNGs by ``build_canonical_cifar10_pickles.py``; the
    # canonical Toronto host has been flaking with 503s and the HF
    # mirrors don't carry the exact tarball, so we materialise the
    # batches locally. :class:`CIFAR10NoMD5` is a thin subclass of
    # ``torchvision.datasets.CIFAR10`` that bypasses the canonical
    # MD5 check (our pickles encode the same image data but are
    # not byte-identical to the tarball). Test-set ordering follows
    # canonical CIFAR-10 -- the same ordering ``cifar10h-counts.npy``
    # uses, so downstream alignment with the human soft labels is
    # automatic.
    from torchvision import transforms as T  # noqa: PLC0415

    from experiments.epistemic_eval.datasets.cifar10_canonical import (  # noqa: PLC0415
        CIFAR10NoMD5,
    )
    from probly_benchmark.resnet import ResNet18  # noqa: PLC0415

    classifier_path = run_dir / "classifier.pth"
    hash_sidecar = run_dir / "classifier.config_hash"
    config_hash = hash_config(config)
    if (
        classifier_path.exists()
        and hash_sidecar.exists()
        and hash_sidecar.read_text().strip() == config_hash
    ):
        print(f"skipped: classifier exists at {classifier_path}.")
        return

    setup_determinism(seed)

    hp = _read_cifar10h_training_hparams(config)
    num_classes = int(config["classifier"]["num_classes"])
    if num_classes != 10:
        msg = (
            "probly_benchmark.resnet.ResNet18 is hardcoded to 10 classes; "
            f"got num_classes={num_classes}. Use a different architecture "
            "or extend probly_benchmark.resnet to accept a num_classes arg."
        )
        raise ValueError(msg)

    # Augmentation pipelines per the locked CIFAR-10 recipe in
    # decisions.md "CIFAR-10H training data".
    train_transforms: list[Any] = [
        T.RandomCrop(hp["crop_size"], padding=hp["crop_padding"]),
    ]
    if hp["horizontal_flip"]:
        train_transforms.append(T.RandomHorizontalFlip())
    train_transforms.extend([T.ToTensor(), T.Normalize(hp["mean"], hp["std"])])
    train_transform = T.Compose(train_transforms)
    val_transform = T.Compose([T.ToTensor(), T.Normalize(hp["mean"], hp["std"])])

    # CIFAR-10 train via the canonical-format pickles. We use the
    # CIFAR-10H dataset's loader_kwargs.root since both
    # ``cifar-10-batches-py/`` (training data) and
    # ``cifar-10h-master/data/cifar10h-counts.npy`` (human soft labels)
    # live under the same dataset root in the project's data layout.
    # We instantiate twice -- once with train-time augmentations,
    # once with eval-only transforms -- and split the indices via a
    # seeded permutation; the val subset uses the eval-transform
    # copy so augmentation noise doesn't perturb val_acc.
    cifar_root = config.get("loader_kwargs", {}).get("root", "data/cifar10h")
    train_ds = CIFAR10NoMD5(root=cifar_root, train=True, transform=train_transform)
    val_ds = CIFAR10NoMD5(root=cifar_root, train=True, transform=val_transform)
    n = len(train_ds)
    val_size = max(1, int(0.1 * n))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    val_indices = perm[:val_size].tolist()
    train_indices = perm[val_size:].tolist()
    train_subset = torch.utils.data.Subset(train_ds, train_indices)
    val_subset = torch.utils.data.Subset(val_ds, val_indices)

    pin_memory = torch.cuda.is_available()
    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=hp["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )
    val_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=hp["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=hp["lr"],
        momentum=hp["momentum"],
        weight_decay=hp["weight_decay"],
        nesterov=hp["nesterov"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp["epochs"])
    criterion = nn.CrossEntropyLoss()

    log_rows: list[tuple[int, float, float, float, float]] = []
    for epoch in range(hp["epochs"]):
        model.train()
        ep_loss = 0.0
        ep_correct = 0
        ep_total = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bsz = x.size(0)
            ep_loss += float(loss.detach().item()) * bsz
            ep_correct += int((logits.argmax(dim=1) == y).sum().item())
            ep_total += bsz
        train_loss = ep_loss / max(ep_total, 1)
        train_acc = ep_correct / max(ep_total, 1)

        model.eval()
        v_loss = 0.0
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)
                loss = criterion(logits, y)
                bsz = x.size(0)
                v_loss += float(loss.item()) * bsz
                v_correct += int((logits.argmax(dim=1) == y).sum().item())
                v_total += bsz
        val_loss = v_loss / max(v_total, 1)
        val_acc = v_correct / max(v_total, 1)

        scheduler.step()
        log_rows.append((epoch, train_loss, val_loss, train_acc, val_acc))
        print(
            f"epoch {epoch + 1}/{hp['epochs']}: "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}",
            flush=True,
        )

    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state_dict, classifier_path)
    log_path = run_dir / "training_log.csv"
    log_path.write_text(
        "epoch,train_loss,val_loss,train_acc,val_acc\n"
        + "\n".join(f"{e},{tl:.6f},{vl:.6f},{ta:.6f},{va:.6f}" for e, tl, vl, ta, va in log_rows)
        + "\n"
    )
    hash_sidecar.write_text(config_hash)
    print(f"trained classifier written to {classifier_path}.")


def _read_dcic_training_hparams(config: dict[str, Any]) -> dict[str, Any]:
    """Extract the DCIC training hyperparameters from a dataset config.

    DCIC datasets share a single fine-tuning recipe (AdamW + cosine
    schedule, 50 epochs with early stopping on a 10% val split,
    patience 10). The recipe was tuned from QualityMRI smoke-test
    findings: lr=1e-3 was too aggressive (val_loss spiked at epoch 2),
    20 epochs left the model under-converged, and patience=4 fired
    on noisy val_loss from a 25-image val set. This helper parses
    the locked block and surfaces typed values; it raises on unknown
    optimizer / schedule values rather than silently falling back.
    """
    training = config["training"]
    optimizer_name = str(training["optimizer"])
    if optimizer_name != "adamw":
        msg = (
            f"unknown optimizer {optimizer_name!r} for DCIC; only "
            f"'adamw' is supported (the locked recipe)."
        )
        raise ValueError(msg)
    schedule = str(training.get("schedule", "cosine"))
    if schedule != "cosine":
        msg = (
            f"unknown schedule {schedule!r} for DCIC; only 'cosine' "
            f"is supported (the locked recipe)."
        )
        raise ValueError(msg)
    return {
        "lr": float(training["lr"]),
        "weight_decay": float(training["weight_decay"]),
        "batch_size": int(training["batch_size"]),
        "epochs": int(training["epochs"]),
        "patience": int(training.get("patience", 10)),
        "val_fraction": float(training.get("val_fraction", 0.1)),
        "num_workers": int(training.get("num_workers", 0)),
        "schedule": schedule,
        "horizontal_flip": bool(training.get("augmentation", {}).get("horizontal_flip", True)),
    }


def _train_dcic_run(
    config: dict[str, Any],
    run_dir: Path,
    seed: int,
) -> None:
    """Train a DCIC base classifier with the locked AdamW recipe.

    Steps:

    1. Resolve the seed's test fold via
       :func:`experiments.epistemic_eval.datasets.dcic.test_fold_for_seed`
       and build (train, val) :class:`FeatureProvider` over the four
       non-test folds with a 10% val carve-out (per
       ``decisions.md`` -> "DCIC datasets" and Oleg's reference recipe
       in :mod:`experiments.first_order_data.dcic_ensemble_pipeline`).
    2. Build a fresh torchvision ResNet-18 with ImageNet weights and
       a K-class linear head via
       :func:`experiments.epistemic_eval.datasets.dcic.make_resnet18_factory`.
    3. Train with AdamW + soft-label cross-entropy. Early-stop on val
       loss with patience ``training.patience`` (default 4).
    4. Persist the best (lowest-val-loss) state_dict as
       ``classifier.pth`` and dump per-epoch metrics to
       ``training_log.csv``.

    Idempotent: if ``classifier.pth`` exists with a matching
    ``classifier.config_hash`` sidecar, return without retraining.
    """
    from experiments.epistemic_eval.datasets.dcic import (  # noqa: PLC0415
        build_train_val_providers,
        make_resnet18_factory,
    )

    classifier_path = run_dir / "classifier.pth"
    hash_sidecar = run_dir / "classifier.config_hash"
    config_hash = hash_config(config)
    if (
        classifier_path.exists()
        and hash_sidecar.exists()
        and hash_sidecar.read_text().strip() == config_hash
    ):
        print(f"skipped: classifier exists at {classifier_path}.")
        return

    setup_determinism(seed)
    hp = _read_dcic_training_hparams(config)
    num_classes = int(config["classifier"]["num_classes"])

    train_provider, val_provider = build_train_val_providers(
        config,
        seed=seed,
        val_fraction=hp["val_fraction"],
        batch_size=hp["batch_size"],
        num_workers=hp["num_workers"],
    )

    # Hard guard against the config/data mismatch that bit MiceBone
    # (the DCIC README table claimed 4 classes but the annotations
    # only carry 3, so a (B, 4) head against (B, 3) soft labels
    # crashed at the first batch). The loader's ``n_classes`` is
    # derived from the unique labels actually present in
    # ``annotations.json``; if the YAML disagrees, fix the YAML.
    if int(train_provider.n_classes) != num_classes:
        msg = (
            f"DCIC dataset config declares classifier.num_classes="
            f"{num_classes} but the loader observes "
            f"n_classes={train_provider.n_classes} unique labels in "
            f"annotations.json. Fix classifier.num_classes (and "
            f"metadata.num_classes) in the dataset YAML so the head "
            f"matches the data."
        )
        raise ValueError(msg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_resnet18_factory(num_classes=num_classes, pretrained=True)().to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=hp["lr"],
        weight_decay=hp["weight_decay"],
    )
    # Cosine LR schedule over the full epoch budget. The
    # ``CosineAnnealingLR`` formula decays smoothly from ``lr`` to ~0
    # across ``T_max`` epochs; combined with ``patience``-based early
    # stopping this means a run that early-stops keeps the higher-lr
    # checkpoint while a run that goes the full distance benefits
    # from the smaller end-of-cosine lr.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(hp["epochs"], 1)
    )

    log_rows: list[tuple[int, float, float, float, float]] = []
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(hp["epochs"]):
        # Read the LR active during the epoch we're about to run
        # before stepping; ``scheduler.get_last_lr()`` reports the
        # most recently set LR (the initial ``lr`` before any
        # ``scheduler.step()`` calls).
        lr_now = float(scheduler.get_last_lr()[0])
        model.train()
        ep_loss = 0.0
        ep_correct = 0
        ep_total = 0
        for x, y in train_provider:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            log_probs = torch.log_softmax(logits, dim=1)
            loss = -(y * log_probs).sum(dim=1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bsz = x.size(0)
            ep_loss += float(loss.detach().item()) * bsz
            ep_correct += int((logits.argmax(dim=1) == y.argmax(dim=1)).sum().item())
            ep_total += bsz
        train_loss = ep_loss / max(ep_total, 1)
        train_acc = ep_correct / max(ep_total, 1)

        model.eval()
        v_loss = 0.0
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for x, y in val_provider:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)
                log_probs = torch.log_softmax(logits, dim=1)
                loss = -(y * log_probs).sum(dim=1).mean()
                bsz = x.size(0)
                v_loss += float(loss.item()) * bsz
                v_correct += int((logits.argmax(dim=1) == y.argmax(dim=1)).sum().item())
                v_total += bsz
        val_loss = v_loss / max(v_total, 1)
        val_acc = v_correct / max(v_total, 1)

        log_rows.append((epoch, train_loss, val_loss, train_acc, val_acc))
        print(
            f"epoch {epoch + 1}/{hp['epochs']}: "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
            f"lr={lr_now:.5f}",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= hp["patience"]:
                print(
                    f"early stopping at epoch {epoch + 1}/{hp['epochs']} "
                    f"(no val_loss improvement in {hp['patience']} epochs).",
                    flush=True,
                )
                break

        # Step the scheduler once per epoch (after early-stopping
        # bookkeeping so a run that early-stops doesn't advance the
        # cosine past the epoch we actually trained).
        scheduler.step()

    if best_state is None:
        # Fall through to the post-final-epoch state if validation never
        # produced a finite loss; this should never happen in practice
        # but keeps the function from silently writing nothing.
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
    torch.save(best_state, classifier_path)
    log_path = run_dir / "training_log.csv"
    log_path.write_text(
        "epoch,train_loss,val_loss,train_acc,val_acc\n"
        + "\n".join(
            f"{e},{tl:.6f},{vl:.6f},{ta:.6f},{va:.6f}"
            for e, tl, vl, ta, va in log_rows
        )
        + "\n"
    )
    hash_sidecar.write_text(config_hash)
    print(f"trained DCIC classifier written to {classifier_path}.")


def main(argv: list[str] | None = None) -> int:
    """Entry point. See module docstring."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Dataset YAML config.")
    parser.add_argument("--seed", type=int, default=0, help="Root seed for this run.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Override the auto-computed run directory (mainly for tests).",
    )
    parser.add_argument(
        "--smoke-test-synthetic",
        action="store_true",
        help="Bypass the dataset and use a tiny synthetic dataset and CNN.",
    )
    args = parser.parse_args(argv)

    if args.smoke_test_synthetic:
        run_dir = args.run_dir or (_REPO_ROOT / "experiments/epistemic_eval/runs" / make_run_id("basecls", "synthetic", args.seed))
        run_dir.mkdir(parents=True, exist_ok=True)
        config_payload = {"name": "synthetic_smoke", "seed": args.seed}
        (run_dir / "config.yaml").write_text(yaml.safe_dump(config_payload, sort_keys=True))
        _write_meta(run_dir, hash_config(config_payload))
        _train_smoke_test(run_dir, args.seed)
        print(f"smoke-test classifier written to {run_dir / 'classifier.pth'}")
        return 0

    if args.config is None:
        print("--config is required unless --smoke-test-synthetic is passed.", file=sys.stderr)
        return 1
    config = yaml.safe_load(args.config.read_text())

    run_dir = args.run_dir or (_REPO_ROOT / "experiments/epistemic_eval/runs" / _resolve_dataset_run_id(config, args.seed))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
    _write_meta(run_dir, hash_config(config))

    name = config.get("name")
    classifier_cfg = config.get("classifier", {})
    pretrained_path = classifier_cfg.get("pretrained_classifier_path")

    if name == "imagenet_real":
        if pretrained_path is None:
            print(
                "ImageNet-ReaL training from scratch is not supported. Provide\n"
                "pretrained_classifier_path in the dataset config. See decisions.md\n"
                "'ImageNet-ReaL training data' for the rationale.",
                file=sys.stderr,
            )
            return 1
        _stage_pretrained(Path(pretrained_path), run_dir)
        print(f"skipped: pretrained provided ({pretrained_path})")
        return 0

    if name == "cifar10h":
        if pretrained_path is not None:
            _stage_pretrained(Path(pretrained_path), run_dir)
            print(f"skipped: pretrained provided ({pretrained_path})")
            return 0
        _train_cifar10h_run(config, run_dir, args.seed)
        return 0

    if name == "appa_real":
        # APPA-REAL has no base classifier; the pipeline uses
        # extract_features.py + a head-only fit. Surface this clearly
        # rather than silently producing nothing.
        print(
            "appa_real does not use train_classifier.py; use extract_features.py "
            "to cache backbone features for the linear-probe path."
        )
        return 0

    if config.get("family") == "dcic":
        # DCIC datasets dispatch to the AdamW + early-stopping recipe
        # in :func:`_train_dcic_run`. The seed picks the test fold via
        # :mod:`experiments.epistemic_eval.datasets.dcic.test_fold_for_seed`
        # so each seed exercises a different held-out fold.
        if pretrained_path is not None:
            _stage_pretrained(Path(pretrained_path), run_dir)
            print(f"skipped: pretrained provided ({pretrained_path})")
            return 0
        _train_dcic_run(config, run_dir, args.seed)
        return 0

    print(f"unknown dataset name: {name!r}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
