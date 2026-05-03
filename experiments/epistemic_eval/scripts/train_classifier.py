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


def _train_cifar10h_run(
    config: dict[str, Any],  # noqa: ARG001
    run_dir: Path,  # noqa: ARG001
    seed: int,  # noqa: ARG001
) -> None:
    """Train a ``ResNet18`` from scratch on CIFAR-10 train per the config.

    Training requires the CIFAR-10 train data to be present at
    ``loader_kwargs['root']``. Pulling and training the real model is
    a multi-hour job on a single H200 GPU; the function exists so the
    code path is in place for the cluster runs (Task 9). The smoke
    test path uses ``--smoke-test-synthetic`` and never executes this
    function.
    """
    msg = (
        "CIFAR-10H from-scratch training loop not yet implemented. "
        "Hyperparameters are in configs/datasets/cifar10h.yaml under "
        "'training:'. Implement in scripts/train_classifier.py before "
        "Task 9. See decisions.md 'CIFAR-10H training data' for context. "
        "For local smoke testing use --smoke-test-synthetic."
    )
    raise NotImplementedError(msg)


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

    print(f"unknown dataset name: {name!r}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
