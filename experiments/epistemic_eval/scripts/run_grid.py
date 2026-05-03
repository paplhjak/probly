#!/usr/bin/env python3
"""Orchestrator: enumerate ``(method x dataset x seed)`` runs.

Composes Stage 1 (``train_classifier.py``), Stage 2b
(``fit_uncertainty.py``), and Stage 3 (``extract_uncertainties.py``)
over a grid of ``(method, dataset, seed)`` tuples. For Task 5 this is
a thin wrapper; the real grid running happens on the cluster
(Task 9+).

``--dry-run`` prints the full set of subprocess invocations without
executing them.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
import subprocess
import sys
from typing import Iterable

import yaml

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent
_RUNS = _REPO_ROOT / "experiments/epistemic_eval/runs"

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.epistemic_eval.methods._base import make_run_id  # noqa: E402


def _enumerate_grid(grid_config: dict) -> Iterable[tuple[str, str, int]]:
    """Yield ``(method, dataset, seed)`` triples from the grid config."""
    methods = list(grid_config.get("methods", []))
    datasets = list(grid_config.get("datasets", []))
    seeds = list(grid_config.get("seeds", []))
    yield from itertools.product(methods, datasets, seeds)


def _commands_for_run(
    method: str, dataset: str, seed: int, configs_root: Path
) -> list[list[str]]:
    """Build the list of subprocess command argv arrays for one run."""
    method_config = configs_root / "methods" / f"{method}.yaml"
    dataset_config = configs_root / "datasets" / f"{dataset}.yaml"
    train_cmd = [
        sys.executable,
        str(_HERE / "train_classifier.py"),
        "--config",
        str(dataset_config),
        "--seed",
        str(seed),
    ]
    fit_cmd = [
        sys.executable,
        str(_HERE / "fit_uncertainty.py"),
        "--method-config",
        str(method_config),
        "--dataset-config",
        str(dataset_config),
        "--seed",
        str(seed),
    ]
    extract_cmd = [
        sys.executable,
        str(_HERE / "extract_uncertainties.py"),
        "--run",
        # Mirrors fit_uncertainty.py:215-218 so stage-3 reads from the
        # exact directory stage-2 wrote to.
        str(_RUNS / make_run_id(method, dataset, seed)),
    ]
    return [train_cmd, fit_cmd, extract_cmd]


def main(argv: list[str] | None = None) -> int:
    """Entry point. See module docstring."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--configs-root",
        type=Path,
        default=_REPO_ROOT / "experiments/epistemic_eval/configs",
    )
    args = parser.parse_args(argv)

    grid_config = yaml.safe_load(args.config.read_text())
    triples = list(_enumerate_grid(grid_config))
    if not triples:
        print("empty grid; nothing to do.", file=sys.stderr)
        return 1

    for method, dataset, seed in triples:
        cmds = _commands_for_run(method, dataset, seed, args.configs_root)
        if args.dry_run:
            for cmd in cmds:
                print(" ".join(cmd))
            continue
        for cmd in cmds:
            print(f"running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=False)  # noqa: S603
            if result.returncode != 0:
                print(f"command failed with exit {result.returncode}", file=sys.stderr)
                return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
