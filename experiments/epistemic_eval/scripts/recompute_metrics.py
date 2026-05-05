#!/usr/bin/env python3
"""Parallel recompute of ``metrics_<loss>.json`` across every method run.

Used after a math change in :mod:`probly.evaluation.regret_coverage`
or :mod:`compute_metrics` that bumps a version constant
(``_METRICS_VERSION`` / ``_AUREC_VERSION`` /
``_PARETO_GAP_VERSION``). Calls
:func:`experiments.epistemic_eval.scripts.compute_metrics.main` with
``--force-recompute`` once per ``(run_dir, loss)`` pair.

Concurrency: a :class:`concurrent.futures.ProcessPoolExecutor` keeps
workers alive across calls so the per-call import cost (~200-500 ms)
is amortised instead of paid every time.

Skips:
    - basecls and oracle runs (no metrics_*.json by design)
    - runs missing config.yaml or decomposition_<loss>.npz (the
      upstream pipeline hasn't reached that stage yet -- skip
      without failing)

Oracle resolution:
    For DCIC the test set varies per seed (annotator-availability
    filtering produces different per-seed subsets), so we pick the
    same-seed oracle when one exists; first-order datasets like
    CIFAR-10H share a single seed=0 oracle and we fall back to it
    when a per-seed oracle is absent.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import yaml

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.epistemic_eval.scripts import compute_metrics  # noqa: E402

_METHOD_NAMES = {"mc_dropout", "evidential", "ddu", "ensemble"}


@dataclass(frozen=True)
class Job:
    """One ``(run_dir, loss, oracle_run)`` triple to recompute."""

    run_dir: Path
    loss: str
    oracle_run: Path


def _enumerate_jobs(runs_root: Path) -> tuple[list[Job], list[str]]:
    """Walk ``runs_root`` once and produce the list of recompute jobs.

    Returns:
        ``(jobs, skip_reasons)``: the work units to dispatch, plus
        human-readable lines explaining the skipped runs.
    """
    jobs: list[Job] = []
    skips: list[str] = []

    method_runs = sorted(
        d for d in runs_root.iterdir()
        if d.is_dir() and any(f"_main_{m}_" in d.name for m in _METHOD_NAMES)
    )

    for run_dir in method_runs:
        cfg_path = run_dir / "config.yaml"
        if not cfg_path.exists():
            skips.append(f"{run_dir.name}: no config.yaml")
            continue

        cfg = yaml.safe_load(cfg_path.read_text())
        dataset_name = (cfg.get("dataset") or {}).get("name") or ""
        run_seed = cfg.get("seed")
        if not dataset_name or run_seed is None:
            skips.append(f"{run_dir.name}: config missing dataset.name or seed")
            continue

        # Same-seed oracle if available, else fall back to seed=0.
        same_seed = sorted(runs_root.glob(f"*_main_oracle_{dataset_name}_seed{run_seed}"))
        seed_zero = sorted(runs_root.glob(f"*_main_oracle_{dataset_name}_seed0"))
        if same_seed:
            oracle_run = same_seed[-1]
        elif seed_zero:
            oracle_run = seed_zero[-1]
        else:
            skips.append(f"{run_dir.name}: no oracle for dataset={dataset_name}")
            continue

        decomps = sorted(run_dir.glob("decomposition_*.npz"))
        if not decomps:
            skips.append(f"{run_dir.name}: no decompositions yet")
            continue

        for decomp in decomps:
            loss = decomp.stem.removeprefix("decomposition_")
            jobs.append(Job(run_dir=run_dir, loss=loss, oracle_run=oracle_run))

    return jobs, skips


def _run_one(job: Job) -> tuple[Job, bool, str]:
    """Worker entry point: invoke ``compute_metrics.main`` in-process."""
    argv = [
        "--run", str(job.run_dir),
        "--loss", job.loss,
        "--oracle-run", str(job.oracle_run),
        "--force-recompute",
    ]
    try:
        # compute_metrics.main writes a "wrote ..." line to stdout on
        # success; we don't care about it here, but suppressing stdout
        # globally would also hide real errors. Capture+discard via
        # contextlib.redirect_stdout on each call.
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = compute_metrics.main(argv)
        if rc != 0:
            return (job, False, f"non-zero exit {rc}")
        return (job, True, "")
    except Exception as exc:  # noqa: BLE001 - report any failure
        return (job, False, f"{type(exc).__name__}: {exc}")


def main(argv: list[str] | None = None) -> int:
    """Entry point. See module docstring."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    runs_root: Path = args.runs_root
    if not runs_root.exists():
        print(f"runs root not found: {runs_root}", file=sys.stderr)
        return 2

    jobs, skips = _enumerate_jobs(runs_root)
    n_total = len(jobs)
    print(f"queued {n_total} recompute jobs across "
          f"{len({j.run_dir for j in jobs})} run directories "
          f"(skipped {len(skips)} runs); running {args.workers} in parallel.\n")

    if not jobs:
        if skips:
            print("Skipped run details:")
            for s in skips:
                print(f"  - {s}")
        return 0

    n_done = 0
    n_failed = 0
    failures: list[tuple[Job, str]] = []
    t0 = time.monotonic()

    # ProcessPoolExecutor: workers persist across futures, so each
    # worker pays the compute_metrics import cost once.
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_one, job): job for job in jobs}
        for fut in cf.as_completed(futures):
            job, ok, msg = fut.result()
            n_done_or_failed = n_done + n_failed + 1
            if ok:
                n_done += 1
                tag = "OK  "
                detail = ""
            else:
                n_failed += 1
                tag = "FAIL"
                detail = f"  -- {msg}"
                failures.append((job, msg))
            elapsed = time.monotonic() - t0
            print(f"[{n_done_or_failed:>4d}/{n_total}] {tag} "
                  f"{job.run_dir.name} loss={job.loss}"
                  f"  ({elapsed:5.1f}s){detail}",
                  flush=True)

    print()
    print("=" * 60)
    print(" recompute_metrics complete")
    print(f"   metrics regenerated: {n_done}")
    print(f"   runs skipped:        {len(skips)}")
    print(f"   failures:            {n_failed}")
    print(f"   wall time:           {time.monotonic() - t0:.1f}s")
    print("=" * 60)

    if skips:
        print()
        print("Skipped run details:")
        for s in skips:
            print(f"  - {s}")

    if failures:
        print()
        print("Failed jobs:")
        for job, msg in failures:
            print(f"  - {job.run_dir.name} loss={job.loss}: {msg}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
