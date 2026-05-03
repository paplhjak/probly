"""Verifies the SLURM array task-id -> (method, seed) mapping.

The mapping in ``cluster/slurm/run_grid.sh`` is the load-bearing
piece: a wrong index assignment would silently train the wrong
method on the wrong seed without any other failure mode. We pin it
here by re-running the same arithmetic in a controlled bash
environment and asserting the resolved values.

Also runs ``bash -n`` against ``run_grid.sh`` so any future syntax
regression in the script is caught at test time, not after `sbatch`.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RUN_GRID = (
    _REPO_ROOT
    / "experiments"
    / "epistemic_eval"
    / "cluster"
    / "slurm"
    / "run_grid.sh"
)

# (TASK_ID, expected_method, expected_seed). This is the locked
# table (also documented in cluster/slurm/README.md). Any divergence
# between the bash arithmetic in run_grid.sh and this table is a
# bug in one of the two; the test asserts they agree.
_EXPECTED: list[tuple[int, str, int]] = [
    (0, "mc_dropout", 0),
    (1, "mc_dropout", 1),
    (2, "mc_dropout", 2),
    (3, "evidential", 0),
    (4, "evidential", 1),
    (5, "evidential", 2),
    (6, "ddu", 0),
    (7, "ddu", 1),
    (8, "ddu", 2),
    (9, "ensemble", 0),
    (10, "ensemble", 1),
    (11, "ensemble", 2),
]


def test_run_grid_script_syntax() -> None:
    """``bash -n run_grid.sh`` succeeds; the script is parseable."""
    result = subprocess.run(  # noqa: S603
        ["bash", "-n", str(_RUN_GRID)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_run_grid_script_is_executable() -> None:
    """The script has the executable bit set (so ``bash run_grid.sh`` works)."""
    import os
    assert os.access(_RUN_GRID, os.X_OK), (
        f"{_RUN_GRID} is not executable; run `chmod +x` on it."
    )


def _resolve_via_bash(task_id: int) -> tuple[str, int]:
    """Re-run run_grid.sh's mapping arithmetic for a given TASK_ID.

    We don't execute run_grid.sh itself (it would try to ``cd`` to
    the cluster path and run uv). Instead we replay the small
    METHODS/METHOD_IDX/SEED snippet that lives in the script;
    ``test_mapping_snippet_matches_script_text`` (below) checks
    the snippet stays in sync with the script's source so a future
    edit can't silently desync the test.
    """
    snippet = f"""
        METHODS=("mc_dropout" "evidential" "ddu" "ensemble")
        METHOD_IDX=$(( {task_id} / 3 ))
        SEED=$(( {task_id} % 3 ))
        METHOD="${{METHODS[$METHOD_IDX]}}"
        echo "$METHOD $SEED"
    """
    result = subprocess.run(  # noqa: S603
        ["bash", "-c", snippet],
        capture_output=True,
        text=True,
        check=True,
    )
    method, seed_str = result.stdout.strip().split()
    return method, int(seed_str)


@pytest.mark.parametrize(("task_id", "expected_method", "expected_seed"), _EXPECTED)
def test_task_id_maps_to_expected_method_and_seed(
    task_id: int, expected_method: str, expected_seed: int
) -> None:
    """Each of the 12 TASK_IDs resolves to its locked (method, seed)."""
    method, seed = _resolve_via_bash(task_id)
    assert method == expected_method
    assert seed == expected_seed


def test_mapping_snippet_matches_script_text() -> None:
    """The mapping arithmetic in this test file matches run_grid.sh.

    Pins that the bash snippet we replay in
    ``_resolve_via_bash`` is byte-identical to the snippet inside
    ``run_grid.sh``. Without this guard, a future refactor of the
    script could change the ordering of ``METHODS`` (e.g. swap two
    entries) and the parametrised test above would still pass
    because both sides change together.
    """
    script_text = _RUN_GRID.read_text()
    # Each fragment must appear verbatim in the script. Any future
    # edit to run_grid.sh's mapping (e.g. swapping ``ensemble`` and
    # ``ddu``) would break this assertion AND break the locked
    # table in README.md, surfacing the change loudly.
    expected_fragments = [
        'METHODS=("mc_dropout" "evidential" "ddu" "ensemble")',
        "METHOD_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))",
        "SEED=$(( SLURM_ARRAY_TASK_ID % 3 ))",
        'METHOD="${METHODS[$METHOD_IDX]}"',
    ]
    for frag in expected_fragments:
        assert frag in script_text, (
            f"missing fragment in run_grid.sh: {frag!r}"
        )


def test_array_size_matches_grid_size() -> None:
    """``#SBATCH --array=0-11`` matches the 4 methods x 3 seeds = 12 size."""
    script_text = _RUN_GRID.read_text()
    assert "#SBATCH --array=0-11" in script_text
    # 4 methods x 3 seeds = 12; the array is inclusive 0..11.
    assert len(_EXPECTED) == 12


def test_locked_method_configs_referenced_not_dryrun() -> None:
    """run_grid.sh must use the locked production configs, NOT *_dryrun.yaml.

    The dryrun configs (epochs:1, n_members:2) are for cluster
    verification only; using them in the production grid would
    silently produce non-paper-quality results. Locking the path
    string here is cheap insurance.
    """
    script_text = _RUN_GRID.read_text()
    assert "configs/methods/${METHOD}.yaml" in script_text
    assert "_dryrun.yaml" not in script_text


def test_oracle_pinned_to_seed_zero() -> None:
    """Per the locked design: the grid reads the seed=0 oracle for all tasks.

    ``decisions.md`` and the README both state the oracle is
    seed/loss-independent; the grid script uses a glob pinned to
    ``seed0`` so all 12 tasks share the same oracle artifacts.
    """
    script_text = _RUN_GRID.read_text()
    assert "_main_oracle_${DATASET_NAME}_seed0" in script_text
