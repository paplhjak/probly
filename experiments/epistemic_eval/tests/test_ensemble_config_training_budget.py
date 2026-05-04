"""Pin: ensemble.yaml carries an explicit training budget.

Pre-fix, ``configs/methods/ensemble.yaml`` had no ``epochs`` field,
so :func:`experiments.epistemic_eval.methods.ensemble._train_member`
fell back to its default of ``epochs = 1`` and trained each member
for a single pass over the train provider in the production grid.
The resulting ensemble had ~45% top-1 error on CIFAR-10 (vs. ~4.5%
for the basecls), masquerading as a "DDU underperforms ensembles"
result that was actually "ensembles barely trained."

This test reads the production ensemble config, drives
``ensemble.fit()`` with a monkeypatched ``_train_member`` recorder
that captures the ``method_config`` it was called with, and asserts
the recorder saw ``epochs == 200``, ``lr == 1e-3``, and
``weight_decay == 5e-4`` for each of ``n_members == 5`` calls.

The test does NOT exercise real training -- it pins the config-to-
wrapper plumbing so the next dryrun → production transition can't
silently regress to the 1-epoch default again.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import yaml  # noqa: E402

from experiments.epistemic_eval.methods import ensemble as ens_module  # noqa: E402
from experiments.epistemic_eval.methods._base import FeatureProvider  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ENSEMBLE_CFG = (
    _REPO_ROOT
    / "experiments"
    / "epistemic_eval"
    / "configs"
    / "methods"
    / "ensemble.yaml"
)


def _make_provider() -> FeatureProvider:
    """Tiny synthetic provider; we never actually train."""
    x = torch.zeros(2, 4, dtype=torch.float32)
    y = torch.zeros(2, 3, dtype=torch.float32)
    y[:, 0] = 1.0
    return FeatureProvider(
        batches=[(x, y)],
        mode="linear_probe",
        feature_dim=4,
        n_classes=3,
        indices=np.arange(2, dtype=np.int64),
    )


def _factory() -> torch.nn.Module:
    return torch.nn.Linear(4, 3)


def test_production_ensemble_yaml_declares_locked_sgd_recipe() -> None:
    """The committed config carries the keys ``_train_member`` reads.

    The locked recipe is SGD + cosine, mirroring basecls. Pre-fix
    ensemble.yaml had no training section at all (1-epoch default);
    a later iteration declared an AdamW recipe that diverged from
    basecls. This test pins the unified SGD recipe so a future
    config edit can't silently drift back.
    """
    cfg = yaml.safe_load(_ENSEMBLE_CFG.read_text())
    # The wrapper reads these five keys in _train_member.
    for key in ("epochs", "lr", "momentum", "nesterov", "weight_decay"):
        assert key in cfg, (key, cfg)
    # Pin the production budget; matches basecls/evidential/ddu.
    assert int(cfg["epochs"]) == 200
    assert float(cfg["lr"]) == pytest.approx(0.1)
    assert float(cfg["momentum"]) == pytest.approx(0.9)
    assert bool(cfg["nesterov"]) is False
    assert float(cfg["weight_decay"]) == pytest.approx(5.0e-4)
    # ``schedule`` is documentation-only (the wrapper hardcodes
    # CosineAnnealingLR) but we still pin it so the recipe is
    # self-documenting.
    assert cfg.get("schedule") == "cosine"


def test_fit_threads_config_to_train_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``fit()`` must call ``_train_member`` once per member with the configured budget.

    Monkeypatches ``_train_member`` with a recorder. Pre-fix, the
    recorder would have seen ``epochs=1`` because the config field
    didn't exist; post-fix it sees the production value.
    """
    cfg = yaml.safe_load(_ENSEMBLE_CFG.read_text())
    n_members = int(cfg["n_members"])

    received: list[dict[str, Any]] = []

    def _recorder(
        model: torch.nn.Module,
        data_provider: FeatureProvider,  # noqa: ARG001
        method_config: dict[str, Any],
        seed: int,
        *,
        member_idx: int = 0,
        n_members: int = 1,
    ) -> torch.nn.Module:
        received.append(
            {
                "epochs": int(method_config.get("epochs", 1)),
                "lr": float(method_config.get("lr", 1.0e-3)),
                "momentum": float(method_config.get("momentum", 0.9)),
                "nesterov": bool(method_config.get("nesterov", False)),
                "weight_decay": float(method_config.get("weight_decay", 0.0)),
                "seed": int(seed),
                "member_idx": int(member_idx),
                "n_members": int(n_members),
            }
        )
        return model

    monkeypatch.setattr(ens_module, "_train_member", _recorder)

    handle = ens_module.fit(
        method_config=cfg,
        dataset_config={"name": "synthetic", "extraction_mode": "linear_probe"},
        data_provider=_make_provider(),
        model_factory=_factory,
        seed=0,
    )
    assert len(handle.state_dicts) == n_members
    assert len(received) == n_members
    for i, call in enumerate(received):
        assert call["epochs"] == 200, call
        assert call["lr"] == pytest.approx(0.1), call
        assert call["momentum"] == pytest.approx(0.9), call
        assert call["nesterov"] is False, call
        assert call["weight_decay"] == pytest.approx(5.0e-4), call
        assert call["member_idx"] == i
        assert call["n_members"] == n_members
