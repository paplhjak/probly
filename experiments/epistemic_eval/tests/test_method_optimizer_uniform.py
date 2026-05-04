"""Pin: every from-scratch wrapper uses SGD-cosine, not Adam/AdamW.

The four UQ methods in the paper compare four UQ *families*; the
training recipe is meant to be a held-fixed nuisance variable, not
a confounder. Pre-fix the wrappers used ``torch.optim.AdamW`` while
basecls and mc_dropout used SGD with a cosine schedule, making the
head-to-head comparison recipe-confounded.

Each test below monkeypatches ``torch.optim.SGD`` and
``torch.optim.AdamW`` with recorders, runs a 1-epoch fit on a tiny
synthetic provider, and asserts:

* ``torch.optim.SGD`` was constructed with the locked-recipe kwargs.
* ``torch.optim.AdamW`` was NOT constructed.
* ``torch.optim.lr_scheduler.CosineAnnealingLR`` was constructed with
  ``T_max == epochs``.

Together these pin the cross-method recipe at the call-site level
(stronger than reading hyperparameter values out of a config).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from experiments.epistemic_eval.methods import ddu as ddu_module  # noqa: E402
from experiments.epistemic_eval.methods import (  # noqa: E402
    evidential_classification as ev_module,
)
from experiments.epistemic_eval.methods import ensemble as ens_module  # noqa: E402
from experiments.epistemic_eval.methods._base import FeatureProvider  # noqa: E402


_NUM_FEATURES = 8
_NUM_CLASSES = 3


def _make_provider() -> FeatureProvider:
    """Tiny one-batch provider; we don't care about loss values here."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(4, _NUM_FEATURES, generator=g)
    y = torch.zeros(4, _NUM_CLASSES)
    y[:, 0] = 1.0
    return FeatureProvider(
        batches=[(x, y)],
        mode="linear_probe",
        feature_dim=_NUM_FEATURES,
        n_classes=_NUM_CLASSES,
        indices=np.arange(4, dtype=np.int64),
    )


class _LinearWithSoftplusAlpha(nn.Module):
    """Stand-in evidential head: emits strictly-positive Dirichlet alphas.

    Same shim as in :mod:`test_method_per_epoch_logging`; the real
    wrapper composes Softplus + ``+1`` via probly's
    ``evidential_classification`` transform, which expects a
    registered ``LogitClassifier``. We replicate the post-condition
    (``alpha > 1``) without going through the registry.
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(_NUM_FEATURES, _NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(self.linear(x)) + 1.0


class _LogitsAndDensities(nn.Module):
    """Stand-in DDU predictor: returns ``(logits, densities)`` from one Linear."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(_NUM_FEATURES, _NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.linear(x)
        densities = torch.zeros_like(logits)
        return logits, densities


class _OptimizerSpy:
    """Records ``torch.optim.SGD`` / ``torch.optim.AdamW`` constructions.

    Wraps the real class so the wrapper code still works (param
    groups, gradients, etc.). Recording is by ``__init__`` interception:
    the spy is callable, captures kwargs, and forwards to the real
    constructor.
    """

    def __init__(self, real: type[torch.optim.Optimizer]) -> None:
        self._real = real
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, params: Any, **kwargs: Any
    ) -> torch.optim.Optimizer:
        # ``params`` may be a generator; materialise so we can both
        # record the count and pass it through.
        params_list = list(params)
        self.calls.append({"kwargs": kwargs, "n_params": len(params_list)})
        return self._real(params_list, **kwargs)


class _SchedulerSpy:
    """Records ``CosineAnnealingLR`` constructions and forwards to the real class.

    Capture the real class at construction time, before the
    monkeypatch swaps it out -- otherwise ``__call__`` would resolve
    ``torch.optim.lr_scheduler.CosineAnnealingLR`` to this spy itself
    and recurse forever.
    """

    def __init__(
        self,
        real: type[torch.optim.lr_scheduler.LRScheduler],
    ) -> None:
        self._real = real
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, optimizer: torch.optim.Optimizer, **kwargs: Any
    ) -> torch.optim.lr_scheduler.LRScheduler:
        self.calls.append({"kwargs": kwargs})
        return self._real(optimizer, **kwargs)


def _install_spies(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_OptimizerSpy, _OptimizerSpy, _SchedulerSpy]:
    sgd_spy = _OptimizerSpy(torch.optim.SGD)
    adamw_spy = _OptimizerSpy(torch.optim.AdamW)
    cosine_spy = _SchedulerSpy(torch.optim.lr_scheduler.CosineAnnealingLR)
    monkeypatch.setattr(torch.optim, "SGD", sgd_spy)
    monkeypatch.setattr(torch.optim, "AdamW", adamw_spy)
    monkeypatch.setattr(
        torch.optim.lr_scheduler, "CosineAnnealingLR", cosine_spy
    )
    return sgd_spy, adamw_spy, cosine_spy


_LOCKED_KWARGS = {
    "lr": 0.1,
    "momentum": 0.9,
    "nesterov": False,
    "weight_decay": 5.0e-4,
}


def _assert_sgd_locked(spy: _OptimizerSpy, n_calls: int) -> None:
    """Common assertions on the SGD spy across the three wrappers."""
    assert len(spy.calls) == n_calls, spy.calls
    for call in spy.calls:
        kwargs = call["kwargs"]
        assert kwargs["lr"] == pytest.approx(_LOCKED_KWARGS["lr"]), kwargs
        assert kwargs["momentum"] == pytest.approx(
            _LOCKED_KWARGS["momentum"]
        ), kwargs
        assert kwargs["nesterov"] is _LOCKED_KWARGS["nesterov"], kwargs
        assert kwargs["weight_decay"] == pytest.approx(
            _LOCKED_KWARGS["weight_decay"]
        ), kwargs


def test_ensemble_uses_sgd_cosine_per_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ensemble.fit() must construct one SGD + one CosineAnnealingLR per member.

    With ``n_members=3, epochs=2``: SGD constructed 3 times (one per
    member), CosineAnnealingLR also 3 times with ``T_max=2``, AdamW
    never constructed.
    """
    sgd_spy, adamw_spy, cosine_spy = _install_spies(monkeypatch)

    method_config: dict[str, Any] = {
        "name": "ensemble",
        "method_module": "experiments.epistemic_eval.methods.ensemble",
        "n_members": 3,
        "epochs": 2,
        **_LOCKED_KWARGS,
    }
    dataset_config = {
        "name": "synthetic",
        "extraction_mode": "linear_probe",
    }
    ens_module.fit(
        method_config=method_config,
        dataset_config=dataset_config,
        data_provider=_make_provider(),
        model_factory=lambda: nn.Linear(_NUM_FEATURES, _NUM_CLASSES),
        seed=0,
    )
    _assert_sgd_locked(sgd_spy, n_calls=3)
    assert adamw_spy.calls == [], adamw_spy.calls
    assert len(cosine_spy.calls) == 3
    for c in cosine_spy.calls:
        assert c["kwargs"]["T_max"] == 2, c


def test_evidential_uses_sgd_cosine(monkeypatch: pytest.MonkeyPatch) -> None:
    """evidential's training loop must construct SGD + CosineAnnealingLR, not AdamW."""
    sgd_spy, adamw_spy, cosine_spy = _install_spies(monkeypatch)

    model: nn.Module = _LinearWithSoftplusAlpha()
    method_config: dict[str, Any] = {
        "epochs": 2,
        **_LOCKED_KWARGS,
    }
    ev_module._train_evidential_model(
        model, _make_provider(), method_config, seed=0
    )
    _assert_sgd_locked(sgd_spy, n_calls=1)
    assert adamw_spy.calls == [], adamw_spy.calls
    assert len(cosine_spy.calls) == 1
    assert cosine_spy.calls[0]["kwargs"]["T_max"] == 2


def test_ddu_uses_sgd_cosine(monkeypatch: pytest.MonkeyPatch) -> None:
    """DDU's Phase A trainer must construct SGD + CosineAnnealingLR, not AdamW."""
    sgd_spy, adamw_spy, cosine_spy = _install_spies(monkeypatch)

    model: nn.Module = _LogitsAndDensities()
    method_config: dict[str, Any] = {
        "epochs": 2,
        **_LOCKED_KWARGS,
    }
    ddu_module._train_ddu_classifier(
        model, _make_provider(), method_config, seed=0
    )
    _assert_sgd_locked(sgd_spy, n_calls=1)
    assert adamw_spy.calls == [], adamw_spy.calls
    assert len(cosine_spy.calls) == 1
    assert cosine_spy.calls[0]["kwargs"]["T_max"] == 2


def test_three_method_configs_share_the_same_optimizer_recipe() -> None:
    """ensemble.yaml, evidential.yaml, ddu.yaml all declare the same SGD recipe.

    Pinning at the config level guards against a future drift where
    one method's lr or momentum gets nudged off the locked value
    without the others. The four basecls/mc_dropout values live in
    cifar10h.yaml and are pinned indirectly: this test asserts only
    the three from-scratch method configs.
    """
    import yaml
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    cfg_dir = repo / "experiments" / "epistemic_eval" / "configs" / "methods"
    cfgs = {
        name: yaml.safe_load((cfg_dir / f"{name}.yaml").read_text())
        for name in ("ensemble", "evidential", "ddu")
    }
    for name, cfg in cfgs.items():
        for key in ("epochs", "lr", "momentum", "nesterov", "weight_decay"):
            assert key in cfg, (name, key, cfg)
        assert int(cfg["epochs"]) == 200, name
        assert float(cfg["lr"]) == pytest.approx(0.1), name
        assert float(cfg["momentum"]) == pytest.approx(0.9), name
        assert bool(cfg["nesterov"]) is False, name
        assert float(cfg["weight_decay"]) == pytest.approx(5.0e-4), name
        assert cfg.get("schedule") == "cosine", name
