"""Pin per-epoch stdout logging in the three from-scratch wrappers.

Pre-fix, ``ensemble._train_member``, ``evidential_classification._train_evidential_model``,
and ``ddu._train_ddu_classifier`` all trained silently. SLURM logs
showed only ``method handle written...`` and gave no signal whether
training was actually happening, so the 1-epoch ensemble misconfig
took longer than necessary to diagnose.

Post-fix each loop emits one line per epoch in the format

    epoch <e>/<E>: train_loss=<float>

and ensemble's prefixes that with ``member <m>/<M>``. We capture
stdout via ``capsys`` and assert the line count + format.
"""

from __future__ import annotations

import re
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
_BATCH = 4


def _make_provider() -> FeatureProvider:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(_BATCH, _NUM_FEATURES, generator=g)
    y = torch.zeros(_BATCH, _NUM_CLASSES)
    y[:, 0] = 1.0
    return FeatureProvider(
        batches=[(x, y)],
        mode="linear_probe",
        feature_dim=_NUM_FEATURES,
        n_classes=_NUM_CLASSES,
        indices=np.arange(_BATCH, dtype=np.int64),
    )


class _LinearWithSoftplusAlpha(nn.Module):
    """Stand-in evidential head: emits strictly-positive Dirichlet alphas.

    We can't use a bare ``nn.Linear`` for evidential training because
    ``_evidential_ce_soft`` calls ``digamma(alpha)`` and requires
    ``alpha > 0``. The real wrapper composes a Softplus + ``+1`` shift;
    we replicate the same invariant here without going through probly's
    full ``evidential_classification`` transform (which expects a
    registered :class:`probly.predictor.LogitClassifier`).
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(_NUM_FEATURES, _NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(self.linear(x)) + 1.0


class _LogitsAndDensities(nn.Module):
    """Stand-in DDU predictor: returns ``(logits, densities)`` from one Linear.

    The real wrapper installs spectral-norm + LeakyReLU swaps on a
    LogitClassifier; for the logging contract we only need the
    forward signature ``(logits, per_class_log_density)`` and a real
    gradient path through ``logits``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(_NUM_FEATURES, _NUM_CLASSES)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.linear(x)
        # Mirror ``probly.method.ddu`` shape contract; densities are
        # discarded by ``_train_ddu_classifier``.
        densities = torch.zeros_like(logits)
        return logits, densities


_EPOCH_LINE = re.compile(
    r"epoch (\d+)/(\d+): train_loss=(-?\d+\.\d{4}) lr=(\d+\.\d{4})"
)
_MEMBER_EPOCH_LINE = re.compile(
    r"member (\d+)/(\d+) epoch (\d+)/(\d+): "
    r"train_loss=(-?\d+\.\d{4}) lr=(\d+\.\d{4})"
)


def test_ensemble_train_member_emits_one_line_per_epoch_with_member_prefix(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """ensemble: 'member 1/3 epoch 1/2: train_loss=X.XXXX' format."""
    provider = _make_provider()
    method_config: dict[str, Any] = {"epochs": 2, "lr": 1.0e-2, "weight_decay": 0.0}
    model = nn.Linear(_NUM_FEATURES, _NUM_CLASSES)
    ens_module._train_member(
        model,
        provider,
        method_config,
        seed=0,
        member_idx=0,
        n_members=3,
    )
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert len(lines) == 2, out
    parsed = [_MEMBER_EPOCH_LINE.match(ln) for ln in lines]
    assert all(p is not None for p in parsed), out
    # member is 1-indexed; epoch counter advances 1 -> 2.
    for i, m in enumerate(parsed):
        assert m is not None
        assert m.group(1) == "1"
        assert m.group(2) == "3"
        assert m.group(3) == str(i + 1)
        assert m.group(4) == "2"


def test_ensemble_fit_logs_for_every_member_and_every_epoch(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """fit() with n_members=2, epochs=2 emits 2*2=4 'member M/N epoch E/T' lines."""
    provider = _make_provider()
    method_config = {
        "name": "ensemble",
        "method_module": "experiments.epistemic_eval.methods.ensemble",
        "n_members": 2,
        "epochs": 2,
        "lr": 1.0e-2,
        "weight_decay": 0.0,
    }
    dataset_config = {"name": "synthetic", "extraction_mode": "linear_probe"}
    ens_module.fit(
        method_config=method_config,
        dataset_config=dataset_config,
        data_provider=provider,
        model_factory=lambda: nn.Linear(_NUM_FEATURES, _NUM_CLASSES),
        seed=0,
    )
    out = capsys.readouterr().out
    member_matches = [_MEMBER_EPOCH_LINE.match(ln) for ln in out.splitlines()]
    member_matches = [m for m in member_matches if m is not None]
    assert len(member_matches) == 4, out
    # Member 1's epochs print before member 2's (sequential training).
    seen_members = [int(m.group(1)) for m in member_matches]
    assert seen_members == [1, 1, 2, 2]


def test_evidential_train_emits_one_line_per_epoch(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """evidential: 'epoch 1/2: train_loss=X.XXXX' format (no acc, no member prefix)."""
    provider = _make_provider()
    model: nn.Module = _LinearWithSoftplusAlpha()
    method_config = {"epochs": 2, "lr": 1.0e-2, "weight_decay": 0.0}
    ev_module._train_evidential_model(model, provider, method_config, seed=0)
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert len(lines) == 2, out
    for i, ln in enumerate(lines):
        m = _EPOCH_LINE.match(ln)
        assert m is not None, ln
        assert m.group(1) == str(i + 1)
        assert m.group(2) == "2"
        # No 'member' prefix on evidential's lines.
        assert not ln.startswith("member ")


def test_ddu_train_emits_one_line_per_epoch(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """ddu: 'epoch 1/2: train_loss=X.XXXX' format (no acc, no member prefix)."""
    provider = _make_provider()
    model: nn.Module = _LogitsAndDensities()
    method_config = {"epochs": 2, "lr": 1.0e-2, "weight_decay": 0.0}
    ddu_module._train_ddu_classifier(model, provider, method_config, seed=0)
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert len(lines) == 2, out
    for i, ln in enumerate(lines):
        m = _EPOCH_LINE.match(ln)
        assert m is not None, ln
        assert m.group(1) == str(i + 1)
        assert m.group(2) == "2"
        assert not ln.startswith("member ")
