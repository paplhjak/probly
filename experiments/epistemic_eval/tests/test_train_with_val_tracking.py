"""Unit tests for ``methods._base.train_with_val_tracking``.

The helper centralises the per-epoch train/eval/best-state bookkeeping
that was duplicated in :func:`ensemble._train_member`,
:func:`evidential_classification._train_evidential_model`, and
:func:`ddu._train_ddu_classifier`. The two key behaviours under test
are:

1. With a ``val_provider``, the returned model loads the state_dict
   from the lowest-val-loss epoch -- not the final epoch. This is the
   safety net against the "diverges-after-best-epoch" trajectory that
   bit ensemble/evidential/ddu on CIFAR-10H pre-fix.
2. Without a ``val_provider``, the returned model retains its
   final-epoch state (back-compat with paths that have no val split).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from experiments.epistemic_eval.methods._base import (  # noqa: E402
    FeatureProvider,
    train_with_val_tracking,
)


def _provider(batches: list[tuple[torch.Tensor, torch.Tensor]]) -> FeatureProvider:
    n = sum(int(x.shape[0]) for x, _ in batches)
    return FeatureProvider(
        batches=batches,
        mode="full_network",
        feature_dim=None,
        n_classes=int(batches[0][1].shape[1]) if batches else 1,
        indices=np.arange(n, dtype=np.int64),
    )


def _mse(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Plain MSE loss closure for the helper."""
    return torch.nn.functional.mse_loss(model(x), y)


class _Linear(torch.nn.Module):
    """One-layer linear model with a bias param we can probe."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(2, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def test_best_val_state_is_restored_after_divergence() -> None:
    """When val_loss rises after a minimum, the helper rolls back to the best epoch.

    Construct a setup where the OPTIMUM is reached early, then SGD with
    a deliberately-too-large learning rate overshoots. The val_loss
    trajectory must be non-monotonic; on exit, the model's parameters
    must match the snapshot from the best-val-loss epoch and NOT the
    final-epoch state.
    """
    torch.manual_seed(0)
    model = _Linear()
    # Closed-form best fit for y = 2x_0 + 3x_1 is fc.weight=[2,3], bias=0.
    train_x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]])
    train_y = torch.tensor([[2.0], [3.0], [5.0], [1.0]])
    val_x = torch.tensor([[0.5, 0.5], [-0.5, 1.0]])
    val_y = torch.tensor([[2.5], [2.0]])

    train_provider = _provider([(train_x, train_y)])
    val_provider = _provider([(val_x, val_y)])

    # Lr = 1.0 with full-batch SGD on this 4-point dataset overshoots
    # the optimum after ~4-5 steps and oscillates outwards. The val
    # loss bottoms out a few epochs in and then grows.
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _e: 1.0)

    val_losses_seen: list[float] = []

    def _capturing_mse(
        m: torch.nn.Module, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        out = _mse(m, x, y)
        if not m.training:
            val_losses_seen.append(float(out.item()))
        return out

    trained = train_with_val_tracking(
        model,
        train_provider,
        val_provider,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=20,
        compute_loss=_capturing_mse,
    )

    # Sanity: val loss must be non-monotonic (otherwise the test
    # doesn't exercise the rollback path).
    assert len(val_losses_seen) == 20
    best_epoch = int(np.argmin(val_losses_seen))
    final_epoch = 19
    assert val_losses_seen[final_epoch] > val_losses_seen[best_epoch], (
        f"test setup did not produce divergence; saw monotonic val_losses "
        f"{val_losses_seen[:10]}... -- pick a larger lr."
    )

    # Compute val_loss on the *returned* model: it must equal the best
    # observed val_loss (the helper rolled back the state_dict).
    trained.eval()
    with torch.no_grad():
        returned_val_loss = float(_mse(trained, val_x, val_y).item())
    assert returned_val_loss == pytest.approx(val_losses_seen[best_epoch], abs=1e-6), (
        f"returned model val_loss={returned_val_loss:.6f} does not match the "
        f"best epoch's val_loss={val_losses_seen[best_epoch]:.6f} (best epoch "
        f"was {best_epoch}); helper failed to roll back the state_dict."
    )
    # Conversely, it must NOT match the final epoch's val_loss (which
    # is strictly worse by construction).
    assert returned_val_loss < val_losses_seen[final_epoch], (
        f"returned model val_loss={returned_val_loss:.6f} matches the final "
        f"epoch's val_loss={val_losses_seen[final_epoch]:.6f}; helper kept "
        f"the diverged final-epoch state instead of the best."
    )


def test_no_val_provider_returns_final_epoch_state() -> None:
    """Without a val_provider, the helper retains the final-epoch state.

    Back-compat guarantee for paths that don't have a held-out split.
    No rollback happens, so the returned model's params equal whatever
    SGD ended on at the last epoch.
    """
    torch.manual_seed(0)
    model = _Linear()
    train_x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    train_y = torch.tensor([[2.0], [3.0]])
    train_provider = _provider([(train_x, train_y)])

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _e: 1.0)

    # Snapshot final-epoch state by running the loop manually for a
    # parallel "control" model and comparing.
    control = _Linear()
    control.load_state_dict(model.state_dict())
    control_opt = torch.optim.SGD(control.parameters(), lr=0.1)
    control.train()
    for _ in range(5):
        for x, y in train_provider:
            loss = _mse(control, x, y)
            control_opt.zero_grad()
            loss.backward()
            control_opt.step()

    trained = train_with_val_tracking(
        model,
        train_provider,
        val_provider=None,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=5,
        compute_loss=_mse,
    )

    for (k1, v1), (k2, v2) in zip(
        trained.state_dict().items(), control.state_dict().items(), strict=True
    ):
        assert k1 == k2
        assert torch.allclose(v1, v2, atol=1e-6), (
            f"final-epoch state for {k1} differs from manual control loop"
        )
