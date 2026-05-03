"""Shallow MLP head for linear-probe UQ on cached features."""

from __future__ import annotations

import torch
from torch import nn


class MlpHead(nn.Module):
    """Two-layer MLP head with ReLU and dropout in between.

    Architecture (locked in ``decisions.md`` "APPA-REAL backbone strategy"):

        Linear(in_features, hidden) -> ReLU -> Dropout(p) -> Linear(hidden, num_classes)

    The Dropout module is part of the architecture: for MC-Dropout we
    activate it at inference via ``model.train()``; for Deep Ensembles
    we leave it at ``model.eval()`` so each member is deterministic
    post-training (see ``decisions.md`` "APPA-REAL backbone strategy").

    Args:
        in_features: Input feature dimensionality (e.g. 2048 for ResNet-101).
        num_classes: Output dimensionality (K, e.g. ``len(age_support)``).
        hidden: Hidden-layer width. Default 256 per ``decisions.md``.
        dropout_p: Dropout probability. Default 0.1 per ``decisions.md``.

    Raises:
        ValueError: If ``in_features``, ``num_classes`` or ``hidden`` is
            non-positive, or if ``dropout_p`` is outside ``[0.0, 1.0)``.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden: int = 256,
        dropout_p: float = 0.1,
    ) -> None:
        """Build the head; see class docstring for arg semantics."""
        super().__init__()
        if in_features <= 0:
            msg = f"in_features must be > 0, got {in_features}"
            raise ValueError(msg)
        if num_classes <= 0:
            msg = f"num_classes must be > 0, got {num_classes}"
            raise ValueError(msg)
        if hidden <= 0:
            msg = f"hidden must be > 0, got {hidden}"
            raise ValueError(msg)
        if not 0.0 <= dropout_p < 1.0:
            msg = f"dropout_p must be in [0.0, 1.0), got {dropout_p}"
            raise ValueError(msg)
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and return raw class logits.

        Args:
            x: Input feature tensor of shape ``(batch, in_features)``.

        Returns:
            Logits of shape ``(batch, num_classes)``.
        """
        return self.net(x)
