"""Tests for :class:`probly.method.head.MlpHead`."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from probly.method.head import MlpHead  # noqa: E402


def test_mlp_head_shape() -> None:
    """Forward pass produces logits of shape ``(batch, num_classes)``."""
    head = MlpHead(in_features=2048, num_classes=100, hidden=256, dropout_p=0.1)
    x = torch.randn(8, 2048)
    out = head(x)
    assert out.shape == (8, 100)


def test_mlp_head_structure() -> None:
    """Internal structure matches the locked architecture."""
    head = MlpHead(2048, 100)
    children = list(head.net.children())
    assert isinstance(children[0], torch.nn.Linear)
    assert isinstance(children[1], torch.nn.ReLU)
    assert isinstance(children[2], torch.nn.Dropout)
    assert isinstance(children[3], torch.nn.Linear)
    assert children[2].p == pytest.approx(0.1)
    assert children[0].in_features == 2048
    assert children[0].out_features == 256
    assert children[3].in_features == 256
    assert children[3].out_features == 100


def test_mlp_head_validation() -> None:
    """Invalid arguments raise :class:`ValueError`."""
    with pytest.raises(ValueError, match="in_features"):
        MlpHead(in_features=0, num_classes=10)
    with pytest.raises(ValueError, match="num_classes"):
        MlpHead(in_features=10, num_classes=0)
    with pytest.raises(ValueError, match="hidden"):
        MlpHead(in_features=10, num_classes=10, hidden=0)
    with pytest.raises(ValueError, match="dropout_p"):
        MlpHead(10, 10, dropout_p=1.0)
    with pytest.raises(ValueError, match="dropout_p"):
        MlpHead(10, 10, dropout_p=-0.1)
