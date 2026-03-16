"""Tests for scripts/train.py — tests freeze_backbone logic only, no real training."""

import pytest
from unittest.mock import MagicMock, patch


from scripts.train import freeze_backbone, BACKBONE_FREEZE_LAYERS


# ---------------------------------------------------------------------------
# freeze_backbone
# ---------------------------------------------------------------------------

def _make_mock_model(n_params: int = 20):
    """Build a mock YOLO model with named parameters spread across layers."""
    params = {}
    for i in range(n_params):
        layer = i // 2  # 2 params per layer
        p = MagicMock()
        p.requires_grad = True
        params[f"model.{layer}.weight_{i}"] = p

    model = MagicMock()
    model.model.named_parameters.return_value = list(params.items())
    return model, params


class TestFreezeBackbone:
    def test_first_n_layers_frozen(self):
        model, params = _make_mock_model(30)
        freeze_backbone(model, n_layers=5)
        for name, param in params.items():
            layer = int(name.split(".")[1])
            if layer < 5:
                assert param.requires_grad is False
            else:
                assert param.requires_grad is True

    def test_zero_layers_freezes_nothing(self):
        model, params = _make_mock_model(20)
        freeze_backbone(model, n_layers=0)
        for param in params.values():
            assert param.requires_grad is True

    def test_default_freeze_layers_constant(self):
        assert BACKBONE_FREEZE_LAYERS == 10

    def test_all_layers_frozen(self):
        model, params = _make_mock_model(20)
        freeze_backbone(model, n_layers=999)
        for param in params.values():
            assert param.requires_grad is False
