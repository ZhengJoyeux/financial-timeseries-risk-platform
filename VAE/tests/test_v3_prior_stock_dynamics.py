"""Synthetic CPU tests for V3 prior stock QoQ penalty; no real/split/test files read."""
import numpy as np
import torch
import pytest

from VAE.src.dataset import FinancialScaler
from VAE.src.timevae import TimeVAE
from VAE.src.prior_stock_dynamics import PriorStockQOQ, STOCK_INDICES


def _dataset():
    rng = np.random.default_rng(2026)
    n = 16
    labels = np.array([0] * 8 + [1] * 8)
    x = rng.uniform(200, 1000, size=(n, 20, 15))
    for i in range(n):
        growth = 0.004 if labels[i] == 0 else 0.025
        for t in range(20):
            x[i, t, list(STOCK_INDICES)] = (1000 + 10 * np.arange(8)) * np.exp(t * growth)
    scaler = FinancialScaler().fit(x)
    return x, labels, scaler


def test_prior_stock_qoq_targets_use_train_class_not_constants():
    x, y, scaler = _dataset()
    obj = PriorStockQOQ(x, y, scaler, torch.device('cpu'))
    assert obj.target.shape == (2, 7, 8)
    assert obj.floor.shape == (8,)
    assert torch.all(obj.floor > 0)
    assert obj.target[1, 3, 0].item() > obj.target[0, 3, 0].item()
    assert obj.state_dict()['stock_indices'] == list(STOCK_INDICES)


def test_prior_loss_backward_into_decoder_and_no_new_model_weights():
    torch.manual_seed(2026)
    x, y, scaler = _dataset()
    obj = PriorStockQOQ(x, y, scaler, torch.device('cpu'))
    model = TimeVAE()
    before_keys = set(model.state_dict())
    penalty = obj(model, samples_per_class=8)
    assert torch.isfinite(penalty)
    assert penalty.item() > 0
    penalty.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
    assert set(model.state_dict()) == before_keys
    with pytest.raises(ValueError):
        obj(model, samples_per_class=1)


def test_v3_decoder_state_and_d0_output_strict_compatible():
    torch.manual_seed(27)
    model = TimeVAE(season_gate_mode='none', seasonal_feature_indices=None)
    loaded = TimeVAE(**model.params)
    loaded.load_state_dict(model.state_dict(), strict=True)
    z = torch.randn(3, 8)
    labels = torch.tensor([0, 1, 0], dtype=torch.long)
    assert torch.allclose(model.decode(z, labels), loaded.decode(z, labels))


def test_prior_target_rejects_missing_classes_or_invalid_values():
    x, y, scaler = _dataset()
    with pytest.raises(ValueError):
        PriorStockQOQ(x, np.zeros_like(y), scaler, torch.device('cpu'))
    x[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        PriorStockQOQ(x, y, scaler, torch.device('cpu'))
