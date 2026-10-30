"""Pure engineering checks of finance ratio construction and V2 head contract."""
import numpy as np
import pytest
import torch
from LSTM.src.model import MultiHeadFinancialLSTM
from LSTM.src.trend_model_v2 import (
    TrendLeverage15HeadLSTM, leverage_numpy, fit_leverage_stats, LEVERAGE_NAMES,
)


def synthetic_input(n=48):
    # Build physically meaningful data and normalize via exact asinh inverse contract
    rng=np.random.default_rng(7)
    raw=rng.uniform(20,150, size=(n,20,15))
    raw[:,:,0] = rng.uniform(100,120,size=(n,20))
    raw[:,:,1] = raw[:,:,0]*np.linspace(.4,.65,20)[None,:]*rng.uniform(.9,1.1,size=(n,1))
    state={'scale':[10.0]*15,'mean':[0.0]*15,'std':[1.0]*15}
    x=np.arcsinh(raw/10).astype(np.float32)
    return x,raw,state


def test_raw_ratio_and_last4_minus_first4():
    x,raw,state=synthetic_input()
    got=leverage_numpy(x,state)
    ratio=raw[:,:,1]/raw[:,:,0]
    expected=np.column_stack((ratio.mean(axis=1),ratio[:,-4:].mean(axis=1)-ratio[:,:4].mean(axis=1)))
    assert np.allclose(got,expected,atol=2e-7)


def test_feature_scaler_is_real_train_only_and_48_firms():
    x,raw,state=synthetic_input()
    stats=fit_leverage_stats(x,state)
    assert stats['feature_names']==list(LEVERAGE_NAMES)
    expected=leverage_numpy(x,state)
    assert np.allclose(stats['mean'],expected.mean(axis=0))
    with pytest.raises(RuntimeError):fit_leverage_stats(x[:30],state)


def test_fifteen_independent_heads_and_gradient():
    x,raw,state=synthetic_input()
    stats=fit_leverage_stats(x,state)
    torch.manual_seed(17)
    model=TrendLeverage15HeadLSTM(scaler_state=state,leverage_stats=stats)
    assert len(model.heads)==15
    assert all(head.input_size==1 for head in model.heads)
    assert len({id(head) for head in model.heads})==15
    assert sum(p.numel() for p in model.parameters())==24625
    xb=torch.as_tensor(x[:8],dtype=torch.float32)
    logits=model(xb)
    assert logits.shape==(8,) and torch.isfinite(logits).all()
    feats=model.compute_leverage(xb).detach().cpu().numpy()
    expected=(leverage_numpy(x[:8],state)-np.asarray(stats['mean']))/np.asarray(stats['std'])
    assert np.allclose(feats,expected,atol=2e-5)
    loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.tensor([0.,1.]*4))
    loss.backward()
    assert model.leverage_adapter.weight.grad is not None
    assert torch.isfinite(model.leverage_adapter.weight.grad).all()
    assert any(h.weight_ih_l0.grad is not None for h in model.heads)


def test_disabled_adapter_equals_trend_v1_view():
    x,raw,state=synthetic_input()
    stats=fit_leverage_stats(x,state)
    torch.manual_seed(13)
    baseline=MultiHeadFinancialLSTM()
    torch.manual_seed(13)
    v2=TrendLeverage15HeadLSTM(scaler_state=state,leverage_stats=stats,leverage_enabled=False)
    baseline.eval();v2.eval()
    with torch.no_grad():
        a=baseline(torch.as_tensor(x[:4])-torch.as_tensor(x[:4,:1,:]))
        b=v2(torch.as_tensor(x[:4]))
    assert torch.allclose(a,b,atol=1e-7)


def test_reject_wrong_axis_or_bad_assets():
    x,raw,state=synthetic_input()
    stats=fit_leverage_stats(x,state)
    model=TrendLeverage15HeadLSTM(scaler_state=state,leverage_stats=stats)
    with pytest.raises(ValueError):model(torch.zeros(3,19,15))
    with pytest.raises(ValueError):model(torch.ones(3,20,15)*float('nan'))
    with pytest.raises(RuntimeError):model(torch.zeros(3,20,15)) # total_assets=0
