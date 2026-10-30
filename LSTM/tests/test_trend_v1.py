import pytest
import torch
from LSTM.src.model import MultiHeadFinancialLSTM
from LSTM.src.trend_model_v1 import TrendCentered15HeadLSTM, centered_view, VIEW_ID

def test_trend_transform_does_not_modify_input_and_zeroes_first_quarter():
    x = torch.randn(4,20,15)
    clone = x.clone()
    z = centered_view(x)
    assert torch.equal(x, clone)
    assert torch.allclose(z[:,0,:],torch.zeros_like(z[:,0,:]))
    assert torch.allclose(z[:,-1,:],x[:,-1,:]-x[:,0,:])

def test_trend_invariant_to_company_feature_level_offset():
    x = torch.randn(3,20,15)
    offsets = torch.randn(3,1,15)*20
    assert torch.allclose(centered_view(x),centered_view(x+offsets),atol=1e-5,rtol=1e-5)

def test_trend_keeps_15_independent_heads_and_same_parameters():
    base=MultiHeadFinancialLSTM()
    trend=TrendCentered15HeadLSTM()
    assert len(trend.heads)==len(base.heads)==15
    assert all(h.input_size==1 for h in trend.heads)
    assert len({id(h) for h in trend.heads})==15
    assert sum(p.numel() for p in base.parameters())==sum(p.numel() for p in trend.parameters())
    trend.load_state_dict(base.state_dict(), strict=True)
    trend.eval(); base.eval()
    x = torch.randn(2,20,15)
    with torch.no_grad():
        assert torch.allclose(trend(x),base(centered_view(x)), atol=1e-6)

def test_trend_forward_backward():
    model=TrendCentered15HeadLSTM()
    x=torch.randn(4,20,15,requires_grad=True)
    output=model(x)
    assert output.shape==(4,)
    output.square().mean().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

def test_view_shape_and_finiteness_validation():
    with pytest.raises(ValueError): centered_view(torch.zeros(4,15,20))
    x=torch.zeros(4,20,15);x[0,0,0]=float('nan')
    with pytest.raises(ValueError): centered_view(x)
