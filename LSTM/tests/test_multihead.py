import numpy as np
import pytest
import torch
from LSTM.src.model import MultiHeadFinancialLSTM
from LSTM.src.metrics import binary_metrics
from LSTM.scripts.train import build_sampler


def test_exactly_15_independent_heads_and_backprop():
    torch.manual_seed(7)
    net=MultiHeadFinancialLSTM(n_features=15,seq_len=20,hidden_size=8)
    assert len(net.heads)==15
    assert len({id(h.weight_ih_l0) for h in net.heads})==15
    x=torch.randn(4,20,15)
    y=torch.tensor([0.,1.,0.,1.])
    loss=torch.nn.functional.binary_cross_entropy_with_logits(net(x),y)
    loss.backward()
    assert all(h.weight_ih_l0.grad is not None and h.weight_ih_l0.grad.abs().sum()>0 for h in net.heads)


def test_each_head_is_univariate_and_variable_specific():
    net=MultiHeadFinancialLSTM(hidden_size=6).eval()
    x=torch.randn(2,20,15)
    seen=[]
    def hook(idx):
        def func(module, args):
            seen.append((idx,args[0].detach().clone()))
        return func
    hooks=[head.register_forward_pre_hook(hook(i)) for i,head in enumerate(net.heads)]
    with torch.no_grad():
        _=net(x)
    for hook_item in hooks:hook_item.remove()
    assert len(seen)==15
    for idx,inp in seen:
        assert inp.shape==(2,20,1)
        assert torch.allclose(inp,x[:,:,idx:idx+1])


def test_wrong_axis_rejected():
    with pytest.raises(ValueError):
        MultiHeadFinancialLSTM()(torch.randn(2,15,20))


def test_sampler_strata_for_generated_only_and_augmented():
    y=np.asarray([0]*38+[1]*10+[0]*100+[1]*100)
    src=np.asarray(['real']*48+['generated']*200)
    sampler=build_sampler(y,src,'real_plus_s1',0.5,2026)
    assert len(list(sampler))>=248
    assert len(list(build_sampler(y[48:],src[48:],'generated_only',0.5,2026)))>=200


def test_binary_metrics_class_imbalance():
    r=binary_metrics([0,0,0,1],[.1,.3,.9,.8])
    assert r['n_risk']==1 and r['tp']==1 and r['fp']==1
    assert r['pr_auc'] is not None
