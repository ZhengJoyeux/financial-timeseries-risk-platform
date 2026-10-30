from pathlib import Path
import numpy as np
from VAE.src.dataset import read_financial,FinancialScaler
from VAE.src.finance_rules import violation_rates,repair_financial
from VAE.src.timevae import TimeVAE
from VAE.src.schema import EXCEL_COLUMNS


def test_dataset_and_stratified_split():
    root=Path(__file__).resolve().parents[2]
    a,x,y=read_financial(root/'data/raw/financial_80.xlsx')
    assert len(a)==80 and x.shape==(80,20,15)
    from VAE.src.dataset import to_frame
    d=to_frame(a,x,y)
    assert d.shape==(400,63)
    assert (d.groupby('企业编号').size()==5).all()
    assert (d.groupby('企业编号')['年份'].apply(list).apply(lambda z: z == [1,2,3,4,5])).all()
    assert len(EXCEL_COLUMNS)==63
    assert int((y=='有风险').sum())==16
    ids=[]
    for split,count,risk in [('train',48,10),('validation',16,3),('test',16,3)]:
        k,z,l=read_financial(root/f'data/splits/{split}.xlsx')
        assert len(k)==count and sum(l=='有风险')==risk
        ids += k.tolist()
    assert sorted(ids)==sorted(a.tolist())
    assert max(violation_rates(x).values())<.01


def test_timevae_forward_backward_and_sampling():
    import torch
    model=TimeVAE()
    x=torch.randn(3,20,15)
    y=torch.tensor([0,1,0])
    pred,mu,lv=model(x,y)
    assert pred.shape==x.shape and mu.shape==(3,8)
    loss,components=model.loss(x,pred,mu,lv)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(np.isfinite(v) for v in components.values())
    assert model.sample(7,torch.device('cpu'),condition=1).shape==(7,20,15)


def test_normalizer_and_repair():
    _,x,_=read_financial(Path(__file__).resolve().parents[2]/'data/splits/train.xlsx')
    scaler=FinancialScaler().fit(x)
    transformed=scaler.transform(x)
    restored=scaler.inverse_transform(transformed)
    assert np.allclose(x,restored,rtol=1e-5,atol=.1)
    invalid=x[:2].copy();invalid[:,:,10]=-99
    repaired=repair_financial(invalid)
    assert max(violation_rates(repaired).values())<.01


def test_one_model_uses_conditional_input():
    import torch
    torch.manual_seed(2026)
    model=TimeVAE()
    z=torch.randn(4,8)
    healthy=model.decode(z,torch.zeros(4,dtype=torch.long))
    risk=model.decode(z,torch.ones(4,dtype=torch.long))
    assert not torch.allclose(healthy,risk), '条件编码应影响同一潜变量对应的输出'
    try:
        model.decode(z,None)
    except ValueError:
        pass
    else:
        raise AssertionError('条件模型不得在没有标签的情况下默默生成。')


def test_evaluation_generates_reports(tmp_path):
    from VAE.src.metrics import metric_report
    import json
    root=Path(__file__).resolve().parents[2]
    _,x,y=read_financial(root/'data/splits/train.xlsx')
    _,vx,vy=read_financial(root/'data/splits/validation.xlsx')
    train=x[y=='有风险'];valid=vx[vy=='有风险']
    scaler=FinancialScaler().fit(x)
    report=metric_report(train,train[:4].copy(),valid,scaler,tmp_path,'risk',raw_generated=train[:4].copy())
    assert 0<=report['coverage_train']<=1
    assert report['postprocessing_changed_point_fraction']==0
    assert (tmp_path/'risk_summary.json').exists()
    assert (tmp_path/'risk_per_variable.csv').exists()
    assert '财务' not in str(report['rbf_mmd2'])


def test_seasonality_mask_only_flow_variables():
    import torch
    model = TimeVAE(seasonal_feature_indices=[2,3,10,11,12,13,14],use_base=False,use_trend=False)
    mask = model.season_feature_mask.reshape(-1)
    assert torch.equal(torch.where(mask == 0)[0],torch.tensor([0,1,4,5,6,7,8,9]))
    assert torch.equal(torch.where(mask == 1)[0],torch.tensor([2,3,10,11,12,13,14]))
    z = torch.randn(6,8)
    y = torch.tensor([0,1,0,1,0,1])
    output = model.decode(z,y)
    assert output.shape == (6,20,15)
    assert torch.count_nonzero(output[:,:,[0,1,4,5,6,7,8,9]]) == 0
    assert torch.any(output[:,:,[2,3,10,11,12,13,14]] != 0)
    assert 'season_feature_mask' not in model.state_dict()


def test_legacy_checkpoint_strict_state_compatibility():
    import torch
    torch.manual_seed(2026)
    old = TimeVAE()
    z=torch.randn(3,8)
    y=torch.tensor([0,1,1])
    output=old.decode(z,y)
    old_state=old.state_dict()
    old_params={k:v for k,v in old.params.items() if k!='seasonal_feature_indices'}
    restored=TimeVAE(**old_params)
    restored.load_state_dict(old_state,strict=True)
    assert torch.allclose(output,restored.decode(z,y),atol=1e-6)
    assert torch.all(restored.season_feature_mask==1)
    # New-model parameter mask is intentionally outside persistent model weights.
    new=TimeVAE(seasonal_feature_indices=[2,3,10,11,12,13,14])
    new.load_state_dict(old_state,strict=True)
    assert torch.any(new.decode(z,y) != restored.decode(z,y))


def test_reject_invalid_seasonality_indices():
    import pytest
    for invalid in [[2,2],[15],[-1],['2'],[2.0]]:
        with pytest.raises(ValueError):
            TimeVAE(seasonal_feature_indices=invalid)


def test_gpu1_only_device_guard(monkeypatch):
    from types import SimpleNamespace
    from VAE.scripts.train import device_from
    import torch
    import subprocess
    import pytest
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout='GPU-test-uuid\n'))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-test-uuid')
    assert str(device_from('cuda:0')) == 'cuda:0'
    with pytest.raises(RuntimeError):
        device_from('cpu')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    with pytest.raises(RuntimeError):
        device_from('cuda:0')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-test-uuid')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError):
        device_from('cuda:0')


def test_soft_stock_gate_values_and_gradient():
    import torch
    model=TimeVAE(season_gate_mode="learned_stock",stock_season_gate_init=.25,
                  use_base=False,use_trend=False)
    gates=model.seasonal_gates()
    assert gates.shape==(15,)
    assert torch.allclose(gates[[0,1,4,5,6,7,8,9]],torch.full((8,),.25),atol=1e-6)
    assert torch.all(gates[[2,3,10,11,12,13,14]]==1)
    z=torch.randn(5,8)
    y=torch.tensor([0,1,0,1,0])
    output=model.decode(z,y)
    loss=output.square().mean()
    loss.backward()
    assert model.stock_season_gate_logits.grad is not None
    assert torch.isfinite(model.stock_season_gate_logits.grad).all()
    assert model.stock_season_gate_logits.grad.abs().sum()>0


def test_soft_stock_gate_state_roundtrip_and_old_strict_loading():
    import torch
    torch.manual_seed(2026)
    v2=TimeVAE(season_gate_mode="learned_stock")
    recovered=TimeVAE(**v2.params)
    recovered.load_state_dict(v2.state_dict(),strict=True)
    z=torch.randn(3,8); y=torch.tensor([0,1,1])
    assert torch.allclose(v2.decode(z,y),recovered.decode(z,y))
    assert 'stock_season_gate_logits' in v2.state_dict()
    d0=TimeVAE()
    v1=TimeVAE(seasonal_feature_indices=[2,3,10,11,12,13,14])
    d0_model=TimeVAE(**d0.params)
    v1_model=TimeVAE(**v1.params)
    d0_model.load_state_dict(d0.state_dict(),strict=True)
    v1_model.load_state_dict(v1.state_dict(),strict=True)
    assert 'stock_season_gate_logits' not in d0.state_dict()
    assert 'stock_season_gate_logits' not in v1.state_dict()


def test_invalid_soft_gate_configuration_rejected():
    import pytest
    with pytest.raises(ValueError):
        TimeVAE(season_gate_mode='typo')
    with pytest.raises(ValueError):
        TimeVAE(season_gate_mode='learned_stock',use_seasonality=False)
    with pytest.raises(ValueError):
        TimeVAE(season_gate_mode='learned_stock',seasonal_feature_indices=[2,3])
    with pytest.raises(ValueError):
        TimeVAE(season_gate_mode='learned_stock',stock_season_gate_init=0.)
