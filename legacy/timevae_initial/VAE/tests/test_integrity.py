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
