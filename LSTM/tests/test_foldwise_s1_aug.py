"""Company-level leakage and fold-local S1 augmentation contract."""
import numpy as np
import pytest
import torch
from LSTM.src.foldwise_s1 import verify_partition,normalized_mode_data,foldwise_generate
from LSTM.scripts.cv_trend_v2_s1_augmentation import sampler_for_equal_steps,aggregate_oof
from VAE.src.dataset import FinancialScaler


def _dummy_dataset():
    ids=np.arange(20,dtype=int)
    labels=np.array([1]*6+[0]*14,dtype=np.int64)
    return ids,labels


def test_verification_rejects_holdout_in_fit_and_small_risk():
    ids,y=_dummy_dataset()
    fit=np.r_[0:6,6:16]
    forbidden=np.arange(16,20)
    a,b=verify_partition(ids,y,fit,forbidden)
    assert len(a)==16 and len(b)==4
    with pytest.raises(RuntimeError,match='重叠'):
        verify_partition(ids,y,fit,np.array([15,16]))
    with pytest.raises(RuntimeError,match='3家'):
        verify_partition(ids,y,np.r_[0:2,6:16],np.arange(16,20))


def test_leakage_safe_fold_scaler_for_generated_and_real():
    rng=np.random.default_rng(11)
    real=np.abs(rng.normal(loc=100,scale=5,size=(8,20,15)))
    gen=np.abs(rng.normal(loc=200,scale=5,size=(6,20,15)))
    scaler=FinancialScaler().fit(real)
    x0=scaler.transform(real)
    state=scaler.state_dict()
    yr=np.array([1,1,0,0,0,0,0,0]); yg=np.array([1,1,1,0,0,0])
    xs,ys,src=normalized_mode_data(x0,yr,gen,yg,state,'real_plus_s1')
    assert xs.shape==(14,20,15)
    assert (ys[:8]==yr).all() and (ys[8:]==yg).all()
    assert (src[:8]=='real').all() and (src[8:]=='generated').all()
    expected=scaler.transform(gen)
    np.testing.assert_allclose(xs[8:],expected)
    assert not np.allclose(xs[8:],FinancialScaler().fit(gen).transform(gen))
    pure,labels,s=normalized_mode_data(x0,yr,gen,yg,state,'generated_only')
    assert len(pure)==6 and np.array_equal(labels,yg) and set(s)=={'generated'}


def test_fixed_step_balanced_sampling_all_modes():
    yr=np.array([1]*6+[0]*20)
    yg=np.array([1]*10+[0]*10)
    cases=[('real_only',yr,np.repeat('real',len(yr))),
           ('generated_only',yg,np.repeat('generated',len(yg))),
           ('real_plus_s1',np.r_[yr,yg],np.array(['real']*len(yr)+['generated']*len(yg)))]
    for name,y,s in cases:
        sampler=sampler_for_equal_steps(y,s,name,.5,2026,200)
        ix=list(sampler)
        assert len(ix)==200
        # sampled risk count should be near half at this many draws
        assert 65<sum(y[ix])<135


def test_s1_fit_sample_never_passes_holdouts(monkeypatch):
    import LSTM.src.foldwise_s1 as component
    ids=np.arange(16)
    yy=np.array([1]*6+[0]*10)
    real=np.zeros((16,20,15),dtype=float)
    real[:,:,0]=1000; real[:,:,1]=500; real[:,:,2]=200
    observed={}
    class FakeS1:
        def __init__(self,**kwargs):self.calls=[]
        def fit(self,x,labels):
            observed['n_fit']=len(x)
            observed['labels']=labels.tolist()
            return self
        def sample(self,label,number,seed):
            self.last_sample_diagnostics={label:{'quantile_slots_without_threshold_pass_fallback_to_max_distance':0}}
            return np.tile(real[0],(number,1,1))
    # simulate accounting checks without requiring a synthetic accounting generator
    monkeypatch.setattr(component,'StructuredFinanceS1',FakeS1)
    monkeypatch.setattr(component,'violation_rates',lambda x:{'dummy':0.})
    train=np.r_[0:6,6:13];held=np.arange(13,16)
    fake,labels,audit=foldwise_generate(real,yy,ids,train,held,device='cuda:0',seed=9,per_class=4)
    assert observed['n_fit']==13
    assert not (set(audit['fit_enterprise_ids']) & set(audit['forbidden_enterprise_ids']))
    assert fake.shape==(8,20,15) and np.array_equal(labels,np.array([1]*4+[0]*4))


def test_oof_requires_48_unique_and_same_labels():
    # Covers the strict no-duplicated-company contract
    rows=[]
    for mode in ('real_only','generated_only','real_plus_s1'):
        for i in range(48):
            rows.append({'mode':mode,'company_id':i,'ground_truth_risk':int(i<10),
                         'risk_probability':float(.7 if i<10 else .3)})
    stats,preds=aggregate_oof(rows)
    assert len(stats)==3 and len(preds['real_only'][0])==48
    rows.append(dict(rows[0]))
    with pytest.raises(RuntimeError,match='48家'):
        aggregate_oof(rows)
