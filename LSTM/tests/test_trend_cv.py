"""No GPU needed: nested splits, leak-free scaling and model interface smoke."""
import numpy as np
import pytest
import torch
from LSTM.src.trend_cv import fold_indices, inner_indices, fold_features
from LSTM.scripts.cv_trend_v1_v2 import make_model, macro_bce, paired_bootstrap
from LSTM.src.trend_model_v2 import leverage_numpy


def synthetic():
    rng=np.random.default_rng(73)
    x=np.abs(rng.normal(150.,20.,size=(48,20,15)))
    x[:,:,0]=rng.uniform(300,400,size=(48,20))
    x[:,:,1]=x[:,:,0]*rng.uniform(.3,.8,size=(48,20))
    y=np.asarray([1]*10+[0]*38,dtype=np.int64)
    return x,y


def test_outer_folds_company_exclusive_and_full_coverage():
    _,y=synthetic()
    folds=fold_indices(y,seed=2026)
    assert len(folds)==5
    held=[]
    for i,(tr,ho) in enumerate(folds,1):
        assert len(tr) in (38,39)
        assert len(ho) in (9,10)
        assert int(sum(y[ho]==1))==2
        a,b=inner_indices(tr,y,seed=2026+i)
        assert set(a).isdisjoint(set(b))
        assert set(a)|set(b)==set(tr)
        assert set(ho).isdisjoint(set(a))
        assert set(ho).isdisjoint(set(b))
        assert (y[a]==1).any() and (y[b]==1).any()
        held.extend(ho.tolist())
    assert sorted(held)==list(range(48))


def test_scaler_is_fold_train_only():
    x,y=synthetic()
    train=x[:31].copy(); held=x[31:40].copy()
    za,zb,state,stats=fold_features(train,held)
    _,_,state_changed,_=fold_features(train,held*100)
    assert state==state_changed
    from VAE.src.dataset import FinancialScaler
    expected=FinancialScaler().fit(train).state_dict()
    assert state==expected
    assert stats['n_fit_firms']==31
    assert stats['fitted_from']=='current_cv_training_subfold_only'
    assert np.allclose(stats['mean'],leverage_numpy(za,state).mean(axis=0))
    assert za.shape==(31,20,15) and zb.shape==(9,20,15)


def test_synthetic_finance_v2_exactly_15_heads_and_gradients():
    x,y=synthetic()
    za,zb,state,stats=fold_features(x[:38],x[38:])
    cfg={'n_features':15,'seq_len':20,'hidden_size':12,'num_layers':1,'fusion_hidden':64,'dropout':.25}
    for version,expected_params in [('trend_v1',24497),('trend_v2_leverage',24625)]:
        model=make_model(version,state,stats,cfg)
        assert len(model.heads)==15
        assert len(set(id(h) for h in model.heads))==15
        assert sum(p.numel() for p in model.parameters())==expected_params
        inp=torch.from_numpy(za[:4]); logits=model(inp)
        assert logits.shape==(4,)
        logits.square().mean().backward()
        assert model.heads[0].weight_ih_l0.grad is not None
        if version=='trend_v2_leverage':
            assert model.leverage_adapter.weight.grad is not None


def test_macro_bce_formula():
    y=np.array([0,1,0,1])
    p=np.array([.1,.9,.2,.8])
    target=(-np.log([.9,.8]).mean()-np.log([.9,.8]).mean())/2
    assert np.isclose(macro_bce(y,p),target)


def test_bootstrap_does_not_change_originals():
    y=np.array([0]*6+[1]*4)
    a=np.linspace(.1,.9,10);b=a[::-1].copy()
    copy_a=a.copy();copy_b=b.copy()
    out=paired_bootstrap(y,a,b,n=8,seed=11)
    assert set(out)=={'auroc','pr_auc','brier'}
    assert np.array_equal(copy_a,a) and np.array_equal(copy_b,b)
    assert all(np.isfinite(x['v2_minus_v1_p025']) for x in out.values())


def test_invalid_fold_labels_rejected():
    with pytest.raises(ValueError):
        fold_indices(np.zeros(48,dtype=int))
