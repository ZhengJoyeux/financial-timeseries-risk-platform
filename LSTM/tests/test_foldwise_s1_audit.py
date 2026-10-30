"""Read-only quality audit tests; no GPU, test.xlsx, validation.xlsx, or training."""
import numpy as np
import pytest

from LSTM.scripts.audit_foldwise_s1 import (
    audit_provenance,validate_npz,audit_class,check_identity,
)
from VAE.src.dataset import FinancialScaler
from VAE.src.finance_rules import repair_financial
from VAE.src.schema import KEYS
from scipy.spatial.distance import cdist


def test_provenance_disjoint_and_exact():
    p={'fit_enterprise_ids':[1,2,3], 'forbidden_enterprise_ids':[4,5],
       'real_fit_ids':[1,2,3], 'other_ids':[4,5], 'real_fit_sha256':'hash',
       'fold':1, 'phase':'inner', 'generated_per_class':100, 'S1_device':'cuda:0',
       'never_used_validation_or_test':True,
       'reference_source':'ONLY fold-fit original simulated enterprises'}
    assert audit_provenance(p,[1,2,3],[4,5],[1,2,3,4,5],
                             fold=1,phase='inner',sourcehash='hash',n=100)=={1,2,3}
    p['fit_enterprise_ids']=[1,2,4]
    with pytest.raises(ValueError,match='拟合ID'):
        audit_provenance(p,[1,2,3],[4,5],[1,2,3,4,5],
                         fold=1,phase='inner',sourcehash='hash',n=100)


def test_npz_origin_and_shape(tmp_path):
    path=tmp_path/'generated_raw.npz'
    np.savez(path,x=np.ones((200,20,15)),y=np.r_[np.ones(100),np.zeros(100)],
             source=np.repeat('generated',200),origin_fold=np.ones(200),
             origin_phase=np.repeat('outer',200))
    x,y=validate_npz(path,fold=1,phase='outer',n=100)
    assert x.shape==(200,20,15) and int(y.sum())==100
    with pytest.raises(ValueError,match='origin_phase'):
        validate_npz(path,fold=1,phase='inner',n=100)


def test_audit_class_independent_novelty_and_rules():
    rng=np.random.default_rng(123)
    x=np.ones((6,20,15),dtype=np.float64)
    x[...,KEYS.index('total_assets')] = rng.uniform(1000,2000,(6,20))
    x[...,KEYS.index('total_liabilities')] = 400
    x[...,KEYS.index('current_assets')] = 600
    x[...,KEYS.index('current_liabilities')] = 200
    x[...,KEYS.index('long_term_debt')] = 150
    x[...,KEYS.index('cash')] = 80
    x[...,KEYS.index('inventory')] = 90
    x[...,KEYS.index('accounts_receivable')] = 100
    x[...,KEYS.index('revenue')] = 650
    x[...,KEYS.index('cogs')] = 400
    x[...,KEYS.index('gross_profit')] = 250
    x[...,KEYS.index('ebit')] = 100
    x[...,KEYS.index('ebitda')] = 120
    x[...,KEYS.index('net_income')] = 60
    x[...,KEYS.index('operating_cash_flow')] = 50
    gen = np.repeat(x[:1],100,axis=0).copy()
    gen[...,0] *= rng.uniform(.9,1.1,size=(100,1))
    scaler=FinancialScaler().fit(x)
    zr=scaler.transform(x).reshape(6,-1).astype(float)
    dist=cdist(zr,zr);np.fill_diagonal(dist,np.inf)
    threshold=.5*np.median(dist.min(axis=1))
    diag={'nearest_neighbor_threshold_standardized_euclidean':float(threshold),
          'quantile_slots_without_threshold_pass_fallback_to_max_distance':100,
          'candidate_draws':1600,'stratified_quantile_slots':100,'accepted':100,
          'threshold_satisfaction_fraction':0.0}
    row, features=audit_class(gen,x,scaler=scaler,novelty_fraction=.5,
                             diagnostics=diag,name='risk',fold=1,phase='inner')
    assert row['generated_enterprises']==100 and len(features)==15
    assert row['max_financial_rule_violation_rate']==0.0
    assert row['fallback_slots_reported']==100


def test_protocol_safeguards():
    ids=np.arange(48)
    y=np.r_[np.ones(10,dtype=int),np.zeros(38,dtype=int)]
    protocol={'train_sha256':'a','read_test':False,'read_validation':False,
              'pre_generated_full_train_S1_used':False,'smoke':False,
              'n_outer_folds':5,'folds_executed':5,'generated_per_class':100,
              'train_file':'data/splits/train.xlsx',
              'S1_fit':'once per phase only current inner/outer real training subfold',
              'scaler_fit':'only current inner/outer real training subfold'}
    assert check_identity(protocol,ids,y,'a')
    protocol['pre_generated_full_train_S1_used']=True
    with pytest.raises(ValueError,match='全量'):
        check_identity(protocol,ids,y,'a')
