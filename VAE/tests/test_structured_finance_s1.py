"""CPU unit tests. Formal fitting and sampling must use physical GPU1."""
import numpy as np
import pytest
from VAE.src.dataset import read_financial,FinancialScaler
from VAE.src.finance_rules import violation_rates
from VAE.src.structured_finance import StructuredFinanceBaseline
from VAE.src.structured_finance_s1 import StructuredFinanceS1
from VAE.src.synthetic_data import build_simulated_companies


def _training():
    _, x, y = build_simulated_companies()
    return x, y


def test_s1_reproducible_and_accounting_constraints():
    x,y=_training()
    m=StructuredFinanceS1(device='cpu',seed=2027).fit(x,y)
    a=m.sample('有风险',12,seed=310)
    b=m.sample('有风险',12,seed=310)
    assert a.shape==(12,20,15)
    assert np.array_equal(a,b)
    assert np.isfinite(a).all()
    assert max(violation_rates(a).values()) < 1e-3
    assert m.last_sample_diagnostics['有风险']['stratified_quantile_slots']==12


def test_s1_tail_preservation_on_fixed_split():
    from pathlib import Path
    train=Path('data/splits/train.xlsx')
    if not train.is_file():pytest.skip('Fixed training Excel unavailable')
    _,x,y=read_financial(train)
    m=StructuredFinanceS1(device='cpu',seed=2027).fit(x,y)
    for label,seed in [('有风险',2027),('无风险',2028)]:
        generated=m.sample(label,100,seed=seed)
        real=x[y==label]
        train_tail=np.quantile(real[:,:,0],.95)
        generated_tail=np.quantile(generated[:,:,0],.95)
        assert .65 < generated_tail / train_tail < 1.50
        assert max(violation_rates(generated).values()) < 1e-3
        assert m.last_sample_diagnostics[label]['accepted']==100


def test_s1_no_validation_or_test_needed():
    x,y=_training()
    m=StructuredFinanceS1(device='cpu').fit(x,y)
    assert set(m.groups)=={'有风险','无风险'}
    assert len(m.references['有风险'])==int(np.sum(y=='有风险'))
    assert m.novelty_threshold['有风险']>0
    with pytest.raises(ValueError):StructuredFinanceS1(novelty_fraction=2)
    with pytest.raises(ValueError):StructuredFinanceS1(max_draw_multiplier=1)


def test_s0_still_loads_and_generates_unchanged():
    x,y=_training()
    s0=StructuredFinanceBaseline(seed=2027,device='cpu').fit(x,y)
    p=s0.sample('无风险',5,seed=41)
    q=s0.sample('无风险',5,seed=41)
    assert np.array_equal(p,q)
    assert np.isfinite(p).all()
