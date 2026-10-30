import numpy as np
import pytest
from VAE.src.schema import KEYS
from VAE.src.finance_rules import violation_rates
from VAE.src.structured_finance import StructuredFinanceBaseline


def _sample_data():
    from VAE.src.synthetic_data import build_simulated_companies
    _, x, y=build_simulated_companies()
    return x,y


def test_structured_15_fields_and_no_raw_violations():
    x,y=_sample_data()
    generator=StructuredFinanceBaseline().fit(x[:],y[:])
    for label in ['有风险','无风险']:
        z=generator.sample(label,12)
        assert z.shape==(12,20,15)
        assert np.isfinite(z).all()
        assert max(violation_rates(z).values())<1e-3
        rev=z[...,KEYS.index('revenue')];cogs=z[...,KEYS.index('cogs')]
        assert np.allclose(z[...,KEYS.index('gross_profit')],rev-cogs)
        assert np.std(z[...,KEYS.index('total_assets')])>1


def test_repeatable_without_replicating_whole_enterprise():
    x,y=_sample_data()
    fitted=StructuredFinanceBaseline(seed=123).fit(x,y)
    r1=fitted.sample('有风险',6,seed=1)
    r2=fitted.sample('有风险',6,seed=1)
    assert np.array_equal(r1,r2)
    assert all(np.min(np.linalg.norm((x-r[None]).reshape(len(x),-1),axis=1))>1
               for r in r1)


def test_fit_requires_two_labels_and_no_invalid_input():
    x,y=_sample_data()
    with pytest.raises(ValueError):StructuredFinanceBaseline().fit(x[y=='有风险'],y[y=='有风险'])
    xx=x.copy();xx[0,0,0]=np.nan
    with pytest.raises(ValueError):StructuredFinanceBaseline().fit(xx,y)
