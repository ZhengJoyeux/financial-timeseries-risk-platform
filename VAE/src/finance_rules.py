"""Post-generation accounting consistency audit and projection (disclosed as postprocessing)."""
import numpy as np
from .schema import KEYS
IDX = {key: n for n, key in enumerate(KEYS)}


def violation_rates(x):
    d = lambda name: x[..., IDX[name]]
    tests = {
        '资产非正': d('total_assets') <= 0,
        '负债为负': d('total_liabilities') < 0,
        '流动资产超过总资产': d('current_assets') > d('total_assets') + .01,
        '流动负债超过总负债': d('current_liabilities') > d('total_liabilities') + .01,
        '长期债务加流动负债超过总负债': d('long_term_debt')+d('current_liabilities') > d('total_liabilities') + .01,
        '货币资金存货应收超过流动资产': d('cash')+d('inventory')+d('accounts_receivable') > d('current_assets') + .01,
        '收入为负': d('revenue') < 0,
        '营业成本为负': d('cogs') < 0,
        '毛利未满足收入减成本': np.abs(d('gross_profit')-(d('revenue')-d('cogs'))) > (.02 + .00001*d('revenue')),
        'EBITDA小于EBIT': d('ebitda') < d('ebit') - .01,
        'EBIT超过毛利润': d('ebit') > d('gross_profit') + .01,
    }
    return {name: float(np.mean(bad)) for name,bad in tests.items()}


def repair_financial(x):
    """Project 15 correlated outputs onto selected *approximate* accounting feasibility constraints."""
    v=np.array(x,dtype=np.float64,copy=True)
    j=IDX
    get=lambda key: v[...,j[key]]
    v[...,j['total_assets']] = np.maximum(get('total_assets'), 250.)
    v[...,j['total_liabilities']] = np.clip(get('total_liabilities'), 0, 1.2*get('total_assets'))
    v[...,j['current_assets']] = np.clip(get('current_assets'), 0, get('total_assets'))
    v[...,j['current_liabilities']] = np.clip(get('current_liabilities'), 0, get('total_liabilities'))
    v[...,j['long_term_debt']] = np.clip(get('long_term_debt'),0,get('total_liabilities')-get('current_liabilities'))
    for k in ('cash','inventory','accounts_receivable'):
        v[...,j[k]]=np.maximum(get(k),0)
    liquid_sum=get('cash')+get('inventory')+get('accounts_receivable')
    factor=np.minimum(1, np.divide(.95*get('current_assets'), liquid_sum,
                                  out=np.ones_like(liquid_sum), where=liquid_sum>0))
    for k in ('cash','inventory','accounts_receivable'):
        v[...,j[k]] *= factor
    v[...,j['revenue']] = np.maximum(get('revenue'),0)
    v[...,j['cogs']] = np.maximum(get('cogs'),0)
    v[...,j['gross_profit']] = get('revenue')-get('cogs')
    v[...,j['ebit']] = np.minimum(get('ebit'),get('gross_profit'))
    v[...,j['ebitda']] = np.maximum(get('ebitda'),get('ebit'))
    # Net income/operating cash flow can be negative and are deliberately not forced positive.
    return v
