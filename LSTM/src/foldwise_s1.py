"""Fold-local S1 sampling, strict no-overlap guards and immutable provenance.

For the *synthetic* finance dataset. Never imports generated Excel made from full train.
"""
from __future__ import annotations

import hashlib
import numpy as np
from VAE.src.dataset import FinancialScaler
from VAE.src.finance_rules import violation_rates
from VAE.src.structured_finance_s1 import StructuredFinanceS1


def _index_array(indices, n, name):
    a = np.asarray(indices, dtype=np.int64)
    if a.ndim != 1 or not len(a) or len(np.unique(a)) != len(a):
        raise ValueError(f'{name}必须是非空、唯一的一维索引')
    if np.any(a < 0) or np.any(a >= n):
        raise ValueError(f'{name}存在越界索引')
    return a


def verify_partition(ids, labels, fit_idx, forbidden_idx):
    ids = np.asarray(ids)
    labels = np.asarray(labels, dtype=np.int64)
    if len(ids) != len(labels) or len(set(ids.tolist())) != len(ids):
        raise RuntimeError('企业ID重复或标签长度不匹配')
    fit = _index_array(fit_idx, len(ids), 'S1拟合企业')
    forbidden = _index_array(forbidden_idx, len(ids), '停训/留出企业')
    if set(fit.tolist()) & set(forbidden.tolist()):
        raise RuntimeError('数据泄漏：S1拟合企业与停训/留出企业重叠')
    counts = np.bincount(labels[fit], minlength=2)
    if len(counts) != 2 or np.any(counts < 3):
        raise RuntimeError('当前折S1每类至少需要3家企业')
    return fit, forbidden


def foldwise_generate(raw, labels, ids, fit_idx, forbidden_idx, *,
                      device, seed, per_class=100, perturb=.20,
                      novelty_fraction=.50, max_draw_multiplier=16):
    """S1.fit receives ONLY fit_idx rows; no validation/test/full-train generator.

    Returns raw synthetic (2*per_class,20,15), binary labels, audit metadata.
    No independent validation/test data is passed to S1 for fit or selection.
    """
    raw = np.asarray(raw, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    ids = np.asarray(ids)
    if raw.ndim != 3 or raw.shape != (len(ids),20,15):
        raise ValueError('输入必须是(N,20,15)')
    if not np.isfinite(raw).all():
        raise ValueError('输入企业财务数据包含NaN/Inf')
    fit, forbidden = verify_partition(ids, labels, fit_idx, forbidden_idx)
    if int(per_class) < 1:
        raise ValueError('每类生成数量必须大于0')
    if not str(device).startswith('cuda'):
        raise RuntimeError('正式折内S1采样强制使用GPU1映射的cuda:0')
    if str(device) != 'cuda:0':
        raise RuntimeError('仅允许cuda:0（外部必须已锁定物理GPU1 UUID）')
    labels_text = np.where(labels[fit] == 1,'有风险','无风险')
    model = StructuredFinanceS1(seed=seed, perturb=perturb, device=str(device),
                                 novelty_fraction=novelty_fraction,
                                 max_draw_multiplier=max_draw_multiplier)
    model.fit(raw[fit], labels_text)
    all_gen=[];all_y=[];checks={};diag={}
    for label, binary, offset in (('有风险',1,0),('无风险',0,1)):
        generated = model.sample(label, int(per_class), seed=int(seed)+offset)
        if generated.shape != (per_class,20,15) or not np.isfinite(generated).all():
            raise RuntimeError('S1生成数据数量、形状或有限性错误')
        rules = violation_rates(generated)
        if max(rules.values()) > 1e-3:
            raise RuntimeError(f'折内S1财务规则检查失败：{label} {rules}')
        all_gen.append(generated)
        all_y.append(np.full(per_class,binary,dtype=np.int64))
        checks[label]={key:float(val) for key,val in rules.items()}
        diag[label]=dict(model.last_sample_diagnostics[label])
    generated = np.concatenate(all_gen, axis=0)
    gen_y = np.concatenate(all_y, axis=0)
    audit={
        'fit_enterprise_ids':[int(v) for v in ids[fit]],
        'forbidden_enterprise_ids':[int(v) for v in ids[forbidden]],
        'fit_enterprises_per_class':{'risk':int(np.sum(labels[fit]==1)),
                                     'normal':int(np.sum(labels[fit]==0))},
        'generated_per_class':int(per_class),
        'S1_device':str(device), 'seed':int(seed),
        'parameters':{'perturb':float(perturb), 'novelty_fraction':float(novelty_fraction),
                      'max_draw_multiplier':int(max_draw_multiplier)},
        'raw_financial_rule_violations':checks,
        'novelty_sampler_diagnostics':diag,
        'reference_source':'ONLY fold-fit original simulated enterprises; not full-train S1',
        'never_used_validation_or_test':True,
    }
    return generated,gen_y,audit


def normalized_mode_data(real_x, real_y, generated_raw, generated_y, scaler_state, mode):
    """Pre-fitted REAL fold scaler only; all modes share identical transform."""
    if mode not in ('real_only','generated_only','real_plus_s1'):
        raise ValueError('未知训练方式')
    real_x = np.asarray(real_x,dtype=np.float32)
    real_y = np.asarray(real_y,dtype=np.int64)
    generated_y=np.asarray(generated_y,dtype=np.int64)
    scaler=FinancialScaler.from_state(scaler_state)
    zgen=scaler.transform(np.asarray(generated_raw,dtype=np.float64))
    if mode == 'real_only':
        x=real_x.copy(); y=real_y.copy(); source=np.repeat('real',len(y))
    elif mode == 'generated_only':
        x=zgen; y=generated_y.copy(); source=np.repeat('generated',len(y))
    else:
        x=np.concatenate([real_x,zgen]);y=np.r_[real_y,generated_y]
        source=np.array(['real']*len(real_y)+['generated']*len(generated_y))
    if x.shape != (len(y),20,15) or not np.isfinite(x).all():
        raise RuntimeError('折内标准化后数据非法')
    return x.astype(np.float32),y,source
