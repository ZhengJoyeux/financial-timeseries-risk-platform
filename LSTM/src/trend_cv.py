"""Leakage-safe company-level nested CV helpers for Trend-V1 vs Trend-V2.

Only raw data/splits/train.xlsx is used: no validation, test or S1 synthetic data.
The V2 production fit_leverage_stats requires exactly 48 firms, so we reproduce
its *same numerical* mean/std formula on each fold's training firms instead.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from VAE.src.dataset import FinancialScaler
from LSTM.src.trend_model_v2 import leverage_numpy, LEVERAGE_NAMES


def fold_indices(labels, n_splits=5, seed=2026):
    y=np.asarray(labels, dtype=np.int64)
    if len(y)!=48 or set(np.unique(y))!={0,1} or int(sum(y==1))!=10:
        raise ValueError('外层五折仅适用于固定48家真实训练企业（10家风险）')
    folds=list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)),y))
    observed=[]
    for tr,ho in folds:
        if len(set(tr)&set(ho)) or int(sum(y[ho]==1))!=2:
            raise RuntimeError('交叉验证企业划分失败')
        observed.extend(ho.tolist())
    if sorted(observed)!=list(range(len(y))):
        raise RuntimeError('外层折未能完整且唯一覆盖48家企业')
    return folds


def inner_indices(outer_train, labels, seed):
    y=np.asarray(labels,dtype=np.int64)
    tr=np.asarray(outer_train,dtype=np.int64)
    i,j=train_test_split(tr, test_size=0.25, random_state=seed, stratify=y[tr])
    if len(set(i)&set(j)) or len(j)==0 or min(np.bincount(y[j],minlength=2))<1:
        raise RuntimeError('内层划分失败/类别缺失')
    return np.asarray(i,dtype=int),np.asarray(j,dtype=int)


def fold_features(x_train_raw, x_other_raw):
    """Fit normalized financial axes using the supplied *training* firms only."""
    a=np.asarray(x_train_raw,dtype=np.float64)
    b=np.asarray(x_other_raw,dtype=np.float64)
    if a.ndim!=3 or a.shape[1:]!=(20,15) or b.ndim!=3 or b.shape[1:]!=(20,15):
        raise ValueError('期望(N,20,15)财务数据')
    if len(a)<2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('数据非法')
    scaler=FinancialScaler().fit(a)
    za=scaler.transform(a)
    zb=scaler.transform(b)
    state=scaler.state_dict()
    feats=leverage_numpy(za,state)
    # Same formula as Trend-V2 production, but fold train may contain only 28-39 firms.
    stats={
        'mean':feats.mean(axis=0).tolist(),
        'std':np.maximum(feats.std(axis=0),1e-3).tolist(),
        'feature_names':list(LEVERAGE_NAMES),
        'fitted_from':'current_cv_training_subfold_only',
        'n_fit_firms':int(len(za)),
    }
    if not np.all(np.isfinite(za)) or not np.all(np.isfinite(zb)):
        raise RuntimeError('归一化包含NaN/Inf')
    return za,zb,state,stats
