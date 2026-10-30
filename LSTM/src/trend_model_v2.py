"""Trend-V2: 15 independent LSTM heads + 2 train-fitted leverage descriptors.

No input files or labels are modified. Invert train-only FinancialScaler
before forming raw financial ratios. NEVER divide asinh-normalized values.
"""
import numpy as np
import torch
from torch import nn
from LSTM.src.model import MultiHeadFinancialLSTM
from LSTM.src.trend_model_v1 import centered_view

VIEW_ID = 'trend_centered_plus_leverage_v2'
LEVERAGE_NAMES = ('mean_debt_to_assets', 'last4_minus_first4_debt_to_assets')


def leverage_numpy(x_normalized, scaler_state):
    """x: normalized [N,20,15]. Matches FinancialScaler.inverse_transform."""
    x = np.asarray(x_normalized, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (20, 15) or not np.isfinite(x).all():
        raise ValueError('leverage输入需为有限(N,20,15)')
    state = [np.asarray(scaler_state[k], dtype=np.float64) for k in ('scale','mean','std')]
    if any(a.shape != (15,) for a in state):
        raise ValueError('FinancialScaler状态不符合15财务变量')
    scale, mean, std = state
    physical = np.sinh(np.clip(x*std[None,None,:] + mean[None,None,:], -10,10))*scale[None,None,:]
    assets = physical[:,:,0]
    debts = physical[:,:,1]
    if np.any(assets<=0) or not np.isfinite(physical).all():
        raise ValueError('总资产非正或反归一化出现NaN/Inf')
    leverage = debts / assets
    features = np.column_stack((leverage.mean(axis=1),
                                 leverage[:,-4:].mean(axis=1)-leverage[:,:4].mean(axis=1)))
    if not np.isfinite(features).all():
        raise ValueError('资产负债率特征异常')
    return features


def fit_leverage_stats(real_x, scaler_state):
    """Fit this normalization using only 48 original TRAIN firms."""
    features = leverage_numpy(real_x, scaler_state)
    if len(features)!=48:
        raise RuntimeError('仅允许48家原始训练企业拟合杠杆特征尺度')
    return {'mean':features.mean(axis=0).tolist(),
            'std':np.maximum(features.std(axis=0), 1e-3).tolist(),
            'feature_names':list(LEVERAGE_NAMES),
            'fitted_from':'train_real_only_48'}


class TrendLeverage15HeadLSTM(MultiHeadFinancialLSTM):
    """Identical 15 univariate recurrent heads; 2-d additive leverage adapter."""
    def __init__(self, *, scaler_state, leverage_stats, leverage_enabled=True, **kwargs):
        super().__init__(**kwargs)
        if self.n_features != 15 or self.seq_len != 20:
            raise RuntimeError('Trend-V2只支持15变量×20季度')
        if tuple(leverage_stats.get('feature_names',())) != LEVERAGE_NAMES:
            raise ValueError('杠杆特征顺序不匹配')
        for name, value in [
            ('scale', scaler_state['scale']), ('mean', scaler_state['mean']),
            ('std',scaler_state['std'])]:
            tensor=torch.as_tensor(value, dtype=torch.float64)
            if tensor.shape!=(15,):raise ValueError(f'归一化状态{name}异常')
            self.register_buffer('scaler_'+name,tensor)
        for name in ('mean','std'):
            tensor=torch.as_tensor(leverage_stats[name],dtype=torch.float64)
            if tensor.shape!=(2,):raise ValueError(f'杠杆统计{name}异常')
            self.register_buffer('leverage_'+name,tensor)
        if (self.leverage_std<=0).any() or (self.scaler_scale<=0).any():
            raise ValueError('归一化尺度必须为正')
        self.leverage_enabled=bool(leverage_enabled)
        self.leverage_adapter=nn.Linear(2,self.classifier[0].out_features,bias=False)
        nn.init.normal_(self.leverage_adapter.weight, mean=0, std=.02)
        self.input_view=VIEW_ID

    def compute_leverage(self, x):
        """Exact inverse transform, then normalized raw debt/assets features."""
        if x.ndim!=3 or x.shape[1:]!=(20,15) or not torch.isfinite(x).all():
            raise ValueError('输入必须是有限(B,20,15)')
        # financial ratio is fixed, non-learned feature; do not backprop through inverse.
        with torch.no_grad():
            x64=x.detach().to(torch.float64)
            z=(x64*self.scaler_std + self.scaler_mean).clamp(-10,10)
            physical=torch.sinh(z)*self.scaler_scale
            assets=physical[:,:,0]
            if not torch.isfinite(physical).all() or (assets<=0).any():
                raise RuntimeError('总资产非正或财务逆归一化异常')
            ratio=physical[:,:,1]/assets
            feats=torch.stack((ratio.mean(dim=1),
                               ratio[:,-4:].mean(dim=1)-ratio[:,:4].mean(dim=1)),dim=1)
            feats=((feats-self.leverage_mean)/self.leverage_std).clamp(-8,8)
            if not torch.isfinite(feats).all():
                raise RuntimeError('杠杆特征包含NaN/Inf')
        return feats.to(dtype=x.dtype)

    def forward(self,x):
        trend=centered_view(x)
        reps=[]
        for i,lstm in enumerate(self.heads):
            _,(h,_) = lstm(trend[:,:,i:i+1])
            reps.append(h[-1])
        fused=torch.cat(reps,dim=-1)
        preactivation=self.classifier[0](fused)
        if self.leverage_enabled:
            preactivation=preactivation+self.leverage_adapter(self.compute_leverage(x))
        return self.classifier[1:](preactivation).squeeze(-1)
