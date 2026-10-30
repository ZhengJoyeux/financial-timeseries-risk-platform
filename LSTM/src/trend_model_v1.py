"""15 independent univariate LSTM heads with a trend-only input view."""
import torch
from LSTM.src.model import MultiHeadFinancialLSTM

VIEW_ID = 'trend_centered_v1'

def centered_view(x):
    """Subtract the first-quarter value per firm and per variable.

    Input/output: (B,20,15), already normalized with TRAIN-only state.
    Does not change labels, financial data files, network width or heads.
    """
    if x.ndim != 3 or x.shape[1:] != (20, 15):
        raise ValueError(f'期望(B,20,15)，实际{x.shape}')
    if not torch.isfinite(x).all():
        raise ValueError('输入包含NaN/Inf')
    return x - x[:, :1, :]

class TrendCentered15HeadLSTM(MultiHeadFinancialLSTM):
    """Model variant: one variable per recurrent head, 15 heads unchanged."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.n_features != 15 or self.seq_len != 20:
            raise RuntimeError('Trend-V1 仅支持15变量和20季度')
    def forward(self, x):
        return super().forward(centered_view(x))
