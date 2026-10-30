"""15 independent univariate LSTM heads for quarterly finance classification."""
import torch
from torch import nn


class MultiHeadFinancialLSTM(nn.Module):
    """Input [batch, quarters, features]; one LSTM for each financial variable.

    These heads are independent recurrent networks, NOT attention heads.
    Output is a single risk logit; apply sigmoid only during inference.
    """
    def __init__(self, n_features=15, seq_len=20, hidden_size=12,
                 num_layers=1, fusion_hidden=64, dropout=0.2):
        super().__init__()
        if min(n_features, seq_len, hidden_size, num_layers, fusion_hidden) <= 0:
            raise ValueError('网络维度必须为正整数')
        if not 0 <= dropout < 1:
            raise ValueError('dropout必须在[0,1)')
        self.n_features = int(n_features)
        self.seq_len = int(seq_len)
        self.heads = nn.ModuleList([
            nn.LSTM(input_size=1, hidden_size=hidden_size,
                    num_layers=num_layers, batch_first=True)
            for _ in range(n_features)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(n_features * hidden_size, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, max(8, fusion_hidden // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(8, fusion_hidden // 2), 1),
        )
        self.config = dict(n_features=int(n_features), seq_len=int(seq_len),
                           hidden_size=int(hidden_size), num_layers=int(num_layers),
                           fusion_hidden=int(fusion_hidden), dropout=float(dropout))

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.n_features):
            raise ValueError(f'输入形状应为(B,{self.seq_len},{self.n_features})，实际{x.shape}')
        representations = []
        for i, lstm in enumerate(self.heads):
            # Each head sees only one feature through all 20 quarters.
            series = x[:, :, i:i+1]
            _, (last_hidden, _) = lstm(series)
            representations.append(last_hidden[-1])
        fused = torch.cat(representations, dim=-1)
        return self.classifier(fused).squeeze(-1)
