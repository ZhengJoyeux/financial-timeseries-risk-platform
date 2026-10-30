"""Train-only class-conditional quarterly dynamics penalty for *prior* samples.

The real reference distribution comes exclusively from raw training enterprises.
Prior samples keep z~N(0,I).  The penalty is not a hard financial rule and
cannot by itself guarantee accounting consistency, diversity or good downstream
predictive performance.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

STOCK_INDICES = (0, 1, 4, 5, 6, 7, 8, 9)
Q_LEVELS = (0.05, 0.15, 0.25, 0.5, 0.75, 0.85, 0.95)


class PriorStockQOQ:
    """Match signed log QoQ growth quantiles for 8 nonnegative stock variables.

    The soft positivity map handles prior samples with negative *unrepaired*
    stocks while remaining differentiable. `floor` is fixed from TRAIN only;
    it prevents near-zero denominators, not an arbitrary fixed growth cap.
    """

    def __init__(self, raw_train, binary_train_labels, scaler, device):
        x = np.asarray(raw_train, dtype=np.float64)
        y = np.asarray(binary_train_labels, dtype=np.int64)
        if x.ndim != 3 or x.shape[1:] != (20, 15) or y.shape != (len(x),):
            raise ValueError("训练数据应为(N,20,15)，标签为(N,)")
        if not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
            raise ValueError("原始训练数据或标签含无效值")
        if sum(y == 0) == 0 or sum(y == 1) == 0:
            raise ValueError("训练集必须同时包含两种风险类别")
        self.device = device
        self.index = torch.tensor(STOCK_INDICES, device=device, dtype=torch.long)
        # 1% median positive TRAIN magnitude, minimum one data unit.
        magnitude = np.median(np.maximum(x[:, :, STOCK_INDICES], 0.0), axis=(0, 1))
        self.floor_numpy = np.maximum(0.01 * magnitude, 1.0)
        self.floor = torch.tensor(self.floor_numpy, device=device, dtype=torch.float32)
        self.scale = torch.tensor(np.asarray(scaler.scale)[list(STOCK_INDICES)], device=device, dtype=torch.float32)
        self.mean = torch.tensor(np.asarray(scaler.mean)[list(STOCK_INDICES)], device=device, dtype=torch.float32)
        self.std = torch.tensor(np.asarray(scaler.std)[list(STOCK_INDICES)], device=device, dtype=torch.float32)
        self.levels = torch.tensor(Q_LEVELS, device=device, dtype=torch.float32)
        targets = []
        for cls in (0, 1):
            stocks = x[y == cls][:, :, list(STOCK_INDICES)]
            # NumPy stable softplus is numerically equivalent to PyTorch softplus.
            pos = np.logaddexp(0.0, stocks / self.floor_numpy) * self.floor_numpy
            log_pos = np.log(pos + self.floor_numpy)
            qoq = np.diff(log_pos, axis=1).reshape(-1, len(STOCK_INDICES))
            targets.append(np.quantile(qoq, Q_LEVELS, axis=0))
        self.target = torch.tensor(np.stack(targets), device=device, dtype=torch.float32)

    def growth_quantiles(self, standardized):
        if standardized.ndim != 3 or standardized.shape[1:] != (20, 15):
            raise ValueError("生成序列必须为(N,20,15)")
        stock_std = standardized.index_select(2, self.index)
        # This matches FinancialScaler.inverse_transform's clipping convention.
        linear = (stock_std * self.std + self.mean).clamp(-10.0, 10.0)
        raw = torch.sinh(linear) * self.scale
        positive = F.softplus(raw / self.floor) * self.floor
        qoq = torch.log(positive[:, 1:] + self.floor) - torch.log(positive[:, :-1] + self.floor)
        return torch.quantile(qoq.reshape(-1, len(STOCK_INDICES)), self.levels, dim=0)

    def __call__(self, model, samples_per_class=16):
        count = int(samples_per_class)
        if count < 2:
            raise ValueError("samples_per_class必须至少为2")
        loss_parts = []
        for cls in (0, 1):
            z = torch.randn(count, model.params['latent_dim'], device=self.device)
            condition = torch.full((count,), cls, dtype=torch.long, device=self.device)
            generated = model.decode(z, condition)
            generated_quantiles = self.growth_quantiles(generated)
            loss_parts.append(F.l1_loss(generated_quantiles, self.target[cls]))
        return torch.stack(loss_parts).mean()

    def state_dict(self):
        return {
            'version': 1,
            'stock_indices': list(STOCK_INDICES),
            'quantile_levels': list(Q_LEVELS),
            'floor_train_only': self.floor_numpy.tolist(),
            'target_quantiles_by_class_0_then_1': self.target.detach().cpu().tolist(),
            'real_qoq_definition': 'signed_diff(log(softplus(raw/floor)*floor + floor))',
        }
