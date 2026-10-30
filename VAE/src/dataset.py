"""Read/write five annual rows per company while modeling complete 20-quarter sequences."""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from .schema import EXCEL_COLUMNS, N_QUARTERS, N_FEATURES, LABELS, validate_columns


def to_frame(ids, x, labels):
    """Represent each company as exactly five annual rows, with 4 quarters per row.

    Same (N,20,15) model tensor as the earlier 302-column format: only Excel layout changes.
    """
    ids = np.asarray(ids, dtype=np.int64)
    x = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels)
    if x.shape != (len(ids), N_QUARTERS, N_FEATURES):
        raise ValueError(f'数据形状错误：{x.shape}，预期(N,20,15)')
    if len(set(ids.tolist())) != len(ids) or any(label not in LABELS for label in labels):
        raise ValueError('企业编号必须唯一，且标签只能是有风险/无风险。')
    if not np.isfinite(x).all():
        raise ValueError('财务数据有缺失或无限值。')
    rows = []
    for i, enterprise_id in enumerate(ids):
        for year in range(1,6):
            four_quarters = np.round(x[i,(year-1)*4:year*4,:].reshape(-1),2)
            rows.append([int(enterprise_id),year] + four_quarters.tolist() + [str(labels[i])])
    return pd.DataFrame(rows,columns=EXCEL_COLUMNS)


def read_financial(path):
    """Read five annual rows per firm; reject incomplete firms or conflicting labels.

    Returns one row per unique enterprise to the model, preserving (N,20,15).
    """
    df = pd.read_excel(path, sheet_name='财务数据')
    validate_columns(df.columns)
    if len(df)==0 or len(df)%5:
        raise ValueError('Excel必须每家企业5行，每行1年；当前总行数不是5的倍数。')
    if df[['企业编号','年份','风险标签']].isna().any().any():
        raise ValueError('企业编号、年份、风险标签不能为空。')
    if df[['企业编号','年份']].duplicated().any():
        raise ValueError('同一企业的同一年出现重复行。')
    if not set(df['风险标签']).issubset(set(LABELS)):
        raise ValueError('存在未知风险标签。')
    vals=df.iloc[:,2:-1].to_numpy(dtype=np.float64)
    if not np.isfinite(vals).all():
        raise ValueError('金融变量含有缺失或无限值；当前阶段不进行自动填充。')
    ids=[];records=[];labels=[]
    for enterprise_id, group in df.groupby('企业编号',sort=False):
        group=group.sort_values('年份')
        if group['年份'].tolist()!=[1,2,3,4,5] or len(group)!=5:
            raise ValueError(f'企业{enterprise_id}必须包含年份1到5各一行。')
        if group['风险标签'].nunique()!=1:
            raise ValueError(f'企业{enterprise_id}的5年风险标签不一致。')
        ids.append(int(enterprise_id))
        records.append(group.iloc[:,2:-1].to_numpy(dtype=np.float64).reshape(20,N_FEATURES))
        labels.append(group['风险标签'].iloc[0])
    if len(ids)*5!=len(df):
        raise ValueError('企业年份数据不完整。')
    return np.asarray(ids,dtype=np.int64),np.stack(records),np.asarray(labels)


def save_excel(path, ids, x, labels):
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(dest) as writer:
        to_frame(ids, x, labels).to_excel(writer, sheet_name='财务数据', index=False)


def stratified_company_split(ids, x, labels, seed=2026):
    all_indices = np.arange(len(ids))
    train_idx, other_idx = train_test_split(all_indices, test_size=.4, random_state=seed, stratify=labels)
    validation_idx, test_idx = train_test_split(other_idx, test_size=.5, random_state=seed, stratify=labels[other_idx])
    result = {'train': (ids[train_idx],x[train_idx],labels[train_idx]),
              'validation':(ids[validation_idx],x[validation_idx],labels[validation_idx]),
              'test':(ids[test_idx],x[test_idx],labels[test_idx])}
    for name, (si, _, sl) in result.items():
        expected = {'train':(48, 10), 'validation':(16, 3), 'test':(16,3)}[name]
        if len(si) != expected[0] or np.sum(sl == '有风险') != expected[1]:
            raise RuntimeError(f'Unexpected split {name}: {len(si)} / {sum(sl=="有风险")}')
    return result


class FinancialScaler:
    """Per-variable signed asinh + standardization; FIT ON TRAIN ONLY, including both classes."""
    def fit(self, x):
        self.scale = np.maximum(np.median(np.abs(x), axis=(0,1)), 1.0)
        y = np.arcsinh(x / self.scale[None,None,:])
        self.mean = y.mean(axis=(0,1))
        self.std = np.maximum(y.std(axis=(0,1)), .08)
        return self
    def transform(self, x):
        return ((np.arcsinh(x/self.scale)-self.mean)/self.std).astype(np.float32)
    def inverse_transform(self, x):
        t = np.clip(x*self.std+self.mean, -10, 10)
        return np.sinh(t)*self.scale
    def state_dict(self):
        return {'scale': self.scale.tolist(), 'mean': self.mean.tolist(), 'std': self.std.tolist()}
    @classmethod
    def from_state(cls, state):
        obj=cls()
        for key in ('scale','mean','std'):
            setattr(obj,key,np.array(state[key],dtype=np.float64))
        return obj
