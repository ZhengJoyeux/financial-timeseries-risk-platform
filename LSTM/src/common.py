"""Reproducibility, file hashing and physical GPU1 lock."""
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import numpy as np


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_save_exclusive(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def physical_gpu1():
    """GPU1 ONLY. No CPU / physical GPU0 fallback for training or inference."""
    result = subprocess.run(
        ['nvidia-smi', '-i', '1', '--query-gpu=uuid', '--format=csv,noheader'],
        capture_output=True, text=True, check=True,
    )
    expected = result.stdout.strip()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    if not expected.startswith('GPU-') or visible != expected:
        raise RuntimeError(f'必须独占物理GPU1 UUID。预期={expected}，当前CUDA_VISIBLE_DEVICES={visible!r}')
    if os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID':
        raise RuntimeError('必须先设置 CUDA_DEVICE_ORDER=PCI_BUS_ID')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('物理GPU1不可用或GPU隔离失败，禁止退回CPU/GPU0')
    device = torch.device('cuda:0')
    x = torch.ones(16, 16, device=device)
    torch.cuda.synchronize()
    if not torch.isfinite(x @ x).all():
        raise RuntimeError('GPU计算异常')
    print('物理GPU1通过：', expected, torch.cuda.get_device_name(0), flush=True)
    return device
