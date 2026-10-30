"""Freeze and report validation predictions for 15-head LSTM modes (GPU1 only).

Does not access data/splits/test.xlsx. Does not tune thresholds or checkpoints.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from LSTM.src.common import physical_gpu1, sha256_file
from LSTM.src.data import load_npz
from LSTM.src.evaluation_reporting import build_report
from LSTM.src.model import MultiHeadFinancialLSTM
from LSTM.scripts.train import MODES, inference


def load_history(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f'history.csv为空：{path}')
    epochs = [int(r['epoch']) for r in rows]
    if epochs != list(range(1, len(rows) + 1)):
        raise RuntimeError(f'history.csv轮次不连续：{path}')
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='LSTM/config/multihead_lstm.yaml')
    parser.add_argument('--smoke', action='store_true',
                        help='只评价generated_only_s1_smoke，不能作为正式结果')
    parser.add_argument('--output', default=None)
    parser.add_argument('--bootstrap-repeats', type=int, default=1000)
    args = parser.parse_args()
    if not 50 <= args.bootstrap_repeats <= 10000:
        raise ValueError('bootstrap次数必须在50到10000之间')
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    base = Path(cfg['project']['dataset'])
    manifest_path = base / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('test_read') is not False:
        raise RuntimeError('预处理不是test密封状态，停止')
    val_file = base / 'validation.npz'
    val_x, val_y, val_ids, val_sources = load_npz(val_file)
    if len(val_y) != 16 or int(np.sum(val_y)) != 3 or not np.all(val_sources == 'real'):
        raise RuntimeError('仅允许16家真实验证企业（3风险）')
    modes = ('generated_only',) if args.smoke else MODES
    run_names = {m: ('generated_only_s1_smoke' if args.smoke else f'{m}_s1_2027')
                 for m in modes}
    out = Path(args.output or (
        'LSTM/outputs/reports/validation_smoke_s1_2027' if args.smoke else
        'LSTM/outputs/reports/validation_s1_2027'))
    if out.exists():
        raise FileExistsError(f'报告目录已存在，拒绝覆盖：{out}')
    # Inspect all required paths before allocating GPU or creating any outputs.
    for mode, run in run_names.items():
        for path in (Path('LSTM/outputs/checkpoints') / run / 'best.pt',
                     Path('LSTM/outputs/logs') / run / 'history.csv',
                     Path('LSTM/outputs/logs') / run / 'run_info.json'):
            if not path.is_file():
                raise FileNotFoundError(f'缺少已完成训练文件：{path}')
    device = physical_gpu1()
    data = {}
    histories = {}
    provenance = {'manifest_sha256': sha256_file(manifest_path),
                  'val_npz_sha256': sha256_file(val_file),
                  'checkpoint_sha256': {}, 'run_infos': {},
                  'smoke_only': args.smoke, 'test_read': False}
    for mode, run in run_names.items():
        ckpt_path = Path('LSTM/outputs/checkpoints') / run / 'best.pt'
        info_path = Path('LSTM/outputs/logs') / run / 'run_info.json'
        history_path = Path('LSTM/outputs/logs') / run / 'history.csv'
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        info = json.loads(info_path.read_text(encoding='utf-8'))
        rows = load_history(history_path)
        if checkpoint.get('mode') != mode or checkpoint.get('test_used') is not False:
            raise RuntimeError(f'模式或测试标记错误：{ckpt_path}')
        if checkpoint['manifest_sha256'] != sha256_file(manifest_path):
            raise RuntimeError(f'预处理清单变化：{mode}')
        if checkpoint['train_npz_sha256'] != sha256_file(base / f'train_{mode}.npz'):
            raise RuntimeError(f'训练源数据变化：{mode}')
        if checkpoint['val_npz_sha256'] != sha256_file(val_file):
            raise RuntimeError(f'验证源数据变化：{mode}')
        if info['epochs_completed'] != len(rows):
            raise RuntimeError(f'真实训练轮数与history不一致：{run}')
        if int(checkpoint['best_epoch']) != int(info['best_epoch']):
            raise RuntimeError(f'最优轮次记录不匹配：{run}')
        if float(checkpoint['threshold']) != float(cfg['training']['threshold']):
            raise RuntimeError(f'阈值与当前训练配置不一致：{run}')
        model = MultiHeadFinancialLSTM(**checkpoint['model_config']).to(device)
        model.load_state_dict(checkpoint['model_state'], strict=True)
        p, val_bce = inference(model, val_x, val_y,
                               int(cfg['training']['batch_size']), device)
        if not np.isclose(val_bce, checkpoint['best_val_macro_bce'], atol=1e-5, rtol=1e-5):
            raise RuntimeError(f'最佳checkpoint与验证BCE不符：{run} {val_bce}')
        data[mode] = dict(y=val_y.copy(), p=p, ids=val_ids.copy(),
                          threshold=float(checkpoint['threshold']))
        histories[mode] = rows
        provenance['checkpoint_sha256'][mode] = sha256_file(ckpt_path)
        provenance['run_infos'][mode] = info
        print(f'{mode}: 最佳轮次={info["best_epoch"]}，'
              f'共训练={info["epochs_completed"]}，验证宏平均BCE={val_bce:.5f}')
    build_report(out, 'smoke_validation' if args.smoke else 'validation',
                 data, histories, args.bootstrap_repeats, 2026, provenance)


if __name__ == '__main__':
    main()
