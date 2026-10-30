"""Postprocess one-time test predictions already written by LSTM.scripts.evaluate.

Never loads data/splits/test.xlsx, weights, or GPU. No checkpoint selection.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from LSTM.src.common import sha256_file
from LSTM.src.evaluation_reporting import build_report
from LSTM.src.metrics import binary_metrics
from LSTM.scripts.train import MODES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation',
                        default='LSTM/outputs/evaluation/s1_medianfix_seed2027_comparison')
    parser.add_argument('--output', default='LSTM/outputs/reports/test_s1_2027')
    parser.add_argument('--bootstrap-repeats', type=int, default=1000)
    args = parser.parse_args()
    if not 50 <= args.bootstrap_repeats <= 10000:
        raise ValueError('bootstrap次数必须在50到10000之间')
    origin, dst = Path(args.evaluation), Path(args.output)
    if dst.exists():
        raise FileExistsError(f'报告目录已存在，拒绝覆盖：{dst}')
    manifest_file = origin / 'evaluation_manifest.json'
    manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
    if (manifest.get('all_models_pretrained_and_selected_on_validation') is not True or
        manifest.get('test_used_for_training_or_checkpoint_selection') is not False or
        manifest.get('test_company_count') != 16 or
        manifest.get('risk_company_count') != 3 or
        set(manifest.get('methods', [])) != set(MODES)):
        raise RuntimeError('正式test评价清单不完整或训练/测试隔离错误')
    origin_metrics = {}
    records_file = origin / 'test_comparison.csv'
    with records_file.open(newline='', encoding='utf-8-sig') as f:
        compare = {r['mode']: r for r in csv.DictReader(f)}
    if set(compare) != set(MODES):
        raise RuntimeError('test比较表的模型不完整')
    data = {}
    hashes = {'evaluation_manifest.json': sha256_file(manifest_file),
              'test_comparison.csv': sha256_file(records_file)}
    reference_ids = None
    reference_y = None
    for mode in MODES:
        metrics_path = origin / f'{mode}_metrics.json'
        csv_path = origin / f'{mode}_company_predictions.csv'
        with csv_path.open(newline='', encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        if len(rows) != 16 or any(r['source'] != 'real_test' for r in rows):
            raise RuntimeError(f'仅允许16家独立test企业的已存储预测：{mode}')
        ids = [r['company_id'] for r in rows]
        y = np.asarray([int(r['ground_truth_risk']) for r in rows], dtype=int)
        p = np.asarray([float(r['risk_probability']) for r in rows], dtype=float)
        expected = np.asarray([int(r['predicted_risk']) for r in rows], dtype=int)
        if np.sum(y) != 3 or set(y.tolist()) != {0, 1}:
            raise RuntimeError(f'test标签或数量异常：{mode}')
        if reference_ids is None:
            reference_ids, reference_y = ids, y
        elif ids != reference_ids or not np.array_equal(y, reference_y):
            raise RuntimeError('三组test结果的企业顺序或真实标签不同')
        original = json.loads(metrics_path.read_text(encoding='utf-8'))
        threshold = float(original['threshold'])
        if not np.array_equal(expected, (p >= threshold).astype(int)):
            raise RuntimeError(f'预测标签与记录的阈值不匹配：{mode}')
        checks = binary_metrics(y, p, threshold)
        for key, value in checks.items():
            if isinstance(value, (float, int)) and original.get(key) is not None:
                if not np.isclose(value, original[key], rtol=1e-7, atol=1e-7):
                    raise RuntimeError(f'原始test指标不一致：{mode} {key}')
        for key in ('auroc', 'pr_auc', 'balanced_accuracy', 'f1_risk'):
            if not np.isclose(float(compare[mode][key]), checks[key], atol=1e-7):
                raise RuntimeError(f'test汇总表指标不一致：{mode} {key}')
        data[mode] = dict(ids=ids, y=y, p=p, threshold=threshold)
        origin_metrics[mode] = original
        hashes[f'{mode}_metrics.json'] = sha256_file(metrics_path)
        hashes[f'{mode}_company_predictions.csv'] = sha256_file(csv_path)
    build_report(dst, 'test', data, None, args.bootstrap_repeats, 2026,
                 {'source_evaluation': str(origin), 'source_hashes': hashes,
                  'original_test_manifest': manifest,
                  'test_xlsx_reloaded': False, 'test_inference_repeated': False,
                  'test_threshold_tuned': False})
    print('仅根据原始test预测CSV离线出具报告；没有重新读取test.xlsx。')


if __name__ == '__main__':
    main()
