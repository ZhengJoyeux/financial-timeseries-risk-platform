"""Offline, company-level binary risk evaluation and report plotting.

The test-report stage consumes frozen predicted probabilities only. It never
loads test.xlsx, loads a checkpoint, or optimizes decision thresholds.
"""
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (confusion_matrix, precision_recall_curve,
                             roc_curve)
from LSTM.src.metrics import binary_metrics

METRIC_KEYS = ('n_companies', 'n_risk', 'accuracy', 'balanced_accuracy',
               'precision_risk', 'recall_risk', 'specificity', 'f1_risk',
               'macro_f1', 'micro_f1', 'auroc', 'pr_auc', 'ks',
               'binary_log_loss', 'brier_score', 'ece_5_bins',
               'prevalence', 'majority_class_accuracy')


def inspect_predictions(y, p, ids, threshold=0.5):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    ids = [str(v) for v in ids]
    if y.ndim != 1 or p.ndim != 1 or len(ids) != len(y) or len(y) != len(p):
        raise ValueError('预测数组维度和企业数量不匹配')
    if len(y) == 0 or set(np.unique(y)) != {0, 1}:
        raise ValueError('必须包含风险、无风险两类企业')
    if len(set(ids)) != len(ids):
        raise ValueError('评价企业ID重复')
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('预测概率必须是有限的[0,1]数值')
    if not 0 < threshold < 1:
        raise ValueError('决策阈值必须在(0,1)')
    return y, p, ids


def extended_metrics(y, p, threshold=0.5, bins=5):
    y, p, _ = inspect_predictions(y, p, [str(i) for i in range(len(y))], threshold)
    result = binary_metrics(y, p, threshold)
    result['brier_score'] = float(np.mean((p - y) ** 2))
    # ECE is descriptive only: 16 firms are not enough for reliable calibration.
    boundaries = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        inside = (p >= boundaries[i]) & (p <= boundaries[i + 1] if i == bins - 1 else p < boundaries[i + 1])
        if np.any(inside):
            ece += float(np.mean(inside) * abs(p[inside].mean() - y[inside].mean()))
    result['ece_5_bins'] = float(ece)
    result['prevalence'] = float(np.mean(y))
    result['majority_class_accuracy'] = float(max(np.mean(y), 1 - np.mean(y)))
    return result


def stratified_bootstrap(y, p, threshold=0.5, draws=1000, seed=2026):
    """Exploratory stratified bootstrap; NOT formal population-level inference."""
    y, p, _ = inspect_predictions(y, p, [str(i) for i in range(len(y))], threshold)
    rng = np.random.default_rng(seed)
    i0, i1 = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    names = ('auroc', 'pr_auc', 'balanced_accuracy', 'recall_risk', 'f1_risk')
    points = {name: [] for name in names}
    for _ in range(draws):
        idx = np.concatenate([rng.choice(i0, len(i0), replace=True),
                              rng.choice(i1, len(i1), replace=True)])
        score = binary_metrics(y[idx], p[idx], threshold)
        for name in names:
            points[name].append(score[name])
    current = binary_metrics(y, p, threshold)
    return [dict(metric=name, point_estimate=float(current[name]),
                 percentile_2p5=float(np.percentile(points[name], 2.5)),
                 percentile_97p5=float(np.percentile(points[name], 97.5)),
                 method='stratified_percentile_bootstrap_exploratory',
                 bootstrap_repeats=int(draws)) for name in names]


def csv_write_exclusive(path, header, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', newline='', encoding='utf-8-sig') as fp:
        w = csv.DictWriter(fp, fieldnames=header, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def json_write_exclusive(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as fp:
        json.dump(content, fp, ensure_ascii=False, indent=2, allow_nan=False)


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def plots_for_model(dst, mode, y, p, threshold=0.5):
    plt = _plt()
    dst = Path(dst)
    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(fpr, tpr, label=mode)
    ax.plot([0, 1], [0, 1], '--', label='Chance reference')
    ax.set(xlabel='False positive rate', ylabel='True positive rate', title=f'ROC - {mode}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(dst / f'{mode}_roc.png', dpi=160)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(y, p)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(recall, precision, label=mode)
    ax.axhline(float(np.mean(y)), linestyle='--', label='Risk prevalence')
    ax.set(xlim=(0, 1), ylim=(0, 1.04), xlabel='Recall', ylabel='Precision', title=f'PR - {mode}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(dst / f'{mode}_pr.png', dpi=160)
    plt.close(fig)

    cm = confusion_matrix(y, p >= threshold, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm)
    fig.colorbar(im, ax=ax)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm[i, j])), ha='center', va='center')
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Normal', 'Risk'],
           yticklabels=['Normal', 'Risk'], xlabel='Predicted', ylabel='True',
           title=f'Confusion matrix - {mode} (threshold={threshold:g})')
    fig.tight_layout()
    fig.savefig(dst / f'{mode}_confusion.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist([p[y == 0], p[y == 1]], bins=np.linspace(0, 1, 11),
            label=['Normal', 'Risk'], alpha=0.8)
    ax.axvline(threshold, linestyle='--', label='Fixed threshold')
    ax.set(xlabel='Predicted risk probability', ylabel='Number of companies',
           title=f'Score distribution - {mode}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(dst / f'{mode}_probability_histogram.png', dpi=160)
    plt.close(fig)


def plot_history(history, mode, dst):
    plt = _plt()
    rows = list(history)
    if not rows:
        raise ValueError('history为空')
    x = [int(r['epoch']) for r in rows]
    for field in ('train_bce', 'val_macro_bce', 'val_auroc',
                  'val_pr_auc', 'val_macro_f1', 'val_bal_acc'):
        series = [float(r[field]) for r in rows]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x, series)
        ax.set(xlabel='Epoch', ylabel=field, title=f'{mode} - {field}')
        fig.tight_layout()
        fig.savefig(Path(dst) / f'{mode}_history_{field}.png', dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, [float(r['train_bce']) for r in rows], label='Train BCE')
    ax.plot(x, [float(r['val_macro_bce']) for r in rows], label='Validation macro BCE')
    ax.set(xlabel='Epoch', ylabel='BCE', title=f'{mode} - train vs validation losses')
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(dst) / f'{mode}_history_loss_comparison.png', dpi=160)
    plt.close(fig)


def plot_comparison_curves(checked, dst):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mode, y, p, _, _ in checked:
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=mode)
    ax.plot([0, 1], [0, 1], '--', label='Chance reference')
    ax.set(xlabel='False positive rate', ylabel='True positive rate',
           title='Three-model ROC comparison')
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(dst) / 'comparison_roc_curves.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mode, y, p, _, _ in checked:
        precision, recall, _ = precision_recall_curve(y, p)
        ax.plot(recall, precision, label=mode)
    ax.axhline(np.mean(checked[0][1]), linestyle='--', label='Risk prevalence')
    ax.set(xlim=(0, 1), ylim=(0, 1.04), xlabel='Recall', ylabel='Precision',
           title='Three-model PR comparison')
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(dst) / 'comparison_pr_curves.png', dpi=160)
    plt.close(fig)


def plot_comparison(records, dst):
    plt = _plt()
    modes = [r['mode'] for r in records]
    for key in ('auroc', 'pr_auc', 'balanced_accuracy', 'f1_risk'):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(modes, [float(r[key]) for r in records])
        if key == 'pr_auc':
            ax.axhline(records[0]['prevalence'], linestyle='--',
                       label='Prevalence reference')
            ax.legend()
        ax.set(ylim=(0, 1.05), ylabel=key, title=f'Model comparison - {key}')
        ax.tick_params(axis='x', rotation=12)
        fig.tight_layout()
        fig.savefig(Path(dst) / f'comparison_{key}.png', dpi=160)
        plt.close(fig)


def build_report(output, split, data, history_by_mode=None,
                 bootstrap_repeats=1000, seed=2026, provenance=None):
    """Write posthoc reports for already frozen predictions. No source dataset opened."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'拒绝覆盖报告：{output}')
    if split not in {'validation', 'test', 'smoke_validation'}:
        raise ValueError(f'不允许的划分：{split}')
    if not data:
        raise ValueError('没有评价数据')
    checked = []
    for mode, item in data.items():
        y, p, ids = inspect_predictions(item['y'], item['p'], item['ids'], item['threshold'])
        checked.append((mode, y, p, ids, item['threshold']))
    # Across formal strategies, evaluation MUST refer to same people in same order.
    if split != 'smoke_validation':
        ref = [(s, v) for s, v in zip(checked[0][3], checked[0][1])]
        for entry in checked[1:]:
            if list(zip(entry[3], entry[1])) != ref:
                raise RuntimeError('跨模型评价企业ID或真实标签不一致')
    output.mkdir(parents=True, exist_ok=False)
    metric_rows = []
    ci_rows = []
    for mode, y, p, ids, threshold in checked:
        scores = extended_metrics(y, p, threshold)
        metric_rows.append({'mode': mode, **scores})
        ci_rows.extend([{'mode': mode, **row} for row in stratified_bootstrap(
            y, p, threshold, bootstrap_repeats, seed)])
        preds = (p >= threshold).astype(np.int64)
        csv_write_exclusive(output / f'{mode}_predictions.csv',
                            ['company_id', 'ground_truth_risk', 'risk_probability', 'predicted_risk'],
                            [dict(company_id=company, ground_truth_risk=int(label),
                                  risk_probability=float(prob), predicted_risk=int(pred))
                             for company, label, prob, pred in zip(ids, y, p, preds)])
        json_write_exclusive(output / f'{mode}_metrics.json', scores)
        plots_for_model(output, mode, y, p, threshold)
        if history_by_mode and mode in history_by_mode:
            plot_history(history_by_mode[mode], mode, output)
    csv_write_exclusive(output / 'metrics_comparison.csv', ['mode', *METRIC_KEYS], metric_rows)
    csv_write_exclusive(output / 'bootstrap_intervals.csv',
                        ['mode', 'metric', 'point_estimate', 'percentile_2p5', 'percentile_97p5',
                         'method', 'bootstrap_repeats'], ci_rows)
    if len(checked) > 1:
        plot_comparison(metric_rows, output)
        plot_comparison_curves(checked, output)
    reference = metric_rows[0]
    lines = [f'# 15头 Multi-head LSTM — {split} 评价报告', '',
             f'- 参与评价的企业数：**{reference["n_companies"]}**；风险企业：**{reference["n_risk"]}**',
             f'- 风险类基准比例（PR-AUC参考）：**{reference["prevalence"]:.4f}**',
             f'- 全预测为无风险时的Accuracy基准：**{reference["majority_class_accuracy"]:.4f}**',
             f'- Decision threshold: fixed from checkpoint at **{checked[0][4]:.4f}**, never tuned on test.',
             '- 最优checkpoint由独立真实验证集宏平均BCE选择，不根据test选择。', '',
             '| Model | AUROC | PR-AUC | Balanced accuracy | Risk recall | Risk F1 | Brier |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for r in metric_rows:
        lines.append(f'| {r["mode"]} | {r["auroc"]:.4f} | {r["pr_auc"]:.4f} | '
                     f'{r["balanced_accuracy"]:.4f} | {r["recall_risk"]:.4f} | '
                     f'{r["f1_risk"]:.4f} | {r["brier_score"]:.4f} |')
    lines.extend(['', '## 解读限制',
                  '- 数据来自**模拟财务企业**，不能等同于真实企业破产数据。',
                  '- 验证与测试各只有3家风险企业；错判1家会使风险Recall变动33.3个百分点。',
                  '- ROC/PR曲线和Bootstrap区间仅作探索性描述，不能证明真实企业泛化能力。',
                  '- 不能根据test比较结果再选择架构、阈值、checkpoint或生成数据比例。',
                  '- 三组对照只提供关联证据，不应据此作因果推断。',
                  '- Brier和ECE仅用于描述概率校准；样本仅16家时ECE不稳定。',
                  '', '## 输出文件',
                  '- metrics_comparison.csv: side-by-side metrics',
                  '- bootstrap_intervals.csv: exploratory 95% stratified bootstrap intervals',
                  '- *_predictions.csv: per-company probabilities and labels',
                  '- *_roc.png, *_pr.png, *_confusion.png, *_probability_histogram.png',
                  '- *_history_*.png: 训练曲线与Train/Validation loss叠加图（如果有history）',
                  '- report_manifest.json: input and provenance details', ''])
    with (output / 'evaluation_report.md').open('x', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    json_write_exclusive(output / 'report_manifest.json', {
        'split': split, 'modes': list(data), 'company_level': True,
        'feature_count': 15, 'sequence_length': 20,
        'trained_on_test': False, 'threshold_tuned_on_test': False,
        'bootstrap_repeats': bootstrap_repeats,
        'seed': seed, 'provenance': provenance or {},
        'warning': 'Simulated finance; risk n=3; small-sample reliability is limited.'})
    print(f'生成{split}评价报告：{output}，模型数={len(checked)}')
    return metric_rows
