import csv
import json
from pathlib import Path

import numpy as np
import pytest

from LSTM.src.evaluation_reporting import (inspect_predictions, extended_metrics,
                                           stratified_bootstrap, build_report)
from LSTM.src.metrics import binary_metrics
from LSTM.scripts.report_test import main as run_test_report_main


def toy_data():
    y = np.asarray([1, 1, 1] + [0] * 13)
    p = np.asarray([0.90, 0.65, 0.20] + list(np.linspace(0.04, 0.53, 13)))
    ids = [f'company_{i}' for i in range(16)]
    return y, p, ids


def test_extended_metrics_consistent_with_existing_metrics():
    y, p, _ = toy_data()
    m = extended_metrics(y, p)
    reference = binary_metrics(y, p)
    for key, val in reference.items():
        assert m[key] == pytest.approx(val)
    assert m['prevalence'] == pytest.approx(3 / 16)
    assert m['majority_class_accuracy'] == pytest.approx(13 / 16)
    assert 0 <= m['brier_score'] <= 1
    assert 0 <= m['ece_5_bins'] <= 1


def test_bootstrap_fixed_and_company_level():
    y, p, _ = toy_data()
    a = stratified_bootstrap(y, p, draws=50, seed=2026)
    b = stratified_bootstrap(y, p, draws=50, seed=2026)
    assert a == b
    assert len(a) == 5
    assert all(row['percentile_2p5'] <= row['percentile_97p5'] for row in a)


def test_inconsistent_ids_rejected_before_report_created(tmp_path):
    y, p, ids = toy_data()
    bad_ids = ids.copy()
    bad_ids[1] = bad_ids[0]
    with pytest.raises(ValueError, match='重复'):
        inspect_predictions(y, p, bad_ids)
    out = tmp_path / 'bad_report'
    with pytest.raises(RuntimeError, match='ID或真实标签不一致'):
        build_report(out, 'validation', {
            'real_only': dict(y=y, p=p, ids=ids, threshold=.5),
            'generated_only': dict(y=y[::-1], p=p, ids=ids, threshold=.5),
        }, bootstrap_repeats=50)
    assert not out.exists()


def test_report_files_are_created_and_never_overwritten(tmp_path):
    y, p, ids = toy_data()
    out = tmp_path / 'reports'
    result = build_report(out, 'smoke_validation',
                          {'generated_only': dict(y=y, p=p, ids=ids, threshold=.5)},
                          history_by_mode={'generated_only': [dict(
                              epoch=1, train_bce=.7, val_macro_bce=.69,
                              val_auroc=.6, val_pr_auc=.25,
                              val_macro_f1=.5, val_bal_acc=.5)]},
                          bootstrap_repeats=50)
    assert len(result) == 1
    for name in ('metrics_comparison.csv', 'bootstrap_intervals.csv',
                 'generated_only_predictions.csv', 'generated_only_metrics.json',
                 'generated_only_roc.png', 'generated_only_pr.png',
                 'generated_only_confusion.png',
                 'generated_only_probability_histogram.png',
                 'generated_only_history_train_bce.png',
                 'evaluation_report.md', 'report_manifest.json'):
        assert (out / name).is_file(), name
    with pytest.raises(FileExistsError):
        build_report(out, 'smoke_validation',
                     {'generated_only': dict(y=y, p=p, ids=ids, threshold=.5)},
                     bootstrap_repeats=50)


def test_offline_test_report_uses_existing_prediction_files_only(tmp_path, monkeypatch):
    y, p, ids = toy_data()
    ev = tmp_path / 'previous_test_evaluation'
    ev.mkdir()
    modes = ('real_only', 'generated_only', 'real_plus_s1')
    (ev / 'evaluation_manifest.json').write_text(json.dumps({
        'all_models_pretrained_and_selected_on_validation': True,
        'test_used_for_training_or_checkpoint_selection': False,
        'test_company_count': 16, 'risk_company_count': 3,
        'methods': list(modes)
    }), encoding='utf-8')
    with (ev / 'test_comparison.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=('mode', 'auroc', 'pr_auc',
                                         'balanced_accuracy', 'f1_risk'), extrasaction='ignore')
        w.writeheader()
        for mode in modes:
            m = binary_metrics(y, p, .5)
            w.writerow({'mode': mode, **m})
            (ev / f'{mode}_metrics.json').write_text(
                json.dumps({'threshold': .5, **m}), encoding='utf-8')
            with (ev / f'{mode}_company_predictions.csv').open(
                    'w', encoding='utf-8-sig', newline='') as f2:
                w2 = csv.DictWriter(f2, fieldnames=('company_id', 'ground_truth_risk',
                                                    'risk_probability', 'predicted_risk', 'source'))
                w2.writeheader()
                for company, label, prob in zip(ids, y, p):
                    w2.writerow(dict(company_id=company,
                                     ground_truth_risk=int(label),
                                     risk_probability=float(prob),
                                     predicted_risk=int(prob >= .5),
                                     source='real_test'))
    import sys
    out = tmp_path / 'test_report'
    monkeypatch.setattr(sys, 'argv', ['report_test', '--evaluation', str(ev),
                                     '--output', str(out), '--bootstrap-repeats', '50'])
    run_test_report_main()
    assert (out / 'metrics_comparison.csv').exists()
    assert len(list(csv.DictReader((out / 'metrics_comparison.csv').open(encoding='utf-8-sig')))) == 3
    saved = json.loads((out / 'report_manifest.json').read_text(encoding='utf-8'))
    assert saved['provenance']['test_xlsx_reloaded'] is False
    assert saved['provenance']['test_threshold_tuned'] is False


def test_validation_report_from_best_checkpoint_without_test_split(tmp_path, monkeypatch):
    """Validate the full smoke report using a synthetic CPU-only test fixture.

    Server execution itself still enforces physical GPU1: CPU is permitted here
    solely because this is an isolated unit test with mocked device selection.
    """
    import sys
    import torch
    import yaml
    from LSTM.src.common import sha256_file
    from LSTM.src.model import MultiHeadFinancialLSTM
    from LSTM.scripts.train import inference
    from LSTM.src.metrics import binary_metrics
    from LSTM.scripts.report_validation import main as run_validation_report

    monkeypatch.chdir(tmp_path)
    val_dir = tmp_path / 'fake_processed'
    val_dir.mkdir()
    rng = np.random.default_rng(17)
    x_val = rng.normal(size=(16, 20, 15)).astype(np.float32)
    y_val = np.asarray([1, 1, 1] + [0] * 13, np.int64)
    ids_val = np.asarray([f'real_{i}' for i in range(16)])
    np.savez_compressed(val_dir/'validation.npz', x=x_val, y=y_val,
                        sample_id=ids_val, source=np.asarray(['real']*16))
    np.savez_compressed(val_dir/'train_generated_only.npz',
                        x=rng.normal(size=(200,20,15)).astype(np.float32),
                        y=np.asarray([0]*100+[1]*100),
                        sample_id=np.asarray([f's1_{i}' for i in range(200)]),
                        source=np.asarray(['generated']*200))
    (val_dir/'manifest.json').write_text(json.dumps({'test_read': False}), encoding='utf-8')
    cfg = {'project': {'dataset': str(val_dir)},
           'training': {'threshold': .5, 'batch_size': 16}}
    config = tmp_path/'config.yaml'
    config.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    torch.manual_seed(5)
    model = MultiHeadFinancialLSTM(n_features=15, seq_len=20, hidden_size=3,
                                   fusion_hidden=8, dropout=0.0)
    _, best_bce = inference(model, x_val, y_val, 16, torch.device('cpu'))
    preds, _ = inference(model, x_val, y_val, 16, torch.device('cpu'))
    scores = binary_metrics(y_val, preds)
    name = 'generated_only_s1_smoke'
    ckpt_dir = tmp_path/'LSTM'/'outputs'/'checkpoints'/name
    log_dir = tmp_path/'LSTM'/'outputs'/'logs'/name
    ckpt_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    torch.save({'model_state': model.state_dict(), 'model_config': model.config,
                'mode': 'generated_only', 'test_used': False,
                'best_epoch': 1, 'best_val_macro_bce': best_bce,
                'threshold': .5,
                'manifest_sha256': sha256_file(val_dir/'manifest.json'),
                'train_npz_sha256': sha256_file(val_dir/'train_generated_only.npz'),
                'val_npz_sha256': sha256_file(val_dir/'validation.npz')},
               ckpt_dir/'best.pt')
    (log_dir/'run_info.json').write_text(json.dumps({
        'mode': 'generated_only', 'epochs_completed': 2,
        'best_epoch': 1, 'best_val_macro_bce': best_bce}), encoding='utf-8')
    rows = [dict(epoch=ep, train_bce=.7,
                 val_macro_bce=best_bce+(.01 if ep==2 else 0),
                 val_auroc=scores['auroc'], val_pr_auc=scores['pr_auc'],
                 val_macro_f1=scores['macro_f1'],
                 val_bal_acc=scores['balanced_accuracy']) for ep in (1, 2)]
    with (log_dir/'history.csv').open('w', encoding='utf-8', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    out = tmp_path/'report'
    monkeypatch.setattr('LSTM.scripts.report_validation.physical_gpu1',
                        lambda: torch.device('cpu'))
    monkeypatch.setattr(sys, 'argv', ['report_validation', '--config', str(config),
                                     '--smoke', '--output', str(out),
                                     '--bootstrap-repeats', '50'])
    run_validation_report()
    assert (out/'generated_only_history_train_bce.png').is_file()
    assert (out/'generated_only_predictions.csv').is_file()
    result = json.loads((out/'report_manifest.json').read_text(encoding='utf-8'))
    assert result['split'] == 'smoke_validation'
    assert result['provenance']['run_infos']['generated_only']['epochs_completed'] == 2
    assert result['provenance']['test_read'] is False
