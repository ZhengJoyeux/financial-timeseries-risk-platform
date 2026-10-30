"""Read-only, post-hoc quality audit of S1's saved inner/outer CV generations.

Run from '/home/wqzheng/Financial platform' in conda sers_ddpm:
    python -m LSTM.scripts.audit_foldwise_s1

Only reads data/splits/train.xlsx, not validation or test. Doesn't fit S1 or train.
New reports are written exclusively under LSTM/outputs/diagnostics/.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import wasserstein_distance

from VAE.src.dataset import read_financial, FinancialScaler
from VAE.src.finance_rules import violation_rates
from VAE.src.schema import KEYS
from LSTM.src.trend_cv import fold_indices, inner_indices

SOURCE_DIR = Path('LSTM/outputs/cross_validation/trend_v2_s1_foldwise_nested5_seed2026')
REPORT_DIR = Path('LSTM/outputs/diagnostics/s1_foldwise_quality_audit_seed2026')
PHASES = ('inner', 'outer')
CLASS_LABELS = (('risk', 1, '有风险'), ('normal', 0, '无风险'))


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def fail_unless(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def check_identity(protocol, train_ids, y, sourcehash, *, expected_hash=None):
    """Verify protocol against the train file; no validation/test file access."""
    fail_unless(len(train_ids) == 48 and len(set(map(int, train_ids))) == 48,
                '训练企业数量/ID不是48个唯一企业')
    fail_unless(int(np.sum(y == 1)) == 10 and int(np.sum(y == 0)) == 38,
                '训练集应包含10风险、38正常企业')
    fail_unless(protocol.get('train_sha256') == sourcehash,
                '训练文件SHA256与原正式实验protocol不一致，停止审计')
    if expected_hash:
        fail_unless(expected_hash == sourcehash, '训练文件与明确指定的SHA256不一致')
    fail_unless(protocol.get('read_test') is False and protocol.get('read_validation') is False,
                '原实验protocol未标注禁止读取validation/test')
    fail_unless(protocol.get('pre_generated_full_train_S1_used') is False,
                '原实验protocol未标注禁止使用全量旧S1')
    fail_unless(protocol.get('smoke') is False and protocol.get('n_outer_folds') == 5,
                '不是正式嵌套五折实验')
    fail_unless(int(protocol.get('folds_executed', -1)) == 5,
                '正式实验必须有5折')
    fail_unless(int(protocol.get('generated_per_class', -1)) == 100,
                '当前质量审计针对每阶段每类100家')
    fail_unless(protocol.get('train_file') == 'data/splits/train.xlsx',
                'protocol的训练输入路径不是固定训练Excel')
    fail_unless(protocol.get('S1_fit', '').startswith('once per phase'),
                'protocol缺少S1逐阶段单独拟合声明')
    fail_unless(protocol.get('scaler_fit', '').startswith('only current'),
                'protocol缺少逐阶段scaler拟合声明')
    return True


def _int_set(values, name):
    seq = [int(v) for v in values]
    fail_unless(len(seq) == len(set(seq)), f'{name}包含重复ID')
    return set(seq)


def audit_provenance(prov, expected_fit, expected_forbidden, all_train_ids, *, fold, phase, sourcehash, n):
    """Check all stored provenance and reconstructed splits, not just boolean claims."""
    fit = _int_set(prov['fit_enterprise_ids'], 'fit_enterprise_ids')
    forbidden = _int_set(prov['forbidden_enterprise_ids'], 'forbidden_enterprise_ids')
    expect_fit = {int(i) for i in expected_fit}
    expect_forbidden = {int(i) for i in expected_forbidden}
    fail_unless(fit == expect_fit, 'S1拟合ID与重新计算的划分不一致')
    fail_unless(forbidden == expect_forbidden, 'S1禁止ID与重新计算的划分不一致')
    fail_unless(not fit.intersection(forbidden), 'S1拟合企业与禁止企业重叠')
    fail_unless(fit.union(forbidden).issubset(set(map(int, all_train_ids))),
                'S1溯源记录出现训练集以外企业')
    fail_unless(_int_set(prov['real_fit_ids'], 'real_fit_ids') == fit,
                'real_fit_ids与fit_enterprise_ids不一致')
    fail_unless(_int_set(prov['other_ids'], 'other_ids') == forbidden,
                'other_ids与forbidden_enterprise_ids不一致')
    fail_unless(prov['real_fit_sha256'] == sourcehash, '阶段记录训练Excel SHA256不一致')
    fail_unless(int(prov['fold']) == fold and prov['phase'] == phase,
                '阶段记录fold/phase错误')
    fail_unless(int(prov['generated_per_class']) == n, '阶段记录生成数量不一致')
    fail_unless(prov.get('S1_device') == 'cuda:0', '正式S1生成设备不是锁定的逻辑cuda:0')
    fail_unless(prov.get('never_used_validation_or_test') is True,
                '溯源记录未声明禁止读取独立validation/test')
    fail_unless(prov.get('reference_source', '').startswith('ONLY fold-fit'),
                '溯源记录缺少仅折内训练数据声明')
    return fit


def validate_npz(path, *, fold, phase, n):
    """allow_pickle=False: treats all saved NPZ contents as data, not code."""
    with np.load(path, allow_pickle=False) as data:
        require = {'x', 'y', 'source', 'origin_fold', 'origin_phase'}
        fail_unless(require.issubset(set(data.files)), f'{path} NPZ字段缺失')
        x = np.asarray(data['x'], dtype=np.float64)
        y = np.asarray(data['y'])
        source = np.asarray(data['source'])
        origin_fold = np.asarray(data['origin_fold'])
        origin_phase = np.asarray(data['origin_phase'])
    fail_unless(x.shape == (2 * n, 20, 15), '生成数据形状异常')
    fail_unless(np.isfinite(x).all(), '生成数据中出现NaN/Inf')
    fail_unless(y.shape == (2 * n,) and np.array_equal(y[:n], np.ones(n))
                and np.array_equal(y[n:], np.zeros(n)), '类别数量或顺序不符合S1正式实验')
    fail_unless(source.shape == (2*n,) and np.all(source == 'generated'),
                '样本source字段错误')
    fail_unless(origin_fold.shape == (2*n,) and np.all(origin_fold == fold),
                '样本origin_fold字段错误')
    fail_unless(origin_phase.shape == (2*n,) and np.all(origin_phase == phase),
                '样本origin_phase字段错误')
    return x, y


def _standardize_float64(x, scaler):
    return ((np.arcsinh(x/scaler.scale)-scaler.mean)/scaler.std).reshape(len(x), -1).astype(np.float64)


def _reference_std(x, scaler):
    # S1 reference's original implementation passes FinancialScaler.transform -> float32.
    return scaler.transform(x).reshape(len(x), -1).astype(np.float64)


def audit_class(x_gen, x_train_fit, *, scaler, novelty_fraction, diagnostics,
                name, fold, phase, exploratory_fallback_review=0.10):
    """Return reproducible metrics and 15 feature shifts. No model/GPU use."""
    x_gen = np.asarray(x_gen, dtype=np.float64)
    x_train_fit = np.asarray(x_train_fit, dtype=np.float64)
    n = len(x_gen)
    fail_unless(n == 100 and x_train_fit.shape[1:] == (20,15), '类别样本数/形状异常')
    fail_unless(len(x_train_fit) >= 3, '拟合企业少于3')
    rule_rates = violation_rates(x_gen)
    z_ref = _reference_std(x_train_fit, scaler)
    z_gen = _standardize_float64(x_gen, scaler)
    d = cdist(z_gen, z_ref)
    nearest = d.min(axis=1)
    tdist = cdist(z_ref, z_ref)
    np.fill_diagonal(tdist, np.inf)
    loo_nearest = tdist.min(axis=1)
    threshold = float(novelty_fraction * np.median(loo_nearest))
    recorded_threshold = float(diagnostics['nearest_neighbor_threshold_standardized_euclidean'])
    fail_unless(np.isclose(threshold, recorded_threshold, rtol=2e-5, atol=1e-5),
                '独立重算的新颖性阈值与S1记录不一致')
    # Tolerate tiny CPU/GPU FP differences at the threshold boundary.
    below = int(np.sum(nearest < threshold - 1e-6))
    fallback = int(diagnostics['quantile_slots_without_threshold_pass_fallback_to_max_distance'])
    fail_unless(0 <= fallback <= n, 'S1回退数量超过样本数')
    fail_unless(int(diagnostics['candidate_draws']) == n * 16, '采样候选预算不一致')
    fail_unless(int(diagnostics['stratified_quantile_slots']) == n, '分位槽数量不一致')
    fail_unless(int(diagnostics['accepted']) == n, '保存的接受数量不一致')
    fail_unless(abs(float(diagnostics['threshold_satisfaction_fraction'])
                    - (1 - fallback/n)) < 1e-8, '原诊断的阈值通过率不自洽')
    fail_unless(below <= fallback + 1, '实际未过阈值数超过回退槽数，存在审计不一致')
    fail_unless(np.isfinite(z_gen).all(), '标准化生成数据出现NaN/Inf')
    duplicate_generated = n - len(np.unique(x_gen.reshape(n, -1), axis=0))
    # Exact raw equality, not an arbitrary 'near copy' threshold.
    exact_train_copies = int(sum(any(np.array_equal(row, ref) for ref in x_train_fit)
                                 for row in x_gen))
    pair_gen = pdist(z_gen)
    pair_ref = pdist(z_ref)
    pair_ratio = float(np.median(pair_gen) / max(np.median(pair_ref), 1e-10))
    # Enterprise-level QoQ asset motion, not falsely counting 20 quarters as 20 enterprises.
    def qoq_stats(arr):
        a = arr[..., KEYS.index('total_assets')]
        return np.abs(np.diff(np.log(np.maximum(a, 1e-10)), axis=1))
    q_gen = qoq_stats(x_gen)
    q_ref = qoq_stats(x_train_fit)
    p95_gen = float(np.quantile(q_gen, .95))
    p95_ref = float(np.quantile(q_ref, .95))
    # Additional descriptive distribution features, intentionally not acceptance gates.
    sgen = scaler.transform(x_gen).astype(np.float64).mean(axis=1)
    sref = scaler.transform(x_train_fit).astype(np.float64).mean(axis=1)
    features = []
    for k, feature in enumerate(KEYS):
        ref_feature = sref[:, k]
        gen_feature = sgen[:, k]
        iqr = max(float(np.quantile(ref_feature, .75)-np.quantile(ref_feature, .25)), 0.25)
        features.append({
            'fold':fold, 'phase':phase, 'class':name, 'feature':feature,
            'fit_enterprises':len(x_train_fit), 'generated_enterprises':n,
            'real_enterprise_mean_median_std_units':float(np.median(ref_feature)),
            'generated_enterprise_mean_median_std_units':float(np.median(gen_feature)),
            'wasserstein_standardized_iqr_units':float(wasserstein_distance(ref_feature,gen_feature)/iqr),
            'generated_below_real_range_fraction':float(np.mean(gen_feature < np.min(ref_feature))),
            'generated_above_real_range_fraction':float(np.mean(gen_feature > np.max(ref_feature))),
        })
    warnings = []
    if fallback/n > exploratory_fallback_review:
        warnings.append(f'S1 novelty回退比例{fallback/n:.1%}超过探索性复核线{exploratory_fallback_review:.0%}')
    if below > 0:
        warnings.append(f'{below}/{n}个生成样本实际低于新颖性阈值')
    if pair_ratio < .5 or pair_ratio > 2.0:
        warnings.append(f'样本两两距离比={pair_ratio:.3f}（探索性复核范围0.5~2.0）')
    if p95_ref > 0 and p95_gen > 3 * p95_ref:
        warnings.append(f'总资产季度log变动P95显著超过训练参考3倍({p95_gen:.3f} vs {p95_ref:.3f})')
    if any(rate > 0 for rate in rule_rates.values()):
        warnings.append('存在财务规则违背的季度（详见规则逐项统计）')
    if duplicate_generated or exact_train_copies:
        warnings.append('发现精确重复/复制样本')
    metrics = {
        'fold':fold,'phase':phase,'class':name,'fit_enterprises':len(x_train_fit),'generated_enterprises':n,
        'novelty_threshold_recomputed':threshold,'novelty_threshold_recorded':recorded_threshold,
        'fallback_slots_reported':fallback,'fallback_rate':fallback/n,
        'below_novelty_threshold_count':below,'below_novelty_threshold_rate':below/n,
        'novelty_min_nearest_fit':float(nearest.min()),
        'novelty_nearest_fit_median':float(np.median(nearest)),
        'novelty_nearest_fit_p05':float(np.quantile(nearest, .05)),
        'exact_duplicate_generated_count':int(duplicate_generated),
        'exact_fit_company_copy_count':exact_train_copies,
        'generated_pairwise_distance_median':float(np.median(pair_gen)),
        'real_pairwise_distance_median':float(np.median(pair_ref)),
        'pairwise_distance_ratio':pair_ratio,
        'assets_abs_log_qoq_p95_generated':p95_gen,
        'assets_abs_log_qoq_p95_real_fit':p95_ref,
        'max_financial_rule_violation_rate':float(max(rule_rates.values())),
        'financial_rule_violation_rates':rule_rates,
        'mean_feature_wasserstein_iqr_units':float(np.mean([f['wasserstein_standardized_iqr_units'] for f in features])),
        'warning_count':len(warnings),'warnings':warnings,
    }
    return metrics,features


def csv_exclusive(path, records):
    if not records:
        return
    with Path(path).open('x', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def run_audit(source_dir=SOURCE_DIR, output_dir=REPORT_DIR):
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f'审计结果目录已存在，拒绝覆盖：{output_dir}')
    fail_unless(source_dir.is_dir(), f'正式五折结果不存在：{source_dir}')
    # Only load train.xlsx; this script never opens validation.xlsx or test.xlsx.
    train_path = Path('data/splits/train.xlsx')
    fail_unless(train_path.is_file(), '固定训练文件data/splits/train.xlsx不存在')
    protocol = read_json(source_dir/'protocol.json')
    ids,raw,labels = read_financial(train_path)
    ids = np.asarray(ids,dtype=np.int64)
    raw = np.asarray(raw,dtype=np.float64)
    y = np.asarray([{'无风险':0,'有风险':1}[str(s)] for s in labels],dtype=int)
    sourcehash = sha256_file(train_path)
    check_identity(protocol, ids,y,sourcehash)
    folds = fold_indices(y,n_splits=5,seed=int(protocol['seed']))
    errors=[]; warnings=[]; group_rows=[]; feature_rows=[]
    for fold, (outer_fit_idx, outer_holdout_idx) in enumerate(folds, 1):
        inner_fit_idx,inner_stop_idx = inner_indices(outer_fit_idx,y,seed=int(protocol['seed'])+fold)
        for phase,fit_idx,forbidden_idx in (
            ('inner',inner_fit_idx,inner_stop_idx),
            ('outer',outer_fit_idx,outer_holdout_idx)):
            base = source_dir/'fold_generated'/f'fold{fold}_{phase}'
            try:
                prov = read_json(base/'provenance.json')
                audit_provenance(prov, ids[fit_idx], ids[forbidden_idx], ids,
                                 fold=fold,phase=phase,sourcehash=sourcehash,n=100)
                fail_unless(set(map(int,ids[fit_idx])).isdisjoint(set(map(int,ids[outer_holdout_idx]))),
                            '外层留出企业进入S1拟合')
                raw_gen,gen_y = validate_npz(base/'generated_raw.npz',fold=fold,phase=phase,n=100)
                fail_unless(len(np.unique(raw_gen.reshape(200,-1), axis=0)) == 200,
                            '整个阶段的两类生成样本存在跨类别或同类别精确重复')
                flat_fit = raw[fit_idx].reshape(len(fit_idx),-1)
                fail_unless(not any(np.array_equal(row, ref) for row in raw_gen.reshape(200,-1)
                                    for ref in flat_fit),
                            '生成样本与当前阶段任意类别训练企业精确一致')
                fail_unless(int(prov['fit_enterprises_per_class']['risk']) == int(np.sum(y[fit_idx]==1))
                            and int(prov['fit_enterprises_per_class']['normal']) == int(np.sum(y[fit_idx]==0)),
                            '保存的S1类别拟合数量不符')
                fail_unless(prov['parameters'] == protocol['S1_parameters'],
                            'S1原实验参数与阶段溯源不符')
                scaler = FinancialScaler().fit(raw[fit_idx])
                stored_scaler = prov['scaler_state']
                for key in ('scale','mean','std'):
                    fail_unless(np.allclose(getattr(scaler,key), stored_scaler[key], atol=1e-7,rtol=1e-7),
                                f'仅折内拟合的scaler参数与保存值不符：{key}')
                rules_all = violation_rates(raw_gen)
                for label,_,local_name in CLASS_LABELS:
                    reports = prov['raw_financial_rule_violations'][local_name]
                    for rule in rules_all:
                        # Check per-class rate below; this is only a key completeness check.
                        fail_unless(rule in reports, f'缺少记录的财务规则:{rule}')
                for group_name,binary,label in CLASS_LABELS:
                    sel_gen = raw_gen[gen_y==binary]
                    sel_fit = raw[fit_idx][y[fit_idx]==binary]
                    diagnostics = prov['novelty_sampler_diagnostics'][label]
                    row,rows = audit_class(sel_gen,sel_fit,scaler=scaler,
                                           novelty_fraction=float(prov['parameters']['novelty_fraction']),
                                           diagnostics=diagnostics,name=group_name,fold=fold,phase=phase)
                    reported_rules = prov['raw_financial_rule_violations'][label]
                    for rule,val in row['financial_rule_violation_rates'].items():
                        fail_unless(np.isclose(reported_rules[rule], val, rtol=0, atol=1e-12),
                                    f'{label}财务规则与保存的溯源不符:{rule}')
                    group_rows.append(row)
                    feature_rows.extend(rows)
                    for w in row['warnings']:
                        warnings.append(f'Fold{fold} {phase} {group_name}: {w}')
            except Exception as exc:
                errors.append(f'Fold{fold} {phase}: {type(exc).__name__}: {exc}')
    # Don't claim quality pass just because S1 was executed without exceptions.
    hard_failed_groups = [r for r in group_rows if (
        r['max_financial_rule_violation_rate'] > 1e-3
        or r['exact_duplicate_generated_count'] > 0
        or r['exact_fit_company_copy_count'] > 0
        or r['below_novelty_threshold_count'] > r['fallback_slots_reported'] + 1)]
    for r in hard_failed_groups:
        errors.append(f'Fold{r["fold"]} {r["phase"]} {r["class"]}: 硬性审计门槛未通过')
    if len(group_rows) != 20:
        errors.append(f'审计完成组数={len(group_rows)}，预期20组(5折×2阶段×2类)')
    status = 'FAIL' if errors else 'REVIEW_REQUIRED' if warnings else 'INTEGRITY_PASS_WITH_LIMITATIONS'
    summary = {
        'status':status, 'scope':'preexisting, fold-local S1 generated finance trajectories',
        'source_dir':str(source_dir),'train_sha256':sourcehash,'n_train_enterprises':48,
        'validation_and_test_files_opened':False,
        'n_groups_expected':20,'n_groups_checked':len(group_rows),
        'n_generated_trajectories_checked':int(sum(r['generated_enterprises'] for r in group_rows)),
        'hard_error_count':len(errors),'review_warning_count':len(warnings),
        'errors':errors,'warnings':warnings,
        'global_diagnostics':{
            'fallback_slots_total':int(sum(r['fallback_slots_reported'] for r in group_rows)),
            'novelty_below_threshold_total':int(sum(r['below_novelty_threshold_count'] for r in group_rows)),
            'exact_generated_duplicates_total':int(sum(r['exact_duplicate_generated_count'] for r in group_rows)),
            'exact_training_copies_total':int(sum(r['exact_fit_company_copy_count'] for r in group_rows)),
            'max_financial_rule_violation_rate':float(max((r['max_financial_rule_violation_rate'] for r in group_rows),default=0)),
        },
        'limitations':[
            'Only simulated enterprises; fold-fit risk sample n=6(inner)/8(outer).',
            'Provenance is checked against train.xlsx and saved fold indices; no proof about external accesses beyond audited source logic.',
            'Novelty threshold is nearest-company distance, not privacy guarantee or proof of non-memorization.',
            'Heuristic warning thresholds (fallback >10%, pairwise ratio outside 0.5..2, assets QoQ P95 >3x) are NOT validated scientific criteria.',
            'No external validation/test data read; distribution checks are descriptive, NOT a definitive generator quality pass.',
            'Original S1 rule gate allowed per-rule violation fractions up to 0.001; independent checks use the same tolerance.',
        ],
    }
    output_dir.parent.mkdir(parents=True,exist_ok=True)
    output_dir.mkdir(exist_ok=False)
    (output_dir/'audit_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    rows_for_csv=[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in group_rows]
    csv_exclusive(output_dir/'per_class_phase_metrics.csv',rows_for_csv)
    csv_exclusive(output_dir/'per_feature_distribution.csv',feature_rows)
    with (output_dir/'audit_report.md').open('x',encoding='utf-8') as f:
        f.write('# S1 折内生成质量审计（不重新训练）\n\n')
        f.write(f'状态：**{status}**；完成 {len(group_rows)}/20 组、检查 {summary["n_generated_trajectories_checked"]} 条阶段性生成轨迹。\n\n')
        f.write('输入：仅原始 train.xlsx 和既有五折 fold_generated 文件。不读取 validation/test。\n\n')
        f.write('## 关键统计\n\n')
        for key,value in summary['global_diagnostics'].items():
            f.write(f'- {key}: {value}\n')
        f.write('\n## 每折各阶段\n\n')
        f.write('| 折 | 阶段 | 类别 | 拟合家数 | 回退/100 | 实际低于阈值/100 | 两两距离比 | 财务最大违反率 | 精确复制 |\n')
        f.write('|---|---|---|---:|---:|---:|---:|---:|---:|\n')
        for r in group_rows:
            f.write(f'| {r["fold"]} | {r["phase"]} | {r["class"]} | {r["fit_enterprises"]} | {r["fallback_slots_reported"]} | {r["below_novelty_threshold_count"]} | {r["pairwise_distance_ratio"]:.3f} | {r["max_financial_rule_violation_rate"]:.5f} | {r["exact_fit_company_copy_count"]} |\n')
        f.write('\n## 硬性错误\n\n')
        for x in errors:f.write(f'- {x}\n')
        if not errors:f.write('无。\n')
        f.write('\n## 建议复核项（探索性阈值，不代表正式验收失败）\n\n')
        for x in warnings:f.write(f'- {x}\n')
        if not warnings:f.write('无预设复核警告；仍不能证明生成质量全面合格。\n')
        f.write('\n## 方法与限制\n\n')
        for x in summary['limitations']:f.write(f'- {x}\n')
        f.write('\n细项见 per_class_phase_metrics.csv 和 per_feature_distribution.csv；本审计不改变任何原实验文件。\n')
    print(f'审计状态：{status}，检查{len(group_rows)}/20组，硬性错误{len(errors)}，复核提醒{len(warnings)}。')
    print('报告：',output_dir/'audit_report.md')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=SOURCE_DIR,help='既有正式五折的只读结果目录')
    parser.add_argument('--output-dir',type=Path,default=REPORT_DIR,help='新建审计输出目录；拒绝覆盖')
    args=parser.parse_args()
    result=run_audit(args.run_dir,args.output_dir)
    if result['status']=='FAIL':
        raise SystemExit(2)

if __name__=='__main__':
    main()
