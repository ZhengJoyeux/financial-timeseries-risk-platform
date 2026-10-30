"""Company-level input, train-only scaler, and provenance verification."""
import json
from pathlib import Path
import numpy as np
from VAE.src.dataset import FinancialScaler, read_financial
from VAE.src.schema import KEYS, LABELS
from LSTM.src.common import sha256_file, json_save_exclusive

MAP = {'无风险': 0, '有风险': 1}
EXPECTED = {'train': (48, 10), 'validation': (16, 3), 'test': (16, 3)}


def read_real_split(path, manifest, split):
    ids, x, labels = read_financial(path)
    exp_n, exp_risk = EXPECTED[split]
    if len(ids) != exp_n or sum(labels == '有风险') != exp_risk:
        raise RuntimeError(f'{split}数量或类别不符合固定划分：{len(ids)}，risk={sum(labels=="有风险")}')
    target_ids = set(manifest['splits'][split]['company_ids'])
    if set(ids.tolist()) != target_ids:
        raise RuntimeError(f'{split} 企业编号与split_manifest不一致，停止避免数据泄漏')
    return ids, x, labels


def read_s1_generated(root, train_sha):
    root = Path(root)
    ev = Path('VAE/outputs/evaluation') / root.name
    gate_path = ev / 'comparison_gate.json'
    provenance_path = ev / 'generation_provenance.json'
    if not gate_path.is_file() or not provenance_path.is_file():
        raise FileNotFoundError('S1缺少正式quality gate或provenance，禁止用于训练')
    gate = json.loads(gate_path.read_text(encoding='utf-8'))
    prov = json.loads(provenance_path.read_text(encoding='utf-8'))
    if gate.get('overall_ready_for_downstream') is not True:
        raise RuntimeError('生成数据未通过S0/S1正式质量门槛')
    if prov.get('train_sha256') != train_sha or prov.get('test_never_read') is not True:
        raise RuntimeError('生成数据训练源哈希不匹配或出现test读取风险')
    if prov.get('class') != 'non_neural_statistical_baseline':
        raise RuntimeError('生成来源不是已验证的S1统计生成器')
    if prov.get('generated_counts') != {'risk': 100, 'normal': 100}:
        raise RuntimeError('S1生成数量并非risk100+normal100')
    parsed = []
    all_ids = []
    for name, label in [('risk', '有风险'), ('normal', '无风险')]:
        path = root / f'generated_{name}_100.xlsx'
        ids, x, y = read_financial(path)
        if len(ids) != 100 or not np.all(y == label):
            raise RuntimeError(f'生成数据数量或标签错误：{path}')
        parsed.append((ids, x, y, path))
        all_ids.extend(f's1_{name}_{i}' for i in ids)
    if len(set(all_ids)) != len(all_ids):
        raise RuntimeError('同一份生成数据的企业标识重复')
    # Check combined export is present, but use per-class files as authoritative.
    if not (root / 'generated_all.xlsx').is_file():
        raise RuntimeError('缺少generated_all.xlsx')
    ids = np.asarray(all_ids, dtype='U64')
    x = np.concatenate([p[1] for p in parsed], axis=0)
    y = np.concatenate([p[2] for p in parsed], axis=0)
    files = {str(p[3]): sha256_file(p[3]) for p in parsed}
    files[str(root / 'generated_all.xlsx')] = sha256_file(root / 'generated_all.xlsx')
    return ids, x, y, files, prov


def save_npz(path, x, labels, sample_ids, source):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'拒绝覆盖：{path}')
    if x.ndim != 3 or x.shape[1:] != (20, len(KEYS)) or not np.isfinite(x).all():
        raise RuntimeError(f'错误财务张量：{x.shape}')
    np.savez_compressed(path, x=np.asarray(x, np.float32),
                        y=np.asarray([MAP[l] for l in labels], np.int64),
                        sample_id=np.asarray(sample_ids, dtype='U64'),
                        source=np.asarray(source, dtype='U16'))


def load_npz(path):
    with np.load(path, allow_pickle=False) as d:
        x, y, ids, src = (d[k].copy() for k in ('x','y','sample_id','source'))
    if x.ndim != 3 or x.shape[1:] != (20, len(KEYS)):
        raise RuntimeError('LSTM输入张量应为(N,20,15)')
    if len(np.unique(ids)) != len(ids):
        raise RuntimeError('sample_id重复，停止')
    if len(x) != len(y) or not np.isfinite(x).all() or not set(np.unique(y)).issubset({0,1}):
        raise RuntimeError('NPZ数量、标签或数值异常')
    return x, y, ids, src


def prepare(root, generated, output):
    root = Path(root)
    generated = Path(generated)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'拒绝覆盖已存在预处理目录：{output}')
    split_dir = root / 'data/splits'
    manifest_path = split_dir / 'split_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    train_path, val_path = split_dir/'train.xlsx', split_dir/'validation.xlsx'
    train_ids, train_x, train_labels = read_real_split(train_path, manifest, 'train')
    val_ids, val_x, val_labels = read_real_split(val_path, manifest, 'validation')
    if set(train_ids) & set(val_ids):
        raise RuntimeError('train/validation企业存在重叠')
    scaler = FinancialScaler().fit(train_x)  # absolutely no generated/val/test fit
    train_sha = sha256_file(train_path)
    gen_ids, gen_x, gen_labels, gen_files, prov = read_s1_generated(generated, train_sha)
    dst = output
    dst.mkdir(parents=True, exist_ok=False)
    real_ids = np.asarray([f'real_{int(i)}' for i in train_ids], dtype='U64')
    val_sample_ids = np.asarray([f'real_{int(i)}' for i in val_ids], dtype='U64')
    real_z = scaler.transform(train_x)
    gen_z = scaler.transform(gen_x)
    save_npz(dst/'train_real_only.npz', real_z, train_labels, real_ids, ['real']*len(real_ids))
    save_npz(dst/'train_generated_only.npz', gen_z, gen_labels, gen_ids, ['generated']*len(gen_ids))
    save_npz(dst/'train_real_plus_s1.npz', np.concatenate([real_z,gen_z]),
             np.concatenate([train_labels,gen_labels]), np.concatenate([real_ids,gen_ids]),
             ['real']*len(real_ids)+['generated']*len(gen_ids))
    save_npz(dst/'validation.npz', scaler.transform(val_x), val_labels, val_sample_ids,
             ['real']*len(val_sample_ids))
    json_save_exclusive(dst/'scaler.json', scaler.state_dict())
    summary = {
        'source': 'S1 repaired medianfix output, company-level data, 15 heads',
        'train_xlsx_sha256': train_sha,
        'validation_xlsx_sha256': sha256_file(val_path),
        'split_manifest_sha256': sha256_file(manifest_path),
        'generated_xlsx_sha256': gen_files,
        's1_provenance': prov,
        'feature_keys': KEYS,
        'sequence_length': 20,
        'feature_count': len(KEYS),
        'label_mapping': MAP,
        'test_read': False,
        'train_sizes': {'real_only': len(real_z), 'generated_only': len(gen_z),
                        'real_plus_s1': len(real_z)+len(gen_z)},
        'validation_size': len(val_x),
        'validation_risk': int(np.sum(val_labels == '有风险')),
    }
    json_save_exclusive(dst/'manifest.json', summary)
    print('LSTM 输入数据准备完成（未读取 test）:', dst)
    print('train真实48，生成200，合并248；validation真实16')
    return summary
