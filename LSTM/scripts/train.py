"""Train independent 15-head LSTM, select checkpoint on REAL validation only."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import yaml
from LSTM.src.common import physical_gpu1, sha256_file, json_save_exclusive, seed_everything
from LSTM.src.data import load_npz
from LSTM.src.metrics import binary_metrics
from LSTM.src.model import MultiHeadFinancialLSTM

MODES = ('real_only', 'generated_only', 'real_plus_s1')


def build_sampler(y, source, mode, real_mass, seed):
    """Draw 50/50 classes; optionally allocate 50/50 real-vs-generated."""
    masses = {}
    if mode == 'real_plus_s1':
        for src, m in [('real', real_mass), ('generated', 1-real_mass)]:
            for label in (0,1):
                masses[(src,label)] = m / 2
    else:
        src = 'real' if mode == 'real_only' else 'generated'
        for label in (0,1):
            masses[(src,label)] = .5
    groups = list(zip(source.tolist(), y.tolist()))
    counts = {k:groups.count(k) for k in masses}
    if any(n==0 for n in counts.values()):
        raise RuntimeError('某个来源/类别为空，不能进行分层采样')
    weights = [masses[k]/counts[k] for k in groups]
    g = torch.Generator().manual_seed(seed)
    num_samples = max(len(y),64)
    return WeightedRandomSampler(torch.tensor(weights, dtype=torch.double),
                                 num_samples=num_samples, replacement=True, generator=g)


def class_macro_bce(logits, y):
    losses = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction='none')
    return (losses[y==0].mean() + losses[y==1].mean()) * 0.5


def inference(model, x, y, batch_size, device):
    model.eval()
    all_probs=[]
    with torch.inference_mode():
        for i in range(0,len(x),batch_size):
            xb = torch.from_numpy(x[i:i+batch_size]).to(device)
            prob = model(xb).sigmoid().detach().cpu().numpy()
            all_probs.extend(prob.tolist())
    probs = np.asarray(all_probs, dtype=np.float64)
    logits = np.log(np.clip(probs,1e-7,1-1e-7)/np.clip(1-probs,1e-7,1-1e-7))
    macro = float(0.5*(np.logaddexp(0, -logits[y==1]).mean() +
                       np.logaddexp(0, logits[y==0]).mean()))
    return probs, macro


def train_mode(config_path, mode, run_name, epochs_override=None):
    if mode not in MODES:
        raise ValueError('模式无效')
    cfg = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    if cfg['model']['n_features'] != 15 or cfg['model']['seq_len'] != 20:
        raise RuntimeError('必须为15变量、20季度')
    if cfg['training']['checkpoint_selection'] != 'val_macro_bce':
        raise RuntimeError('不支持使用test选checkpoint')
    device = physical_gpu1()
    seed_everything(cfg['project']['seed'])
    data_dir = Path(cfg['project']['dataset'])
    train_file = data_dir/f'train_{mode}.npz'
    val_file = data_dir/'validation.npz'
    manifest_path = data_dir/'manifest.json'
    scaler_path = data_dir/'scaler.json'
    for f in (train_file,val_file,manifest_path,scaler_path):
        if not f.is_file():
            raise FileNotFoundError(f)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('test_read') is not False or manifest['feature_count'] != 15:
        raise RuntimeError('数据预处理审计失败')
    x_train, y_train, ids_train, src_train = load_npz(train_file)
    x_val, y_val, ids_val, src_val = load_npz(val_file)
    if set(ids_train.tolist()) & set(ids_val.tolist()):
        raise RuntimeError('训练/验证sample_id重叠')
    if not np.all(src_val=='real') or len(x_val)!=16 or sum(y_val==1)!=3:
        raise RuntimeError('validation必须为独立真实16家（风险3家）')
    if len(x_train) != manifest['train_sizes'][mode]:
        raise RuntimeError('训练数量与预处理清单不一致')
    path = Path('LSTM/outputs/checkpoints')/run_name
    logs = Path('LSTM/outputs/logs')/run_name
    if path.exists() or logs.exists():
        raise FileExistsError(f'输出已存在，拒绝覆盖：{path} 或 {logs}')
    path.mkdir(parents=True)
    logs.mkdir(parents=True)
    model = MultiHeadFinancialLSTM(**cfg['model']).to(device)
    assert len(model.heads)==15
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg['training']['learning_rate'],
                                  weight_decay=cfg['training']['weight_decay'])
    sampler = build_sampler(y_train, src_train, mode,
                            float(cfg['training']['real_weight_in_augmented']),
                            cfg['project']['seed'])
    ds = TensorDataset(torch.from_numpy(x_train), torch.tensor(y_train,dtype=torch.float32))
    loader = DataLoader(ds, batch_size=cfg['training']['batch_size'], sampler=sampler,
                        num_workers=0, pin_memory=True)
    epochs = int(epochs_override or cfg['training']['epochs'])
    if epochs<1:
        raise ValueError('epochs必须为正数')
    patience=int(cfg['training']['patience'])
    best=float('inf'); bad=0; history=[]; best_epoch=None
    print('mode=',mode, 'num_train=',len(x_train), 'num_val=',len(x_val),
          'heads=',len(model.heads),'trainable_parameters=',sum(p.numel() for p in model.parameters()), flush=True)
    print('测试集不用于训练或验证。早停仅按真实validation宏平均BCE。',flush=True)
    for ep in range(1,epochs+1):
        model.train(); batch_losses=[]
        for bx,by in loader:
            bx=bx.to(device,non_blocking=True); by=by.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logit=model(bx)
            loss=nn.functional.binary_cross_entropy_with_logits(logit,by)
            if not torch.isfinite(loss):
                raise RuntimeError(f'epoch {ep} loss出现NaN/Inf')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),float(cfg['training']['grad_clip']))
            optimizer.step()
            batch_losses.append(loss.item())
        probs,val_macro=inference(model,x_val,y_val,cfg['training']['batch_size'],device)
        m=binary_metrics(y_val,probs, cfg['training']['threshold'])
        record={'epoch':ep,'train_bce':float(np.mean(batch_losses)),
                'val_macro_bce':val_macro,'val_auroc':m['auroc'],
                'val_pr_auc':m['pr_auc'],'val_macro_f1':m['macro_f1'],
                'val_bal_acc':m['balanced_accuracy']}
        history.append(record)
        if val_macro < best-float(cfg['training']['min_delta']):
            best=val_macro; best_epoch=ep; bad=0
            chk = {'model_state': model.state_dict(), 'model_config':model.config,
                   'mode':mode,'run_name':run_name,'best_epoch':ep,
                   'best_val_macro_bce':best,
                   'config':cfg,'test_used':False,
                   'manifest_sha256':sha256_file(manifest_path),
                   'train_npz_sha256':sha256_file(train_file),
                   'val_npz_sha256':sha256_file(val_file),
                   'scaler_state':json.loads(scaler_path.read_text(encoding='utf-8')),
                   'feature_keys':manifest['feature_keys'],
                   'train_origin_sha256':manifest['train_xlsx_sha256'],
                   'generated_xlsx_sha256':manifest['generated_xlsx_sha256'],
                   'threshold':cfg['training']['threshold']}
            torch.save(chk, path/'best.pt')
        else:
            bad+=1
        if ep == 1 or ep%10==0 or bad>=patience:
            print(f'{mode} epoch={ep}/{epochs} train={record["train_bce"]:.5f} '
                  f'val_macro_bce={val_macro:.5f} val_pr_auc={m["pr_auc"]:.4f} '
                  f'val_bal_acc={m["balanced_accuracy"]:.4f} best_epoch={best_epoch}',flush=True)
        if bad>=patience:
            print('Early stopping:',ep,flush=True)
            break
    with (logs/'history.csv').open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(history[0])); writer.writeheader();writer.writerows(history)
    json_save_exclusive(logs/'run_info.json', {'mode':mode,'best_epoch':best_epoch,
        'best_val_macro_bce':best,'train_n':len(x_train),'val_n':len(x_val),
        'epochs_completed':len(history),'test_used':False})
    print('训练完成:',path/'best.pt','最优epoch=',best_epoch,flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='LSTM/config/multihead_lstm.yaml')
    ap.add_argument('--mode',choices=MODES,required=True)
    ap.add_argument('--run-name',required=True)
    ap.add_argument('--epochs',type=int,default=None)
    args=ap.parse_args()
    train_mode(args.config,args.mode,args.run_name,args.epochs)

if __name__=='__main__':
    main()
