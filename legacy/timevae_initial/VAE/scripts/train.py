"""One jointly-trained conditional TimeVAE, using 48 merged TRAIN firms and class-aware sampling.
Validation is strictly for early stopping. Never accesses the independent test.xlsx.
"""
import argparse,csv,hashlib,json,random
from pathlib import Path
import numpy as np
import torch,yaml
from torch.utils.data import DataLoader,TensorDataset,WeightedRandomSampler
from VAE.src.dataset import FinancialScaler,read_financial
from VAE.src.schema import NAMES
from VAE.src.timevae import TimeVAE
ROOT=Path(__file__).resolve().parents[2]


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def device_from(value):
    if value=='auto':return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dev=torch.device(value)
    if dev.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA不可用，请用--device cpu。')
    return dev


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='config/vae.yaml')
    p.add_argument('--epochs',type=int)
    p.add_argument('--device',default=None)
    args=p.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    if args.epochs is not None and args.epochs <1:raise SystemExit('--epochs必须大于0')
    epochs=args.epochs or cfg['training']['epochs']
    seed_all(cfg['training']['random_seed']);torch.set_num_threads(min(torch.get_num_threads(),4))
    device=device_from(args.device or cfg['training']['device'])
    print('device:',device,'epochs:',epochs)
    ids,x,y=read_financial(ROOT/'data/splits/train.xlsx')
    vi,vx,vy=read_financial(ROOT/'data/splits/validation.xlsx')
    manifest=json.loads((ROOT/'data/splits/split_manifest.json').read_text(encoding='utf-8'))
    if set(ids.tolist()) != set(manifest['splits']['train']['company_ids']):raise RuntimeError('训练Excel不匹配划分manifest')
    if set(vi.tolist()) != set(manifest['splits']['validation']['company_ids']):raise RuntimeError('验证Excel不匹配划分manifest')
    if len(ids)!=48 or sum(y=='有风险')!=10 or len(vi)!=16 or sum(vy=='有风险')!=3:
        raise RuntimeError('划分样本数不符合48/16及风险分层设置。')
    scaler=FinancialScaler().fit(x) # FIT TRAIN ONLY, mixed class 48 companies.
    x_train=torch.from_numpy(scaler.transform(x))
    y_train=torch.as_tensor((y=='有风险').astype(np.int64),dtype=torch.long)
    x_val=torch.from_numpy(scaler.transform(vx)).to(device)
    y_val=torch.as_tensor((vy=='有风险').astype(np.int64),dtype=torch.long,device=device)
    # Both stratified train subsets merged; balanced sampler resamples the minority class.
    class_count=torch.bincount(y_train,minlength=2).float()
    weights=(1/class_count)[y_train]
    sampler=WeightedRandomSampler(weights,num_samples=len(y_train),replacement=True,
                                   generator=torch.Generator().manual_seed(cfg['training']['random_seed']))
    dl=DataLoader(TensorDataset(x_train,y_train),batch_size=cfg['training']['batch_size'],sampler=sampler,drop_last=False)
    mdl=TimeVAE(seq_len=20,feature_dim=15,**cfg['model']).to(device)
    opt=torch.optim.AdamW(mdl.parameters(),lr=cfg['training']['learning_rate'],
                          weight_decay=cfg['training']['weight_decay'])
    ckpt=ROOT/'VAE/outputs/checkpoints/timevae_conditional_best.pt'
    log=ROOT/'VAE/outputs/logs/conditional_history.csv'
    ckpt.parent.mkdir(parents=True,exist_ok=True);log.parent.mkdir(parents=True,exist_ok=True)
    best=float('inf');bad=0;history=[]
    for epoch in range(1,epochs+1):
        mdl.train();train_losses=[]
        for xb,yb in dl:
            xb=xb.to(device);yb=yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred,mu,logvar=mdl(xb,yb)
            loss,_=mdl.loss(xb,pred,mu,logvar,
                            cfg['training']['reconstruction_weight'],cfg['training']['beta_kl'],
                            cfg['training']['temporal_weight'])
            if not torch.isfinite(loss):raise RuntimeError('训练loss不再有限，已停止。')
            loss.backward();torch.nn.utils.clip_grad_norm_(mdl.parameters(),cfg['training']['grad_clip']);opt.step()
            train_losses.append(float(loss.detach()))
        mdl.eval()
        with torch.no_grad():
            mu,_=mdl.encode(x_val,y_val)
            reconstructed=mdl.decode(mu,y_val)
            class_vals=[]
            for c in [0,1]:
                chosen=y_val==c
                if not chosen.any():raise RuntimeError('验证集缺少某风险类别。')
                mse=torch.nn.functional.mse_loss(reconstructed[chosen],x_val[chosen])
                tr=torch.nn.functional.mse_loss(
                    reconstructed[chosen,1:]-reconstructed[chosen,:-1],
                    x_val[chosen,1:]-x_val[chosen,:-1])
                class_vals.append(float(cfg['training']['reconstruction_weight']*mse+
                                        cfg['training']['temporal_weight']*tr))
            # Both classes equally weighted for selection (not biased by 13 vs 3 validation frequencies).
            val_loss=float(np.mean(class_vals))
        history.append((epoch,float(np.mean(train_losses)),val_loss,class_vals[0],class_vals[1]))
        if val_loss<best-1e-6:
            best=val_loss;bad=0
            torch.save({'model_state':mdl.state_dict(),'model_params':mdl.params,
                        'scaler_state':scaler.state_dict(),'feature_names':NAMES,
                        'n_train':len(ids),'n_train_risk':int(y_train.sum()),
                        'n_validation':len(vi),'n_validation_risk':int(y_val.sum()),
                        'label_map':{'无风险':0,'有风险':1},
                        'split_manifest_sha256':hashlib.sha256((ROOT/'data/splits/split_manifest.json').read_bytes()).hexdigest(),
                        'train_xlsx_sha256':hashlib.sha256((ROOT/'data/splits/train.xlsx').read_bytes()).hexdigest(),
                        'validation_xlsx_sha256':hashlib.sha256((ROOT/'data/splits/validation.xlsx').read_bytes()).hexdigest(),
                        'best_epoch':epoch,'best_val_loss':best,'config':cfg},ckpt)
        else:bad+=1
        if epoch==1 or epoch%20==0 or epoch==epochs:
            print(f'epoch {epoch:03d}/{epochs}, train={history[-1][1]:.5f}, '
                  f'val_macro={val_loss:.5f}, val_normal={class_vals[0]:.5f}, val_risk={class_vals[1]:.5f}',flush=True)
        if bad>=cfg['training']['patience']:
            print('Early stopping at epoch',epoch);break
    with log.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['epoch','train_loss','val_macro','val_normal','val_risk']);w.writerows(history)
    print(f'已保存联合条件TimeVAE：{ckpt}；best_val_macro={best:.5f}。没有读取 test.xlsx。')

if __name__=='__main__':main()
