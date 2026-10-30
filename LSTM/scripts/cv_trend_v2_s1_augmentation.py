"""Leakage-safe, nested 5-fold S1 augmentation experiment for Trend-V2.

Modes: real_only / generated_only / real_plus_s1.
ALL S1 generators and normalization statistics are fit inside each training
subfold, independently at INNER epoch-selection and OUTER refit stages.
Only data/splits/train.xlsx is read; no project validation/test is touched.
Run with sers_ddpm from /home/wqzheng/Financial platform.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import yaml

from VAE.src.dataset import FinancialScaler
from LSTM.src.common import physical_gpu1,seed_everything,json_save_exclusive,sha256_file
from LSTM.src.metrics import binary_metrics
from LSTM.src.trend_cv import fold_indices,inner_indices,fold_features
from LSTM.src.trend_model_v2 import TrendLeverage15HeadLSTM
from LSTM.scripts.train_trend_v1 import inference
from LSTM.scripts.cv_trend_v1_v2 import prepare_train,macro_bce
from LSTM.src.foldwise_s1 import foldwise_generate,normalized_mode_data

MODES=('real_only','generated_only','real_plus_s1')


def sampler_for_equal_steps(y,source,mode,real_mass,seed,samples_per_epoch):
    """Fixed 200 training draws per epoch for ALL modes; class-balanced.

    Prevents augmented training from obtaining more SGD steps per epoch.
    """
    y=np.asarray(y,dtype=np.int64);source=np.asarray(source)
    expected=({'real_only':['real'],'generated_only':['generated'],
               'real_plus_s1':['real','generated']})[mode]
    if len(y)!=len(source) or len(y)==0 or not (0<real_mass<1):
        raise ValueError('采样标签、来源或训练质量配比不合法')
    groups={}
    for src in expected:
        for label in (0,1):
            idx=np.flatnonzero((source==src)&(y==label))
            if not len(idx):raise RuntimeError(f'{src}与类别{label}没有训练样本')
            mass=(real_mass if src=='real' else 1-real_mass) if mode=='real_plus_s1' else 1.0
            groups[(src,label)]=(idx,mass/2)
    weights=np.zeros(len(y),dtype=np.float64)
    for (src,label),(ix,mass) in groups.items():
        weights[ix]=mass/len(ix)
    if np.any(weights<=0):raise RuntimeError('训练数据包含未知来源或类别')
    gen=torch.Generator().manual_seed(int(seed))
    return WeightedRandomSampler(torch.from_numpy(weights),num_samples=int(samples_per_epoch),
                                 replacement=True,generator=gen)


def fit_mode(mode,real_train_x,real_train_y,gen_raw,gen_y,score_x,score_y,
             state,levstats,cfg,seed,device,*,max_epochs,refit_epochs=None,samples_per_epoch=200):
    x,y,sources=normalized_mode_data(real_train_x,real_train_y,gen_raw,gen_y,state,mode)
    seed_everything(seed)
    model=TrendLeverage15HeadLSTM(scaler_state=state,leverage_stats=levstats,
                                  leverage_enabled=True,**cfg['model']).to(device)
    if len(model.heads)!=15 or len({id(a) for a in model.heads})!=15:
        raise RuntimeError('Trend-V2没有15个独立LSTM头')
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(cfg['training']['learning_rate']),
                                 weight_decay=float(cfg['training']['weight_decay']))
    sampler=sampler_for_equal_steps(y,sources,mode,float(cfg['training']['real_weight_in_augmented']),
                                    seed,int(samples_per_epoch))
    loader=DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(y.astype(np.float32))),
                      sampler=sampler,batch_size=int(cfg['training']['batch_size']),
                      num_workers=0,pin_memory=True)
    upper=int(max_epochs) if refit_epochs is None else int(refit_epochs)
    if upper<=0:raise ValueError('训练epoch必须为正')
    best=float('inf');best_ep=0;bad=0;history=[]
    for epoch in range(1,upper+1):
        model.train();losses=[]
        for xb,yb in loader:
            xb=xb.to(device,non_blocking=True);yb=yb.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss=F.binary_cross_entropy_with_logits(model(xb),yb)
            if not torch.isfinite(loss):raise RuntimeError('训练BCE出现NaN/Inf')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg['training']['grad_clip']))
            optimizer.step();losses.append(float(loss.item()))
        row={'epoch':epoch,'train_bce':float(np.mean(losses))}
        if refit_epochs is None:
            probs,val_bce=inference(model,score_x,score_y,int(cfg['training']['batch_size']),device)
            row['inner_stop_macro_bce']=float(val_bce)
            if val_bce<best-float(cfg['training']['min_delta']):
                best=val_bce;best_ep=epoch;bad=0
            else:bad+=1
        history.append(row)
        if refit_epochs is None and bad>=int(cfg['training']['patience']):break
    if refit_epochs is None:
        if best_ep<1:raise RuntimeError('内层未选择到有效epoch')
        return {'selected_epoch':best_ep,'best_inner_macro_bce':best,
                'epochs_completed':len(history),'history':history}
    pred,holdout_bce=inference(model,score_x,score_y,int(cfg['training']['batch_size']),device)
    if len(pred)!=len(score_y) or not np.isfinite(pred).all():
        raise RuntimeError('外层预测无效')
    return {'probabilities':pred,'outer_macro_bce':float(holdout_bce),
            'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
            'epochs_completed':len(history),'history':history,
            'n_train_real':int(len(real_train_y)), 'n_train_generated':int(len(gen_y) if mode!='real_only' else 0),
            'n_draws_per_epoch':int(samples_per_epoch)}


def export_csv_exclusive(path,rows):
    if not rows:raise ValueError('没有可写的结果')
    path=Path(path)
    with path.open('x',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def aggregate_oof(oof,modes=MODES):
    summary={};by_mode={}
    for mode in modes:
        ordered=sorted((x for x in oof if x['mode']==mode),key=lambda r:r['company_id'])
        ids=[r['company_id'] for r in ordered]
        if len(ids)!=48 or len(set(ids))!=48:
            raise RuntimeError(f'{mode}没有恰好48家唯一折外企业')
        y=np.asarray([r['ground_truth_risk'] for r in ordered],dtype=np.int64)
        p=np.asarray([r['risk_probability'] for r in ordered],dtype=np.float64)
        m=binary_metrics(y,p,threshold=.5)
        m['brier_score']=float(np.mean((p-y)**2));m['macro_bce']=macro_bce(y,p)
        summary[mode]=m;by_mode[mode]=(ids,y,p)
    reference=by_mode['real_only']
    for name,rec in by_mode.items():
        if reference[0]!=rec[0] or not np.array_equal(reference[1],rec[1]):
            raise RuntimeError(f'{name}的企业ID或标签无法与L0逐一对应')
    return summary,by_mode


def paired_intervals(by_mode,seed=2026,repeats=500):
    from sklearn.metrics import roc_auc_score,average_precision_score
    rng=np.random.default_rng(seed)
    _,y,p0=by_mode['real_only']
    ix0=np.flatnonzero(y==0);ix1=np.flatnonzero(y==1)
    result={}
    for mode in ['generated_only','real_plus_s1']:
        p1=by_mode[mode][2];scores={'auroc':[],'pr_auc':[],'brier':[]}
        for _ in range(repeats):
            ix=np.r_[rng.choice(ix0,len(ix0),replace=True),rng.choice(ix1,len(ix1),replace=True)]
            yy=y[ix]
            scores['auroc'].append(float(roc_auc_score(yy,p1[ix])-roc_auc_score(yy,p0[ix])))
            scores['pr_auc'].append(float(average_precision_score(yy,p1[ix])-average_precision_score(yy,p0[ix])))
            scores['brier'].append(float(np.mean((p1[ix]-yy)**2)-np.mean((p0[ix]-yy)**2)))
        result[f'{mode}_minus_real_only']={k:{'median':float(np.median(v)),
                  'p025':float(np.quantile(v,.025)),'p975':float(np.quantile(v,.975))}
                  for k,v in scores.items()}
    return result


def run(args):
    device=physical_gpu1()  # Hard-enforces GPU1 UUID; no CPU/GPU0 fallback.
    cfg,ids,raw,y,sourcehash=prepare_train(args.config)
    if float(cfg['training']['threshold']) != 0.5:
        raise RuntimeError('本对照固定0.5分类阈值；请先恢复原始配置，不能在CV中调阈值')
    if cfg['training'].get('class_balance') is not True:
        raise RuntimeError('本对照需要原配置class_balance=true')
    base_seed=int(cfg['project']['seed'])
    folds=fold_indices(y,n_splits=5,seed=base_seed)
    n_generated=int(args.generated_per_class) if args.generated_per_class is not None else (8 if args.smoke else 100)
    if n_generated<3:raise ValueError('每类生成数量至少3家')
    epochs=2 if args.smoke else int(cfg['training']['epochs'])
    sampler_draws=32 if args.smoke else 200
    output=Path(args.output) if args.output else Path('LSTM/outputs/cross_validation') / (
        'trend_v2_s1_foldwise_smoke_seed%d'%base_seed if args.smoke else 'trend_v2_s1_foldwise_nested5_seed%d'%base_seed)
    if output.exists():raise FileExistsError(f'结果目录存在，拒绝覆盖：{output}')
    output.mkdir(parents=True,exist_ok=False)
    (output/'checkpoints').mkdir();(output/'fold_generated').mkdir()
    protocol={
        'protocol':'nested company-stratified 5-fold augmentation: inner epoch selection and outer refit',
        'model':'trend_v2_leverage_15_heads','modes':list(MODES),
        'n_outer_folds':5,'folds_executed':1 if args.smoke else 5,
        'inner_stop_fraction':0.25,'scaler_fit':'only current inner/outer real training subfold',
        'leverage_stats_fit':'only current inner/outer real training subfold',
        'S1_fit':'once per phase only current inner/outer real training subfold',
        'generated_per_class':n_generated,
        'S1_parameters':{'perturb':.20,'novelty_fraction':.50,'max_draw_multiplier':16},
        'sampler_draws_per_epoch':sampler_draws,
        'same_optimizer_updates_per_epoch_across_modes':True,
        'train_file':'data/splits/train.xlsx','train_sha256':sourcehash,
        'read_validation':False,'read_test':False,
        'pre_generated_full_train_S1_used':False,
        'device':'physical GPU1, UUID locked; exposed as cuda:0',
        'seed':base_seed,'training_config':cfg['training'],
        'smoke':bool(args.smoke),
        'warning':'risk inner 6 / outer 8: S1 statistical fit may be unstable; inspect novelty fallback and fold quality.'
    }
    json_save_exclusive(output/'protocol.json',protocol)
    results=[];oof=[]
    for fold,(outer_train,outer_holdout) in enumerate(folds[:1 if args.smoke else 5],1):
        inner_train,inner_stop=inner_indices(outer_train,y,seed=base_seed+fold)
        print(f'Fold {fold}/5: outer fit={len(outer_train)}, holdout={len(outer_holdout)}; '
              f'inner fit={len(inner_train)}, stop={len(inner_stop)}',flush=True)
        phase_data={}
        for name,fit_indices,score_indices,add in (
            ('inner',inner_train,inner_stop,0),
            ('outer',outer_train,outer_holdout,10000)):
            scaler_fit_x,score_x,state,levstats=fold_features(raw[fit_indices],raw[score_indices])
            seed=base_seed+100*fold+add+19
            generated_raw,gen_y,audit=foldwise_generate(raw,y,ids,fit_indices,score_indices,
                                  device='cuda:0',seed=seed,per_class=n_generated,
                                  perturb=.20,novelty_fraction=.50,max_draw_multiplier=16)
            # No S1 generated data from the full 48-enterprise fit are ever imported.
            phase_path=output/'fold_generated'/f'fold{fold}_{name}'
            phase_path.mkdir()
            json_save_exclusive(phase_path/'provenance.json',{
                **audit, 'fold':fold,'phase':name, 'real_fit_ids':[int(v) for v in ids[fit_indices]],
                'other_ids':[int(v) for v in ids[score_indices]],
                'real_fit_sha256':sha256_file(Path('data/splits/train.xlsx')),
                'scaler_state':state,'leverage_stats':levstats})
            np.savez_compressed(phase_path/'generated_raw.npz',
                                x=generated_raw,y=gen_y,
                                source=np.repeat('generated',len(gen_y)),
                                origin_fold=np.full(len(gen_y),fold),
                                origin_phase=np.repeat(name,len(gen_y)))
            phase_data[name]=(scaler_fit_x,y[fit_indices],score_x,y[score_indices],state,
                              levstats,generated_raw,gen_y)
            print(f'  S1 {name}: risk-fit={int(sum(y[fit_indices]==1))}, '
                  f'normal-fit={int(sum(y[fit_indices]==0))}, generated={len(gen_y)}; '
                  f'fallback risk={audit["novelty_sampler_diagnostics"]["有风险"]["quantile_slots_without_threshold_pass_fallback_to_max_distance"]}, '
                  f'normal={audit["novelty_sampler_diagnostics"]["无风险"]["quantile_slots_without_threshold_pass_fallback_to_max_distance"]}',flush=True)
        for mode in MODES:
            tr,yy,stop,ys,state,lev,gen,yg=phase_data['inner']
            seed=base_seed+fold*1000+29
            fit=fit_mode(mode,tr,yy,gen,yg,stop,ys,state,lev,cfg,seed,device,
                         max_epochs=epochs,samples_per_epoch=sampler_draws)
            best_ep=int(fit['selected_epoch'])
            tr,yy,hold,yh,state,lev,gen,yg=phase_data['outer']
            outer=fit_mode(mode,tr,yy,gen,yg,hold,yh,state,lev,cfg,seed,device,
                           max_epochs=epochs,refit_epochs=best_ep,samples_per_epoch=sampler_draws)
            m=binary_metrics(y[outer_holdout],outer['probabilities'],
                             threshold=float(cfg['training']['threshold']))
            result={'fold':fold,'mode':mode,'inner_selected_epoch':best_ep,
                    'inner_best_macro_bce':fit['best_inner_macro_bce'],
                    'outer_holdout_macro_bce':outer['outer_macro_bce'],
                    'outer_auroc':m['auroc'],'outer_pr_auc':m['pr_auc'],
                    'outer_f1_risk':m['f1_risk'],'outer_balanced_accuracy':m['balanced_accuracy'],
                    'outer_brier':float(np.mean((outer['probabilities']-y[outer_holdout])**2)),
                    'inner_real_n':len(inner_train),'outer_real_n':len(outer_train),
                    'generated_n_per_fit':len(yg),
                    'inner_real_ids':json.dumps(ids[inner_train].tolist()),
                    'inner_stop_ids':json.dumps(ids[inner_stop].tolist()),
                    'outer_real_ids':json.dumps(ids[outer_train].tolist()),
                    'outer_holdout_ids':json.dumps(ids[outer_holdout].tolist())}
            results.append(result)
            for j,k in enumerate(outer_holdout):
                oof.append({'fold':fold,'mode':mode,'company_id':int(ids[k]),
                            'ground_truth_risk':int(y[k]),
                            'risk_probability':float(outer['probabilities'][j]),
                            'predicted_risk':int(outer['probabilities'][j]>=.5)})
            torch.save({'fold':fold,'mode':mode,'model_version':'trend_v2_leverage',
                        'model_state':outer['state_dict'],'model_config':cfg['model'],
                        'scaler_state':state,'leverage_stats':lev,
                        'selected_epoch':best_ep,'train_enterprise_ids':ids[outer_train].tolist(),
                        'holdout_enterprise_ids':ids[outer_holdout].tolist(),
                        'n_real_train':len(outer_train),'n_generated_train':0 if mode=='real_only' else len(yg),
                        'source':'fold_local_s1' if mode!='real_only' else 'real_only',
                        'train_sha256':sourcehash,'validation_used':False,'test_used':False},
                       output/'checkpoints'/f'{mode}_fold{fold}.pt')
            print(f'  {mode}: inner-best-epoch={best_ep}, outer-AUROC={m["auroc"]:.4f}, '
                  f'PR-AUC={m["pr_auc"]:.4f}, F1={m["f1_risk"]:.4f}, '
                  f'Brier={result["outer_brier"]:.4f}',flush=True)
    export_csv_exclusive(output/'fold_metrics.csv',results)
    export_csv_exclusive(output/'out_of_fold_predictions.csv',oof)
    if args.smoke:
        json_save_exclusive(output/'smoke_result.json',
                            {'status':'SMOKE PASSED','folds':1,'n_train_sha256':sourcehash,
                             'S1_fit_inner_only':True,'S1_fit_outer_only':True,
                             'validation_and_test_not_read':True})
        print('SMOKE PASSED: 一折/三模式 S1折内生成+内层选轮次+外层重训与预测。',flush=True)
        return
    summary,by_mode=aggregate_oof(oof)
    intervals=paired_intervals(by_mode,seed=base_seed+400,repeats=500)
    json_save_exclusive(output/'aggregate_metrics.json',summary)
    json_save_exclusive(output/'paired_bootstrap_exploratory.json',intervals)
    with (output/'report.md').open('x',encoding='utf-8') as f:
        f.write('# Trend-V2 + S1 折内生成：企业级嵌套五折数据增强实验\n\n')
        f.write('仅原始train.xlsx 48家企业参与划分；每个内外训练阶段独立拟合S1；禁止使用基于全部48家生成的旧S1。\n\n')
        f.write('每个模式每轮固定200次有放回抽样；0.5风险阈值固定；内层选epoch，外层全训练折重训。\n\n')
        f.write('| 模式 | OOF AUROC | OOF PR-AUC | 风险F1 | Balanced Acc | Brier |\n')
        f.write('|---|---:|---:|---:|---:|---:|\n')
        for mode in MODES:
            m=summary[mode]
            f.write(f'| {mode} | {m["auroc"]:.4f} | {m["pr_auc"]:.4f} | {m["f1_risk"]:.4f} | '
                    f'{m["balanced_accuracy"]:.4f} | {m["brier_score"]:.4f} |\n')
        f.write('\nL2与L0差值及探索性配对Bootstrap参见paired_bootstrap_exploratory.json。\n')
        f.write('风险内层训练约6家，外层约8家；S1小样本生成不保证质量。独立validation/test不参与。\n')
        f.write('本实验为模拟金融数据内部OOF，不能证明真实公司泛化、因果数据增强或隐私保护。\n')
    print('正式折内S1增强五折完成：',output/'report.md',flush=True)
    for mode in MODES:
        m=summary[mode]
        print(f'{mode}: OOF AUROC={m["auroc"]:.4f}, PR-AUC={m["pr_auc"]:.4f}, '
              f'F1={m["f1_risk"]:.4f}, Brier={m["brier_score"]:.4f}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='LSTM/config/multihead_lstm.yaml')
    parser.add_argument('--smoke',action='store_true',help='one fold, 2 epochs, 8 generated per class')
    parser.add_argument('--generated-per-class',type=int,default=None,
                        help='formal default 100/class; smoke default 8/class')
    parser.add_argument('--output',default=None,help='must not already exist')
    run(parser.parse_args())

if __name__=='__main__':main()
