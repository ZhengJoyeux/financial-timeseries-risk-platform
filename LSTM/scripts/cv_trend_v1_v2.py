"""Five-fold outer, holdout-only inner-epoch selection, full outer-train refit.

L0 real-only. 15 independent LSTM heads. GPU1 UUID strictly enforced.
Never accesses independent validation/test, generated data, or existing scalers.
Run from /home/wqzheng/Financial platform in sers_ddpm environment.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from VAE.src.dataset import read_financial
from LSTM.src.common import physical_gpu1, seed_everything, sha256_file, json_save_exclusive
from LSTM.src.trend_cv import fold_indices, inner_indices, fold_features
from LSTM.src.trend_model_v1 import TrendCentered15HeadLSTM
from LSTM.src.trend_model_v2 import TrendLeverage15HeadLSTM
from LSTM.scripts.train_trend_v1 import build_sampler, inference
from LSTM.src.metrics import binary_metrics
import yaml

MODELS=('trend_v1','trend_v2_leverage')


def macro_bce(y,p):
    y=np.asarray(y,dtype=np.int64)
    p=np.clip(np.asarray(p,dtype=np.float64),1e-7,1-1e-7)
    if set(np.unique(y))!={0,1}:
        raise ValueError('宏平均BCE需要两个类别')
    return float((-np.log(p[y==1]).mean()-np.log1p(-p[y==0]).mean())/2)


def make_model(variant, state, levstats, model_kwargs):
    if variant=='trend_v1':
        obj=TrendCentered15HeadLSTM(**model_kwargs)
    elif variant=='trend_v2_leverage':
        obj=TrendLeverage15HeadLSTM(scaler_state=state, leverage_stats=levstats,
                                    leverage_enabled=True, **model_kwargs)
    else:
        raise ValueError(variant)
    if len(obj.heads)!=15 or len({id(h) for h in obj.heads})!=15:
        raise RuntimeError('没有15个独立LSTM头')
    return obj


def fit_epochs(variant, x_raw, y, ids, fit_indices, score_indices, *, cfg, seed, device,
               max_epochs, save_predictions, fit_phase, fit_n_epochs=None):
    """score_indices NEVER participates in gradient/scaler fitting.

    for inner selection, score_indices is early-stop partition; for final refit,
    score_indices is the sealed outer fold (inference only after refit).
    """
    if len(set(fit_indices)&set(score_indices)):
        raise RuntimeError('训练与评价企业重叠')
    y_train=np.asarray(y[fit_indices],dtype=np.int64)
    y_score=np.asarray(y[score_indices],dtype=np.int64)
    xtrain,xscore,state,levstats=fold_features(x_raw[fit_indices],x_raw[score_indices])
    if min(np.bincount(y_train,minlength=2))<1 or min(np.bincount(y_score,minlength=2))<1:
        raise RuntimeError('训练与评价折均需同时包含风险/正常企业')
    seed_everything(seed)
    model=make_model(variant,state,levstats,cfg['model']).to(device)
    opt=torch.optim.AdamW(model.parameters(), lr=float(cfg['training']['learning_rate']),
                          weight_decay=float(cfg['training']['weight_decay']))
    src=np.asarray(['real']*len(y_train))
    sampler=build_sampler(y_train,src,'real_only',float(cfg['training']['real_weight_in_augmented']),seed)
    loader=DataLoader(TensorDataset(torch.from_numpy(xtrain),torch.from_numpy(y_train.astype(np.float32))),
                      batch_size=int(cfg['training']['batch_size']),sampler=sampler,num_workers=0,pin_memory=True)
    limit=int(fit_n_epochs) if fit_n_epochs is not None else int(max_epochs)
    best=np.inf; best_epoch=0; no_improve=0; history=[]
    best_state=None
    for ep in range(1,limit+1):
        model.train(); losses=[]
        for xb,yb in loader:
            xb=xb.to(device,non_blocking=True);yb=yb.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss=F.binary_cross_entropy_with_logits(model(xb),yb)
            if not torch.isfinite(loss):raise RuntimeError('Loss出现NaN/Inf')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg['training']['grad_clip']))
            opt.step();losses.append(float(loss.item()))
        if fit_n_epochs is None:
            prob,score=inference(model,xscore,y_score,int(cfg['training']['batch_size']),device)
            history.append({'epoch':ep,'train_bce':float(np.mean(losses)),
                            'inner_val_macro_bce':float(score)})
            if score < best-float(cfg['training']['min_delta']):
                best=float(score);best_epoch=ep;no_improve=0
                # No need to save internal-best weights; only the selected epoch is used.
            else:
                no_improve+=1
            if no_improve>=int(cfg['training']['patience']):
                break
        else:
            history.append({'epoch':ep,'train_bce':float(np.mean(losses))})
    if fit_n_epochs is None:
        if best_epoch==0:raise RuntimeError('未选择到内部最佳轮次')
        return {'selected_epoch':best_epoch,'inner_best_macro_bce':best,
                'epochs_completed':len(history),'history':history,
                'inner_train_n':len(y_train),'inner_stop_n':len(y_score)}
    probs,score=inference(model,xscore,y_score,int(cfg['training']['batch_size']),device)
    if not np.isfinite(probs).all():raise RuntimeError('外层预测包含NaN/Inf')
    if len(probs)!=len(score_indices):raise RuntimeError('外层预测数量异常')
    info={'train_ids':[int(x) for x in ids[fit_indices]],
          'holdout_ids':[int(x) for x in ids[score_indices]],
          'scaler_state':state,'leverage_stats':levstats,
          'epochs':limit, 'outer_holdout_macro_bce':macro_bce(y_score,probs),
          'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}}
    return probs,info


def paired_bootstrap(y,prob_v1,prob_v2,seed=2026,n=500):
    from sklearn.metrics import roc_auc_score, average_precision_score
    y=np.asarray(y,dtype=np.int64)
    rng=np.random.default_rng(seed)
    idx0=np.flatnonzero(y==0);idx1=np.flatnonzero(y==1)
    stats={'auroc':[],'pr_auc':[],'brier':[]}
    for _ in range(n):
        take=np.r_[rng.choice(idx0,len(idx0),replace=True),rng.choice(idx1,len(idx1),replace=True)]
        yy=y[take]; a=prob_v1[take];b=prob_v2[take]
        stats['auroc'].append(roc_auc_score(yy,b)-roc_auc_score(yy,a))
        stats['pr_auc'].append(average_precision_score(yy,b)-average_precision_score(yy,a))
        stats['brier'].append(float(np.mean((b-yy)**2)-np.mean((a-yy)**2)))
    return {k:{'v2_minus_v1_p025':float(np.quantile(v,.025)),
               'v2_minus_v1_p975':float(np.quantile(v,.975)),
               'v2_minus_v1_median':float(np.median(v))} for k,v in stats.items()}


def prepare_train(config_path):
    cfg=yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    if cfg['model']['n_features']!=15 or cfg['model']['seq_len']!=20:
        raise RuntimeError('配置必须是15变量/20季度')
    if cfg['training']['checkpoint_selection']!='val_macro_bce':
        raise RuntimeError('仅支持宏平均BCE早停')
    train_path=Path('data/splits/train.xlsx')
    ids,raw,labels=read_financial(train_path)
    ids=np.asarray(ids,dtype=int)
    y=np.asarray([{'无风险':0,'有风险':1}[s] for s in labels],dtype=np.int64)
    if len(ids)!=48 or int(sum(y==1))!=10 or len(set(ids.tolist()))!=48:
        raise RuntimeError('企业级训练数据不是固定48家/10风险')
    manifest= json.loads(Path('data/splits/split_manifest.json').read_text(encoding='utf-8'))
    if set(ids.tolist())!=set(manifest['splits']['train']['company_ids']):
        raise RuntimeError('真实训练企业与划分清单不符')
    prepared=Path(cfg['project']['dataset'])/'manifest.json'
    if not prepared.is_file():raise FileNotFoundError(prepared)
    previous=json.loads(prepared.read_text(encoding='utf-8'))
    sourcehash=sha256_file(train_path)
    if previous['train_xlsx_sha256'] != sourcehash:
        raise RuntimeError('训练企业Excel与当前LSTM数据准备来源哈希不同')
    if previous.get('test_read') is not False:
        raise RuntimeError('LSTM原始数据准备存在测试集读取标记')
    return cfg,ids,raw,y,sourcehash


def run(args):
    device=physical_gpu1()
    cfg,ids,raw,y,sourcehash=prepare_train(args.config)
    base_seed=int(cfg['project']['seed'])
    epochs=2 if args.smoke else int(cfg['training']['epochs'])
    folds=fold_indices(y,n_splits=5,seed=base_seed)
    out=Path(args.output) if args.output else Path('LSTM/outputs/cross_validation')/(('trend_v1_v2_nested5_smoke_seed%d' if args.smoke else 'trend_v1_v2_nested5_seed%d')%base_seed)
    if out.exists():raise FileExistsError(f'拒绝覆盖已有输出目录：{out}')
    out.mkdir(parents=True,exist_ok=False)
    (out/'checkpoints').mkdir()
    json_save_exclusive(out/'protocol.json',{
        'protocol':'nested stratified company CV; outer holdout never used in epoch selection',
        'outer_folds':5,'run_folds':1 if args.smoke else 5,
        'inner_early_stop_fraction':.25,
        'outer_refit':'fresh initialization on all outer train firms, fixed epoch from inner selection',
        'scaler_and_leverage_fit':'fold training firms only, separately at inner and outer stages',
        'n_features':15,'n_quarters':20,'models':list(MODELS),
        'train_sha256':sourcehash,'seed':base_seed,'training_config':cfg['training'],
        'read_validation':False,'read_test':False,'use_s1_generated':False,
        'gpu':'physical GPU1, UUID locked','smoke':bool(args.smoke)})
    records=[]; oof=[]
    nfold=1 if args.smoke else 5
    for fold,(outer_train,outer_holdout) in enumerate(folds[:nfold],1):
        inner_train,inner_early=inner_indices(outer_train,y,seed=base_seed+fold)
        print(f'Fold {fold}/5: outer-train={len(outer_train)}, outer-holdout={len(outer_holdout)}, '
              f'inner-train={len(inner_train)}, inner-stop={len(inner_early)}',flush=True)
        for vi,variant in enumerate(MODELS):
            # Each version sees the exact same enterprise split.
            innerseed=base_seed+fold*100
            selected=fit_epochs(variant,raw,y,ids,inner_train,inner_early,cfg=cfg,seed=innerseed,
                                device=device,max_epochs=epochs,save_predictions=False,fit_phase='inner')
            best_ep=int(selected['selected_epoch'])
            # Different scaler, freshly fitted on 38/39 OUTER TRAIN ONLY; no outer holdout leakage.
            probs,artifact=fit_epochs(variant,raw,y,ids,outer_train,outer_holdout,cfg=cfg,
                                     seed=innerseed,device=device,max_epochs=epochs,save_predictions=True,
                                     fit_phase='refit',fit_n_epochs=best_ep)
            metrics=binary_metrics(y[outer_holdout],probs,threshold=float(cfg['training']['threshold']))
            row={'fold':fold,'variant':variant,'outer_train_n':len(outer_train),
                 'inner_train_n':len(inner_train),'inner_stop_n':len(inner_early),
                 'outer_holdout_n':len(outer_holdout),'inner_selected_epoch':best_ep,
                 'inner_best_macro_bce':selected['inner_best_macro_bce'],
                 'outer_holdout_macro_bce':artifact['outer_holdout_macro_bce'],
                 'outer_auroc':metrics['auroc'],'outer_pr_auc':metrics['pr_auc'],
                 'outer_f1_risk':metrics['f1_risk'],'outer_balanced_accuracy':metrics['balanced_accuracy'],
                 'outer_brier':float(np.mean((probs-y[outer_holdout])**2)),
                 'inner_train_ids':json.dumps(ids[inner_train].tolist()),
                 'inner_stop_ids':json.dumps(ids[inner_early].tolist()),
                 'outer_train_ids':json.dumps(ids[outer_train].tolist()),
                 'outer_holdout_ids':json.dumps(ids[outer_holdout].tolist())}
            records.append(row)
            print(f'  {variant}: selected_epoch={best_ep}, '
                  f'outer_AUROC={metrics["auroc"]:.4f}, PR-AUC={metrics["pr_auc"]:.4f}, '
                  f'F1={metrics["f1_risk"]:.4f}, Brier={row["outer_brier"]:.4f}',flush=True)
            for k,idx in enumerate(outer_holdout):
                oof.append({'fold':fold,'variant':variant,'company_id':int(ids[idx]),
                            'ground_truth_risk':int(y[idx]),'risk_probability':float(probs[k]),
                            'predicted_risk':int(probs[k]>=float(cfg['training']['threshold']))})
            # No checkpoint is accessed by the other model, no existing checkpoints overwritten.
            torch.save({'variant':variant,'fold':fold,'model_state':artifact['state_dict'],
                        'model_config':cfg['model'],'scaler_state':artifact['scaler_state'],
                        'leverage_stats':artifact['leverage_stats'],
                        'selected_epoch':best_ep,'train_enterprise_ids':artifact['train_ids'],
                        'heldout_enterprise_ids':artifact['holdout_ids'],
                        'train_sha256':sourcehash,'test_used':False,'validation_used':False,
                        'source':'real_only_outer_fold_train'},
                       out/'checkpoints'/f'{variant}_fold{fold}.pt')
    def write_csv(file,rows):
        with file.open('x',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_csv(out/'fold_metrics.csv',records)
    write_csv(out/'out_of_fold_predictions.csv',oof)
    if args.smoke:
        json_save_exclusive(out/'smoke_result.json',{'status':'passed one outer fold both variants',
                                                     'outer_test_unused':True,'train_sha256':sourcehash})
        print('SMOKE PASSED: 两个变体完成一次内层选择 + 外层重训/预测；不代表正式CV结论。',flush=True)
        return
    from sklearn.metrics import roc_auc_score,average_precision_score
    summary={};preds={}
    for variant in MODELS:
        ordered=sorted([r for r in oof if r['variant']==variant],key=lambda r:r['company_id'])
        if len(ordered)!=48 or len(set(r['company_id'] for r in ordered))!=48:
            raise RuntimeError('OOF没有唯一覆盖全部48家企业')
        yy=np.asarray([r['ground_truth_risk'] for r in ordered],np.int64)
        pp=np.asarray([r['risk_probability'] for r in ordered],np.float64)
        preds[variant]=(yy,pp)
        m=binary_metrics(yy,pp,threshold=float(cfg['training']['threshold']))
        m['brier_score']=float(np.mean((pp-yy)**2))
        m['macro_bce']=macro_bce(yy,pp)
        m['mean_inner_selected_epoch']=float(np.mean([r['inner_selected_epoch'] for r in records if r['variant']==variant]))
        summary[variant]=m
    if not np.array_equal(preds[MODELS[0]][0],preds[MODELS[1]][0]):
        raise RuntimeError('两个模型OOF的真实标签不一致')
    deltas=paired_bootstrap(preds[MODELS[0]][0],preds[MODELS[0]][1],preds[MODELS[1]][1],
                            seed=base_seed+400,n=500)
    json_save_exclusive(out/'aggregate_metrics.json',summary)
    json_save_exclusive(out/'paired_bootstrap_exploratory.json',deltas)
    with (out/'report.md').open('x',encoding='utf-8') as f:
        f.write('# Trend-V1 / Trend-V2 企业级嵌套五折交叉验证\n\n')
        f.write('仅使用原始 train.xlsx 的48家企业；不读取 validation/test；不使用预生成S1。\n\n')
        f.write('内层选择训练轮次，外层训练折重新拟合归一化和重训，外层留出企业只用于最终OOF预测。\n\n')
        f.write('| 模型 | OOF AUROC | OOF PR-AUC | 风险F1 | Balanced Acc | Brier |\n|---|---:|---:|---:|---:|---:|\n')
        for name in MODELS:
            m=summary[name]
            f.write(f'| {name} | {m["auroc"]:.4f} | {m["pr_auc"]:.4f} | {m["f1_risk"]:.4f} | {m["balanced_accuracy"]:.4f} | {m["brier_score"]:.4f} |\n')
        f.write('\n注意：这是模拟企业训练集内部的OOF估计，不是独立test。风险企业仅10家，统计波动大。\n')
        f.write('配对Bootstrap区间仅作探索性说明；不可据此证明真实企业泛化或数据增强有效。\n')
        f.write('当前CV只比较原始真实企业的V1/V2输入表征，不涉及S1增强。\n')
    print('嵌套五折完成。汇总：',out/'report.md',flush=True)
    for name in MODELS:
        m=summary[name]
        print(f'{name}: OOF AUROC={m["auroc"]:.4f}, PR-AUC={m["pr_auc"]:.4f}, '
              f'F1={m["f1_risk"]:.4f}, Brier={m["brier_score"]:.4f}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='LSTM/config/multihead_lstm.yaml')
    p.add_argument('--smoke',action='store_true',help='仅运行首个外层折，内层最多2轮')
    p.add_argument('--output',default=None,help='输出目录，不能存在')
    args=p.parse_args();run(args)


if __name__=='__main__':
    main()
