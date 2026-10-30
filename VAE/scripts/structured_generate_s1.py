"""Standalone train-only fitted S1 structured baseline with original Excel/metrics interface."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from VAE.scripts.train import device_from
from VAE.src.dataset import FinancialScaler,read_financial,save_excel
from VAE.src.finance_rules import violation_rates
from VAE.src.metrics import metric_report
from VAE.src.structured_finance_s1 import StructuredFinanceS1
ROOT=Path(__file__).resolve().parents[2]


def _sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _quarter_abs_log(x,feature=0):
    v=np.maximum(x[...,feature],1e-6)
    return float(np.median(np.abs(np.diff(np.log(v),axis=1))))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--risk-number',type=int,default=100)
    p.add_argument('--normal-number',type=int,default=100)
    p.add_argument('--device',default='cuda:0',choices=['cuda:0'])
    p.add_argument('--seed',type=int,default=2027)
    p.add_argument('--perturb',type=float,default=.20)
    p.add_argument('--novelty-fraction',type=float,default=.50)
    p.add_argument('--output-name',default='structured_s1_n100')
    args=p.parse_args()
    # no fallback: this project requires physical GPU1 and a single CUDA device
    device=device_from(args.device)
    print("已验证仅使用物理GPU1:",device,flush=True)
    if not args.output_name.replace('_','').replace('-','').isalnum():
        p.error('--output-name only supports letters/numbers/_/-')
    if args.risk_number<=0 or args.normal_number<=0:p.error('Counts must be positive')
    out=ROOT/'VAE/outputs/generated'/args.output_name
    ev=ROOT/'VAE/outputs/evaluation'/args.output_name
    if out.exists() or ev.exists():raise SystemExit('Outputs already exist; choose a new --output-name')
    train=ROOT/'data/splits/train.xlsx';valid=ROOT/'data/splits/validation.xlsx'
    manifest=ROOT/'data/splits/split_manifest.json'
    for f in [train,valid,manifest]:
        if not f.exists():raise FileNotFoundError(f)
    ids,x,labels=read_financial(train)
    vid,vx,vy=read_financial(valid)
    assert len(ids)==48 and int(np.sum(labels=='有风险'))==10,'Unexpected TRAIN split'
    assert len(vid)==16 and int(np.sum(vy=='有风险'))==3,'Unexpected validation split'
    assert not set(ids.tolist()).intersection(set(vid.tolist())),'Data leakage'
    scaler=FinancialScaler().fit(x)
    model=StructuredFinanceS1(seed=args.seed,perturb=args.perturb,device=str(device),
                               novelty_fraction=args.novelty_fraction).fit(x,labels)
    out.mkdir(parents=True,exist_ok=False);ev.mkdir(parents=True,exist_ok=False)
    provenance={'experiment':'structured_finance_s1_tail_aware','class':'non_neural_statistical_baseline',
                'not_a_timevae_checkpoint':True,
                'training_file':str(train),'train_sha256':_sha(train),
                'validation_sha256':_sha(valid),'split_manifest_sha256':_sha(manifest),
                'test_never_read':True,'device':str(device),
                'physical_gpu1_required':True,'seed':args.seed,
                'generated_counts':{'risk':args.risk_number,'normal':args.normal_number},
                'fitted_model':model.stats(),
                's0_sampling_replaced_only':True}
    (ev/'generation_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
    results=[];summary={};qoq={};novelty={}
    for tag,label,count,base,seed_inc in [('risk','有风险',args.risk_number,300000,0),
                                            ('normal','无风险',args.normal_number,400000,1)]:
        y=model.sample(label,count,seed=args.seed+seed_inc)
        vr=violation_rates(y)
        if not np.isfinite(y).all() or max(vr.values())>1e-3:
            raise RuntimeError(f'Raw accounting rules failed for {tag}: {vr}')
        out_ids=np.arange(base+1,base+1+count)
        out_labels=np.repeat(label,count)
        save_excel(out/f'generated_{tag}_{count}.xlsx',out_ids,y,out_labels)
        results.append((out_ids,y,out_labels))
        report=metric_report(x[labels==label],y,vx[vy==label],scaler,ev,tag,raw_generated=y)
        summary[tag]=report
        from sklearn.metrics.pairwise import euclidean_distances
        r=scaler.transform(x[labels==label]).reshape(-1,300).astype(float)
        gen=scaler.transform(y).reshape(-1,300).astype(float)
        rr=euclidean_distances(r,r);np.fill_diagonal(rr,np.inf)
        loo=rr.min(axis=1)
        closest=euclidean_distances(gen,r).min(axis=1)
        v_scaled=scaler.transform(vx[vy==label]).reshape(-1,300).astype(float)
        closest_val=euclidean_distances(v_scaled,r).min(axis=1)
        thr=float(np.median(loo))
        novelty[tag]={
            'training_loo_nearest_median':thr,
            'generated_to_train_nearest_median':float(np.median(closest)),
            'fraction_generated_below_half_train_loo_median':float(np.mean(closest<.5*thr)),
            'fraction_generated_below_train_loo_median':float(np.mean(closest<thr)),
            'validation_to_train_nearest_median_aux_only':float(np.median(closest_val)),
            'validation_sample_count':len(v_scaled),
            'filter_statistics':model.last_sample_diagnostics.get(label,{}),
            'warning':'Train proximity alone cannot establish non-memorization; n-risk=10'
        }
        qoq[tag]={
            'train_asset_abs_log_qoq_median':_quarter_abs_log(x[labels==label]),
            'generated_asset_abs_log_qoq_median':_quarter_abs_log(y),
            'train_assets_mean':float(x[labels==label,:,0].mean()),
            'generated_assets_mean':float(y[:,:,0].mean()),
            'train_asset_p95':float(np.quantile(x[labels==label,:,0],.95)),
            'generated_asset_p95':float(np.quantile(y[:,:,0],.95)),
            'train_firm_mean_asset_p95':float(np.quantile(x[labels==label,:,0].mean(axis=1),.95)),
            'generated_firm_mean_asset_p95':float(np.quantile(y[:,:,0].mean(axis=1),.95)),
            'train_liability_abs_log_qoq_median':_quarter_abs_log(x[labels==label],1),
            'generated_liability_abs_log_qoq_median':_quarter_abs_log(y,1),
        }
        print(f'{tag}: COS={report["cosine_nn_train_mean"]:.4f} '
              f'MMD2={report["rbf_mmd2"]:.4f} '
              f'pre_violation={max(vr.values()):.5f} '
              f'asset_QoQ={qoq[tag]["generated_asset_abs_log_qoq_median"]:.4f}',flush=True)
    save_excel(out/'generated_all.xlsx',
               np.concatenate([i for i,_,_ in results]),
               np.concatenate([a for _,a,_ in results]),
               np.concatenate([v for _,_,v in results]))
    (ev/'sampler_diagnostics.json').write_text(json.dumps(model.last_sample_diagnostics,ensure_ascii=False,indent=2),encoding='utf-8')
    (ev/'summary_all.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (ev/'stock_qoq_diagnostics.json').write_text(json.dumps(qoq,ensure_ascii=False,indent=2),encoding='utf-8')
    (ev/'novelty_diagnostics.json').write_text(json.dumps(novelty,ensure_ascii=False,indent=2),encoding='utf-8')
    print('生成完成:',out)
    print('评价完成:',ev)
    print('未读取test；GPU1拟合及生成；S0、D0–V3源文件和输出未修改。')

if __name__=='__main__':main()
