"""Read-only comparison: S0/S1 vs train, plus validation-nearest auxiliary stats.

Never reads test. Writes ONLY to new S1 evaluation directory.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from VAE.src.dataset import read_financial,FinancialScaler
from VAE.src.finance_rules import violation_rates

ROOT=Path(__file__).resolve().parents[2]


def _sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(data,ref,scaler,train_loo_median):
    a=data[:,:,0];assets=np.maximum(a,1e-6)
    debt=np.maximum(data[:,:,1],1e-6)
    rs=scaler.transform(ref).reshape(len(ref),300).astype(float)
    xs=scaler.transform(data).reshape(len(data),300).astype(float)
    dist=cdist(xs,rs).min(axis=1)
    return {
       'asset_mean':float(a.mean()),
       'asset_p95_all_quarters':float(np.quantile(a,.95)),
       'asset_p95_firm_means':float(np.quantile(a.mean(axis=1),.95)),
       'asset_abs_log_qoq_median':float(np.median(np.abs(np.diff(np.log(assets),axis=1)))),
       'liability_abs_log_qoq_median':float(np.median(np.abs(np.diff(np.log(debt),axis=1)))),
       'nearest_to_train_median':float(np.median(dist)),
       'nearest_below_half_loo_fraction':float(np.mean(dist < .5*train_loo_median)),
       'pre_rule_violation_max':float(max(violation_rates(data).values())),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--s0',default='structured_baseline_v1_n100')
    p.add_argument('--s1',default='structured_s1_n100')
    args=p.parse_args()
    paths={k:ROOT/'VAE/outputs/evaluation'/v for k,v in [('s0',args.s0),('s1',args.s1)]}
    for name,path in paths.items():
        if not (path/'summary_all.json').is_file():raise SystemExit(f'{name}指标缺失：{path}')
        if not (path/'generation_provenance.json').is_file():raise SystemExit(f'{name}来源缺失：{path}')
    s0p=json.loads((paths['s0']/'generation_provenance.json').read_text())
    s1p=json.loads((paths['s1']/'generation_provenance.json').read_text())
    train=ROOT/'data/splits/train.xlsx'
    digest=_sha(train)
    if s0p.get('train_sha256')!=digest or s1p.get('train_sha256')!=digest:
        raise SystemExit('S0/S1训练集SHA与当前train.xlsx不一致，拒绝比较')
    _,x,labels=read_financial(train)
    scaler=FinancialScaler().fit(x)
    s0=json.loads((paths['s0']/'summary_all.json').read_text())
    s1=json.loads((paths['s1']/'summary_all.json').read_text())
    rows=[];gates={}
    for tag,lab in [('risk','有风险'),('normal','无风险')]:
        ref=x[labels==lab]
        flat=scaler.transform(ref).reshape(len(ref),300).astype(float)
        dd=cdist(flat,flat);np.fill_diagonal(dd,np.inf)
        loo=float(np.median(dd.min(axis=1)))
        realstats=summary(ref,ref,scaler,loo)
        metric={}
        for version in ['s0','s1']:
            fname=ROOT/'VAE/outputs/generated'/ (args.s0 if version=='s0' else args.s1)/f'generated_{tag}_100.xlsx'
            if not fname.is_file():raise FileNotFoundError(fname)
            ids,arr,ylab=read_financial(fname)
            if len(ids)!=100 or np.any(ylab!=lab):raise ValueError(f'{version}数量/标签有误：{fname}')
            metric[version]=summary(arr,ref,scaler,loo)
            metric[version].update({
                'cos_nn':(s0 if version=='s0' else s1)[tag]['cosine_nn_train_mean'],
                'mmd2':(s0 if version=='s0' else s1)[tag]['rbf_mmd2'],
                'corr_mae':(s0 if version=='s0' else s1)[tag]['correlation_matrix_mae'],
                'trajectory_rmse':(s0 if version=='s0' else s1)[tag]['mean_trajectory_rmse_standardized'],
            })
        row={'label':tag,'train_count':len(ref),'train_loo_nearest_median':loo}
        for key,value in realstats.items():row['train_'+key]=value
        for version in ['s0','s1']:
            for key,value in metric[version].items():row[version+'_'+key]=value
        rows.append(row)
        checks={
           'asset_mean_gap_not_worse_than_s0':abs(metric['s1']['asset_mean']/realstats['asset_mean']-1) <= abs(metric['s0']['asset_mean']/realstats['asset_mean']-1),
           'asset_p95_gap_not_worse_than_s0':abs(metric['s1']['asset_p95_all_quarters']/realstats['asset_p95_all_quarters']-1) <= abs(metric['s0']['asset_p95_all_quarters']/realstats['asset_p95_all_quarters']-1),
           'asset_qoq_gap_under_2_percentage_points':abs(metric['s1']['asset_abs_log_qoq_median']-realstats['asset_abs_log_qoq_median']) <= .02,
           'raw_financial_constraints':metric['s1']['pre_rule_violation_max']<1e-3,
           'near_copy_fraction_not_worse_by_over_3pp':metric['s1']['nearest_below_half_loo_fraction'] <= metric['s0']['nearest_below_half_loo_fraction']+.03,
           'corr_mae_no_more_than_1p5x_s0':metric['s1']['corr_mae'] <= metric['s0']['corr_mae']*1.5,
        }
        gates[tag]={'checks':checks,'all_checks_pass':all(checks.values()),
                    'warning':'This gate is a conservative diagnostic; not proof of realism or privacy.'}
    df=pd.DataFrame(rows)
    out=paths['s1']
    csv=out/'comparison_s0_s1.csv'
    if csv.exists() or (out/'comparison_gate.json').exists():
        raise SystemExit('对比文件已存在，为避免覆盖请先检查已有结果')
    df.to_csv(csv,index=False,encoding='utf-8-sig')
    (out/'comparison_gate.json').write_text(json.dumps({
        'training_sha256':digest,
        's0':args.s0,'s1':args.s1,'risk':gates['risk'],'normal':gates['normal'],
        'overall_ready_for_downstream':all(g['all_checks_pass'] for g in gates.values()),
        'caveat':'Only simulated firms; validation size 3 risk/13 normal; never used test; independent downstream trial required',
    },ensure_ascii=False,indent=2),encoding='utf-8')
    for tag in ['risk','normal']:
        print(tag,'质量检查：',gates[tag]['checks'])
    print('S1是否通过全部保守质量门槛：',all(g['all_checks_pass'] for g in gates.values()))
    print('已保存：',csv)

if __name__=='__main__':main()
