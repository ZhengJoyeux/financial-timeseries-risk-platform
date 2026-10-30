"""Finite-sample diagnostics: no single metric establishes synthetic-data validity."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
from sklearn.preprocessing import StandardScaler
from .finance_rules import violation_rates
from .schema import KEYS,NAMES


def _scalar(x):
    return float(x) if np.isfinite(x) else None


def metric_report(real_train,generated,reference_validation,scaler,output_dir,tag,raw_generated=None):
    """Coverage = fraction of train companies covered by a generated NN inside train leave-one-out P95 radius.
    All distances/cosines calculated on the TRAIN-fitted transform, flattened across 20x15.
    Validation stats are reported as auxiliary diagnostic only; test is not accessed.
    """
    root=Path(output_dir);root.mkdir(parents=True,exist_ok=True)
    r=scaler.transform(real_train).reshape(len(real_train),-1).astype(float)
    g=scaler.transform(generated).reshape(len(generated),-1).astype(float)
    v=scaler.transform(reference_validation).reshape(len(reference_validation),-1).astype(float)
    dist=euclidean_distances(r,r)
    np.fill_diagonal(dist,np.inf)
    l1=dist.min(axis=1)
    radius=float(np.quantile(l1,.95))
    d_rg=euclidean_distances(r,g)
    nearest_train=d_rg.min(axis=0)
    nearest_real=d_rg.min(axis=1)
    coverage=float(np.mean(nearest_real <= radius))
    # Cosine to nearest train reference is NOT a paired similarity score.
    cos_nn=cosine_similarity(g,r).max(axis=1)
    real_variability=float(np.median(l1))
    synth_div=euclidean_distances(g,g)
    np.fill_diagonal(synth_div,np.inf)
    diversity=float(np.median(synth_div.min(axis=1))) if len(g)>1 else float('nan')
    # Lag-1 per feature: average over enterprises, handles tiny groups.
    def lag_avg(z):
        mat=[]
        for j in range(z.shape[2]):
            scores=[]
            for sample in z:
                a=sample[:-1,j];b=sample[1:,j]
                if np.std(a)>1e-8 and np.std(b)>1e-8:
                    scores.append(np.corrcoef(a,b)[0,1])
            mat.append(np.mean(scores) if scores else 0)
        return np.nan_to_num(mat)
    # Cross-feature contemporaneous correlation, not covariance across flattened time coordinates.
    rt=scaler.transform(real_train).reshape(-1,15)
    gt=scaler.transform(generated).reshape(-1,15)
    def robust_corr(m):
        v=m.std(axis=0)
        z=(m-m.mean(axis=0))/np.maximum(v,1e-6)
        c=(z.T@z)/len(z)
        np.fill_diagonal(c,1.)
        return np.clip(c,-1,1)
    rc=robust_corr(rt);gc=robust_corr(gt)
    corr_gap=float(np.mean(np.abs(rc-gc)))
    trend_error=float(np.sqrt(np.mean((scaler.transform(real_train).mean(axis=0)-scaler.transform(generated).mean(axis=0))**2)))
    # RBF MMD^2 with median-distance heuristic on combined standardized samples.
    def mmd2(a,b):
        joined=np.concatenate([a,b]);dst=euclidean_distances(joined,squared=False) if False else euclidean_distances(joined)
        q=dst[np.triu_indices_from(dst,k=1)]
        q=q[q>0]
        gamma=1/(2*(np.median(q)**2+1e-8)) if len(q) else 1.0
        def kernel(c,d):return np.exp(-gamma*euclidean_distances(c,d)**2)
        return float(max(0.0,kernel(a,a).mean()+kernel(b,b).mean()-2*kernel(a,b).mean()))
    mmd=mmd2(r,g)
    per=[]
    for j,(key,name) in enumerate(zip(KEYS,NAMES)):
        a=real_train[:,:,j].ravel();b=generated[:,:,j].ravel()
        per.append({'variable':key,'中文名称':name,'train_mean':float(a.mean()),'generated_mean':float(b.mean()),
                    'train_std':float(a.std()),'generated_std':float(b.std()),
                    'wasserstein_scaled':float(wasserstein_distance(rt[:,j],gt[:,j])),
                    'ks_statistic':float(ks_2samp(rt[:,j],gt[:,j]).statistic),
                    'lag1_abs_gap':float(abs(lag_avg(scaler.transform(real_train))[j]-lag_avg(scaler.transform(generated))[j]))})
    pd.DataFrame(per).to_csv(root/f'{tag}_per_variable.csv',index=False,encoding='utf-8-sig')
    dvr=euclidean_distances(v,g)
    def financial_ratios(z):
        ix={k:i for i,k in enumerate(KEYS)}
        div=lambda numerator,denom: z[:,:,ix[numerator]]/(np.maximum(np.abs(z[:,:,ix[denom]]),1.0))
        return {'资产负债率':div('total_liabilities','total_assets'),
                '流动比率':div('current_assets','current_liabilities'),
                '净利率':div('net_income','revenue'),
                '经营现金流收入比':div('operating_cash_flow','revenue')}
    fr=financial_ratios(real_train);fg=financial_ratios(generated)
    ratio_report={}
    for name in fr:
        a=fr[name].ravel();b=fg[name].ravel()
        lo,hi=np.quantile(a,[.01,.99])
        ratio_report[name]={'train_median':float(np.median(a)),
             'generated_median':float(np.median(b)),
             'generated_outside_train_1_99_fraction':float(np.mean((b<lo)|(b>hi)))}
    report={'label':tag,'reference_train_companies':len(r),'reference_validation_companies':len(v),
            'generated_companies':len(g),'cosine_nn_train_mean':float(cos_nn.mean()),
            'cosine_nn_train_std':float(cos_nn.std()),'coverage_train':coverage,
            'coverage_radius_train_loo_p95':radius,
            'nearest_train_distance_mean':float(nearest_train.mean()),
            'nearest_train_distance_min':float(nearest_train.min()),
            'training_loo_nearest_median':real_variability,
            'nearest_distance_ratio_median':float(np.median(nearest_train)/(real_variability+1e-9)),
            'generated_nearest_neighbor_median':_scalar(diversity),
            'generated_diversity_relative_to_real':_scalar(diversity/(real_variability+1e-9)),
            'wasserstein_scaled_mean':float(np.mean([p['wasserstein_scaled'] for p in per])),
            'ks_statistic_mean':float(np.mean([p['ks_statistic'] for p in per])),
            'correlation_matrix_mae':corr_gap,
            'mean_trajectory_rmse_standardized':trend_error,
            'lag1_gap_mean':float(np.mean([p['lag1_abs_gap'] for p in per])),
            'rbf_mmd2':mmd,
            'validation_nn_distance_mean':float(dvr.min(axis=1).mean()),
            'financial_ratio_diagnostics':ratio_report,
            'generated_rule_violation_rates':violation_rates(generated),
            'train_rule_violation_rates':violation_rates(real_train),
            'note':'模拟数据；风险训练样本仅10家，coverage/COS/MMD均需解释定义与置信不确定性；test未使用。'}
    if raw_generated is not None:
        norm_raw=scaler.transform(raw_generated)
        norm_post=scaler.transform(generated)
        report['postprocessing_standardized_rmse']=float(np.sqrt(np.mean((norm_raw-norm_post)**2)))
        report['pre_projection_rule_violations']=violation_rates(raw_generated)
        report['postprocessing_changed_point_fraction']=float(np.mean(np.abs(norm_raw-norm_post)>1e-4))
    with (root/f'{tag}_summary.json').open('w',encoding='utf-8') as f:
        json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        both=np.vstack([r,g]);coords=PCA(n_components=2).fit_transform(both)
        fig,ax=plt.subplots(figsize=(7,5));ax.scatter(coords[:len(r),0],coords[:len(r),1],s=36,label='train real')
        ax.scatter(coords[len(r):,0],coords[len(r):,1],s=16,alpha=.55,label='generated')
        ax.set(xlabel='PC1',ylabel='PC2',title=f'TimeVAE distribution: {tag}');ax.legend()
        fig.tight_layout();fig.savefig(root/f'{tag}_pca.png',dpi=170);plt.close(fig)
        fig,axes=plt.subplots(3,5,figsize=(19,10))
        for j,ax in enumerate(axes.flat):
            a=scaler.transform(real_train)[:,:,j];b=scaler.transform(generated)[:,:,j]
            ax.plot(np.arange(1,21),a.mean(axis=0),label='train real')
            ax.plot(np.arange(1,21),b.mean(axis=0),label='generated')
            ax.set_title(KEYS[j],fontsize=8)
        axes[0,0].legend(fontsize=7);fig.tight_layout();fig.savefig(root/f'{tag}_mean_trajectories.png',dpi=150);plt.close(fig)
    except ImportError:
        pass
    return report
