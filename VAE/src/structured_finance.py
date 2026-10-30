"""Train-only hierarchical structured generator for SIMULATED quarterly company data.

This is a non-neural baseline, NOT a trained TimeVAE or proof of realistic finance.
Fit company-level joint coefficients and class-specific innovation statistics;
reconstruct 15 reported fields from interpretable financial drivers.
"""
from __future__ import annotations

import hashlib
import numpy as np
from .schema import KEYS
from .finance_rules import violation_rates

J = {key: i for i, key in enumerate(KEYS)}
DRIVER_NAMES = (
    'log_assets', 'leverage_logit', 'current_asset_logit',
    'current_liability_logit', 'long_debt_logit',
    'cash_stick_logit', 'inventory_stick_logit', 'receivables_stick_logit',
    'log_revenue_to_assets', 'margin_logit', 'log_operating_cost_ratio',
    'log_depreciation_to_assets', 'log_ebit_net_gap_ratio',
    'asinh_operating_cash_residual',
)
D = len(DRIVER_NAMES)
STICK_LIMIT = .93
# This single seasonal column is permitted in the quarterly revenue/asset driver only.
SEASONAL_DRIVER = 8
EPS = 1e-9


def _logit(x):
    v = np.clip(x, 1e-5, 1-1e-5)
    return np.log(v/(1-v))


def _sigmoid(x):
    z = np.clip(x, -25, 25)
    return 1/(1+np.exp(-z))


def _driver_encode(x, workcap_beta):
    """x: N x 20 x 15, raw accounting units, training only."""
    z = np.asarray(x, dtype=np.float64)
    if z.ndim != 3 or z.shape[1:] != (20,15) or not np.isfinite(z).all():
        raise ValueError('Expected finite (N,20,15) finance tensor')
    a=np.maximum(z[...,J['total_assets']], EPS)
    li=np.maximum(z[...,J['total_liabilities']], EPS)
    ca=np.maximum(z[...,J['current_assets']], EPS)
    cl=np.maximum(z[...,J['current_liabilities']], 0)
    long=np.maximum(z[...,J['long_term_debt']], 0)
    cash=np.maximum(z[...,J['cash']],0)
    inv=np.maximum(z[...,J['inventory']],0)
    rec=np.maximum(z[...,J['accounts_receivable']],0)
    rev=np.maximum(z[...,J['revenue']], EPS)
    cogs=z[...,J['cogs']]
    gp=z[...,J['gross_profit']]
    ebit=z[...,J['ebit']]
    ebitda=z[...,J['ebitda']]
    net=z[...,J['net_income']]
    ocf=z[...,J['operating_cash_flow']]
    wc = inv+rec-cl
    dwc=np.diff(wc,axis=1,prepend=wc[:,:1])
    dest=np.empty((*z.shape[:2],D),dtype=np.float64)
    dest[...,0]=np.log(a)
    dest[...,1]=_logit((li/a)/1.20)
    dest[...,2]=_logit((ca/a)/.98)
    dest[...,3]=_logit((cl/li)/.98)
    dest[...,4]=_logit(long/np.maximum(li-cl,EPS)/.98)
    budget=STICK_LIMIT*ca
    dest[...,5]=_logit(cash/np.maximum(budget,EPS))
    budget=np.maximum(budget-cash,EPS)
    dest[...,6]=_logit(inv/budget)
    budget=np.maximum(budget-inv,EPS)
    dest[...,7]=_logit(rec/budget)
    dest[...,8]=np.log(rev/a)
    margin=(rev-cogs)/rev
    dest[...,9]=_logit((margin+.5)/1.45) # possible gross margin [-.5,.95]
    dest[...,10]=np.log(np.maximum((gp-ebit)/rev,1e-5))
    dest[...,11]=np.log(np.maximum((ebitda-ebit)/a,1e-6))
    dest[...,12]=np.log(np.maximum((ebit-net)/rev,1e-5))
    resid=(ocf-net-(ebitda-ebit)+workcap_beta*dwc)/rev
    dest[...,13]=np.arcsinh(resid/.10)
    if not np.isfinite(dest).all():raise ValueError('Invalid transformed financial drivers')
    return dest


def _driver_decode(d, workcap_beta):
    d=np.asarray(d,dtype=np.float64)
    if d.ndim!=3 or d.shape[1:]!=(20,D):raise ValueError('Wrong driver shape')
    z=np.zeros((len(d),20,15),dtype=np.float64)
    A=np.exp(np.clip(d[...,0],np.log(250),np.log(2e7)))
    L=A*(1.20*_sigmoid(d[...,1]))
    CA=A*(.98*_sigmoid(d[...,2]))
    CL=L*(.98*_sigmoid(d[...,3]))
    LTD=(L-CL)*(.98*_sigmoid(d[...,4]))
    budget=STICK_LIMIT*CA
    cash=budget*_sigmoid(d[...,5]);budget-=cash
    inv=budget*_sigmoid(d[...,6]);budget-=inv
    rec=budget*_sigmoid(d[...,7])
    revenue=A*np.exp(np.clip(d[...,8],-10,3.5))
    margin=-.5 + 1.45*_sigmoid(d[...,9])
    cogs=revenue*(1-margin)
    gross=revenue-cogs
    op_cost=revenue*np.exp(np.clip(d[...,10],-10,2.))
    ebit=gross-op_cost
    depr=A*np.exp(np.clip(d[...,11],-13,-1.5))
    ebitda=ebit+depr
    net_gap=revenue*np.exp(np.clip(d[...,12],-11,2.))
    net=ebit-net_gap
    wc=inv+rec-CL
    dwc=np.diff(wc,axis=1,prepend=wc[:,:1])
    resid=.10*np.sinh(np.clip(d[...,13],-6,6))*revenue
    ocf=net+depr-workcap_beta*dwc+resid
    data={'total_assets':A,'total_liabilities':L,'revenue':revenue,'net_income':net,
          'current_assets':CA,'current_liabilities':CL,'cash':cash,'long_term_debt':LTD,
          'inventory':inv,'accounts_receivable':rec,'cogs':cogs,
          'gross_profit':gross,'ebit':ebit,'ebitda':ebitda,'operating_cash_flow':ocf}
    for name, v in data.items(): z[...,J[name]]=v
    if not np.isfinite(z).all():raise RuntimeError('Generator returned invalid numerical values')
    return z


def _design_matrix():
    t=np.arange(20,dtype=float)
    slope=(t-9.5)/19
    # Joint model uses only a revenue seasonality term; sum to zero across four quarters.
    seas=np.eye(4)[np.arange(20)%4]
    seas=seas-seas.mean(axis=0,keepdims=True)
    return np.column_stack([np.ones(20),slope,seas[:,:3]])


def _fit_workcap_beta(x):
    j=J
    wc=x[...,j['inventory']]+x[...,j['accounts_receivable']]-x[...,j['current_liabilities']]
    rev=np.maximum(x[...,j['revenue']],1)
    dwc=np.diff(wc,axis=1,prepend=wc[:,:1])
    u=(-dwc/rev).ravel()
    v=((x[...,j['operating_cash_flow']]-x[...,j['net_income']]
        -(x[...,j['ebitda']]-x[...,j['ebit']]))/rev).ravel()
    u=u-u.mean();v=v-v.mean()
    slope=float(np.dot(u,v)/(np.dot(u,u)+1.))
    return float(np.clip(slope,0,.6))


class StructuredFinanceBaseline:
    """No gradient training. All statistical state is fitted on train companies ONLY."""
    def __init__(self, seed=2027, perturb=0.10, device="cpu"):
        self.seed=int(seed);self.perturb=float(perturb);self.device=str(device)
        if not (0<=self.perturb<=.3):raise ValueError('perturb must be in [0,.3]')
        self.groups={};self.workcap_beta=None

    def fit(self, x, labels):
        x=np.asarray(x,dtype=np.float64);labels=np.asarray(labels)
        if x.ndim!=3 or x.shape[1:]!=(20,15):raise ValueError('Wrong tensor shape')
        if set(labels.tolist())!={'有风险','无风险'}:raise ValueError('Both training classes required')
        if not np.isfinite(x).all():raise ValueError('non-finite data')
        if max(violation_rates(x).values())>.02:raise ValueError('Training data violate accounting rules')
        if self.device.startswith('cuda'):
            import torch
            torch_device=torch.device(self.device)
            # Financial data conversion is preprocessing; statistical parameter estimation
            # and sampling are computed on the isolated physical GPU1.
            t=torch.as_tensor(x,dtype=torch.float64,device=torch_device)
            j=J
            wc=t[...,j['inventory']]+t[...,j['accounts_receivable']]-t[...,j['current_liabilities']]
            rev=torch.clamp(t[...,j['revenue']],min=1.)
            dwc=torch.diff(wc,dim=1,prepend=wc[:,:1])
            u=(-dwc/rev).reshape(-1);v=((t[...,j['operating_cash_flow']]-t[...,j['net_income']]
                     -(t[...,j['ebitda']]-t[...,j['ebit']]))/rev).reshape(-1)
            u=u-u.mean();v=v-v.mean()
            self.workcap_beta=float(torch.clamp(torch.dot(u,v)/(torch.dot(u,u)+1.),0,.6).item())
        else:
            self.workcap_beta=_fit_workcap_beta(x)
        z=_driver_encode(x,self.workcap_beta)
        if self.device.startswith('cuda'):
            self._fit_cuda(z,labels)
            return self
        design=_design_matrix();basis=design[:,:2]
        fit_basis=np.linalg.pinv(basis)
        # Seasonality estimated using only in-class training companies;
        # no seasonality allowed for the balance sheet drivers.
        for label in ['有风险','无风险']:
            sel=z[labels==label]
            if len(sel)<3:raise ValueError('At least 3 train firms per class required')
            slopes=np.einsum('pt,ntd->npd',fit_basis,sel) # n x 2 x D
            reconstructed=np.einsum('tp,npd->ntd',basis,slopes)
            residual=sel-reconstructed
            seas=np.zeros((4,D))
            by_q=np.stack([residual[:,q::4,SEASONAL_DRIVER].mean() for q in range(4)])
            seas[:,SEASONAL_DRIVER]=by_q-by_q.mean()
            reconstructed+=seas[np.arange(20)%4][None]
            residual=sel-reconstructed
            flat=residual.reshape(-1,D)
            cov=np.cov(flat,rowvar=False)
            # Finite-sample shrinkage ensures PSD and reduces spurious cross-driver coupling.
            cov=.55*cov+.45*np.diag(np.diag(cov))
            evals,vecs=np.linalg.eigh(cov)
            cov=(vecs*np.maximum(evals,1e-8))@vecs.T
            lag_num=(residual[:,:-1]*residual[:,1:]).mean(axis=(0,1))
            lag_den=(residual[:,:-1]**2).mean(axis=(0,1))+1e-8
            phi=np.clip(lag_num/lag_den,-.15,.85)
            # small samples: extra coefficient shrinkage toward overall class center
            coeff_mean=slopes.mean(axis=0)
            coeff_sd=np.maximum(slopes.std(axis=0),1e-5)
            self.groups[label]={'coeff':slopes,'mean':coeff_mean,'sd':coeff_sd,
                                'season':seas,'cov':cov,'phi':phi,'n_train':len(sel)}
        return self

    def _fit_cuda(self, z, labels):
        import torch
        dev=torch.device(self.device)
        basis=torch.as_tensor(_design_matrix()[:,:2],dtype=torch.float64,device=dev)
        fit_basis=torch.linalg.pinv(basis)
        for label in ['有风险','无风险']:
            sel=torch.as_tensor(z[labels==label],dtype=torch.float64,device=dev)
            if len(sel)<3:raise ValueError('At least 3 train firms per class required')
            slopes=torch.einsum('pt,ntd->npd',fit_basis,sel)
            recon=torch.einsum('tp,npd->ntd',basis,slopes)
            resid=sel-recon
            seas=torch.zeros((4,D),dtype=torch.float64,device=dev)
            by_q=torch.stack([resid[:,q::4,SEASONAL_DRIVER].mean() for q in range(4)])
            seas[:,SEASONAL_DRIVER]=by_q-by_q.mean()
            qidx=torch.arange(20,device=dev)%4
            resid=sel-(recon+seas[qidx][None])
            flat=resid.reshape(-1,D);c=flat-flat.mean(dim=0)
            cov=(c.T@c)/(len(flat)-1)
            cov=.55*cov+.45*torch.diag(torch.diag(cov))
            val,vec=torch.linalg.eigh(cov)
            cov=(vec*torch.clamp(val,min=1e-8))@vec.T
            phi=torch.clamp((resid[:,:-1]*resid[:,1:]).mean((0,1)) /
                            ((resid[:,:-1]**2).mean((0,1))+1e-8),-.15,.85)
            coeff_mean=slopes.mean(dim=0)
            coeff_sd=torch.clamp(slopes.std(dim=0,unbiased=False),min=1e-5)
            self.groups[label]={'coeff':slopes,'mean':coeff_mean,'sd':coeff_sd,
                                'season':seas,'cov':cov,'phi':phi,'n_train':len(sel)}

    def _sample_cuda(self,label,number,seed):
        import torch
        dev=torch.device(self.device)
        g=self.groups[label]
        r=torch.Generator(device=dev);r.manual_seed(int(seed))
        ntrain=g['n_train']
        i1=torch.randint(ntrain,(number,),generator=r,device=dev)
        i2=torch.randint(ntrain-1,(number,),generator=r,device=dev)
        i2=i2+(i2>=i1).long()
        weights=.55+.25*torch.rand((number,1,1),dtype=torch.float64,generator=r,device=dev)
        coef=weights*g['coeff'][i1]+(1-weights)*g['coeff'][i2]
        coef=.95*coef+.05*g['mean'][None]
        coef=coef+torch.randn(coef.shape,dtype=torch.float64,generator=r,device=dev)*self.perturb*g['sd'][None]
        basis=torch.as_tensor(_design_matrix()[:,:2],dtype=torch.float64,device=dev)
        latent=torch.einsum('tp,npd->ntd',basis,coef)
        latent=latent+g['season'][(torch.arange(20,device=dev)%4)][None]
        cov=g['cov'];phi=g['phi']
        # PSD jitter safeguards small-class covariance without changing labels or raw data
        ch=torch.linalg.cholesky(cov+torch.eye(D,dtype=torch.float64,device=dev)*1e-7)
        noises=torch.randn((number,20,D),dtype=torch.float64,generator=r,device=dev)@ch.T
        e=torch.zeros_like(noises);e[:,0]=noises[:,0]
        sigma=torch.sqrt(torch.clamp(1-phi**2,min=1e-3))
        for t in range(1,20):e[:,t]=phi[None]*e[:,t-1]+sigma[None]*noises[:,t]
        latent=latent+e
        return _driver_decode_cuda(latent,self.workcap_beta).detach().cpu().numpy()

    def sample(self, label, number, seed=None):
        if label not in self.groups:raise ValueError('Unknown fitted label')
        if number<=0:raise ValueError('number must be positive')
        if self.device.startswith('cuda'):
            return self._sample_cuda(label,number,self.seed if seed is None else seed)
        g=self.groups[label];rng=np.random.default_rng(self.seed if seed is None else seed)
        ntrain=g['n_train'];coeff=g['coeff']
        i1=rng.integers(0,ntrain,size=number)
        i2=rng.integers(0,ntrain-1,size=number)
        i2+=(i2>=i1) # distinct training firms; no literal single-company bootstrap
        weights=rng.uniform(.55,.80,size=(number,1,1))
        coef=weights*coeff[i1]+(1-weights)*coeff[i2]
        # modest regularization and perturbation calibrated from train-class variability
        coef=.95*coef+.05*g['mean'][None]
        coef=coef+rng.standard_normal(coef.shape)*self.perturb*g['sd'][None]
        baseline=np.einsum('tp,npd->ntd',_design_matrix()[:,:2],coef)
        baseline+=g['season'][np.arange(20)%4][None]
        # correlated and autocorrelated residuals fitted exclusively from train data
        phi=g['phi'];cov=g['cov'];sigma=np.sqrt(np.maximum(1-phi**2,1e-3))
        noise=rng.multivariate_normal(np.zeros(D),cov,size=(number,20))
        e=np.zeros_like(noise);e[:,0]=noise[:,0]
        for t in range(1,20):
            e[:,t]=phi[None]*e[:,t-1]+sigma[None]*noise[:,t]
        latent=baseline+e
        output=_driver_decode(latent,self.workcap_beta)
        return output

    def stats(self):
        return {'name':'structured_finance_baseline_v1',
                'not_timevae':True,'seed':self.seed,'perturb':self.perturb,
                'fit_and_sampling_device':self.device,
                'workcap_beta_fit_from_train':self.workcap_beta,
                'class_firm_counts':{k:int(v['n_train']) for k,v in self.groups.items()},
                'driver_names':list(DRIVER_NAMES),
                'constraints':'deterministic accounting composition; raw output not projected',
                'caveat':'simulated training enterprises; bootstrapped joint coefficients; risk n=10'}


def _driver_decode_cuda(d,beta):
    import torch
    d=d.to(dtype=torch.float64)
    sigmoid=lambda z:torch.sigmoid(z.clamp(-25,25))
    A=torch.exp(d[...,0].clamp(np.log(250),np.log(2e7)))
    L=A*(1.20*sigmoid(d[...,1]))
    CA=A*(.98*sigmoid(d[...,2]))
    CL=L*(.98*sigmoid(d[...,3]))
    LTD=(L-CL)*(.98*sigmoid(d[...,4]))
    budget=.93*CA
    cash=budget*sigmoid(d[...,5]);budget=budget-cash
    inv=budget*sigmoid(d[...,6]);budget=budget-inv
    rec=budget*sigmoid(d[...,7])
    revenue=A*torch.exp(d[...,8].clamp(-10,3.5))
    margin=-.5+1.45*sigmoid(d[...,9])
    cogs=revenue*(1-margin)
    gross=revenue-cogs
    op_cost=revenue*torch.exp(d[...,10].clamp(-10,2))
    ebit=gross-op_cost
    depr=A*torch.exp(d[...,11].clamp(-13,-1.5))
    ebitda=ebit+depr
    net=ebit-revenue*torch.exp(d[...,12].clamp(-11,2))
    wc=inv+rec-CL
    dwc=torch.diff(wc,dim=1,prepend=wc[:,:1])
    ocf=net+depr-beta*dwc+.10*torch.sinh(d[...,13].clamp(-6,6))*revenue
    values={'total_assets':A,'total_liabilities':L,'revenue':revenue,'net_income':net,
          'current_assets':CA,'current_liabilities':CL,'cash':cash,'long_term_debt':LTD,
          'inventory':inv,'accounts_receivable':rec,'cogs':cogs,
          'gross_profit':gross,'ebit':ebit,'ebitda':ebitda,'operating_cash_flow':ocf}
    return torch.stack([values[key] for key in KEYS],dim=-1)
