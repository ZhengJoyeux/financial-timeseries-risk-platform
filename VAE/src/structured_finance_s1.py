"""S1: tail-aware coefficient sampler, inheriting S0's accounting/time dynamics.

Only the *enterprise-level joint-coefficient sampling* is modified. Seasonal
patterns, residual dynamics and deterministic accounting decode remain S0's.

This is a statistical SIMULATED-data baseline, not a TimeVAE checkpoint.
"""
from __future__ import annotations

import numpy as np

from .dataset import FinancialScaler
from .structured_finance import (
    D, StructuredFinanceBaseline, _design_matrix,
    _driver_decode, _driver_decode_cuda,
)


class StructuredFinanceS1(StructuredFinanceBaseline):
    def __init__(self, seed=2027, perturb=0.20, device='cpu',
                 novelty_fraction=0.50, max_draw_multiplier=16):
        super().__init__(seed=seed, perturb=perturb, device=device)
        self.novelty_fraction = float(novelty_fraction)
        self.max_draw_multiplier = int(max_draw_multiplier)
        if not (0 <= self.novelty_fraction <= 1.0):
            raise ValueError('novelty_fraction must be 0..1')
        if not (2 <= self.max_draw_multiplier <= 30):
            raise ValueError('max_draw_multiplier must be in 2..30')
        self.scaler = None
        self.references = {}
        self.novelty_threshold = {}
        self.last_sample_diagnostics = {}

    def fit(self, x, labels):
        x = np.asarray(x, dtype=np.float64)
        labels = np.asarray(labels)
        super().fit(x, labels)
        self.scaler = FinancialScaler().fit(x)
        # Train-only linear conditional correction: when a sampled size quantile
        # differs from blended donors, preserve size-driver covariance instead
        # of treating the new asset scale as independent from every ratio.
        for label in ('有风险', '无风险'):
            coeff=self.groups[label]['coeff']
            centered=coeff-coeff.mean(axis=0) if not self.device.startswith('cuda') else coeff-coeff.mean(dim=0)
            size_center=centered[:,0,0]
            if self.device.startswith('cuda'):
                import torch
                beta=(centered*size_center[:,None,None]).mean(dim=0)/(size_center.square().mean()+1e-7)
            else:
                beta=(centered*size_center[:,None,None]).mean(axis=0)/(np.mean(size_center**2)+1e-7)
            beta=beta*.80  # finite-sample shrinkage, no validation fitting
            beta[0,0]=0.
            self.groups[label]['asset_cond_beta']=beta
        for label in ('有风险', '无风险'):
            ref = self.scaler.transform(x[labels == label]).reshape(-1,300).astype(np.float64)
            if self.device.startswith('cuda'):
                import torch
                torchdev = torch.device(self.device)
                tr = torch.as_tensor(ref, dtype=torch.float64, device=torchdev)
                d = torch.cdist(tr, tr)
                d.fill_diagonal_(float('inf'))
                threshold = float(torch.quantile(d.min(dim=1).values, 0.5).item())
                self.references[label] = tr
            else:
                from scipy.spatial.distance import cdist
                d = cdist(ref, ref)
                np.fill_diagonal(d, np.inf)
                threshold = float(np.median(d.min(axis=1)))
                self.references[label] = ref
            self.novelty_threshold[label] = self.novelty_fraction * threshold
        return self

    def _candidates_cpu(self, label, u, rng):
        n = len(u)
        g = self.groups[label]
        c = g['coeff']
        ntrain = g['n_train']
        # Sample a class-conditional empirical asset-scale quantile, rather than
        # shrinking all firms toward the class mean. Quantiles are clamped at
        # the observed endpoints, with only a separately bounded jitter.
        size = c[:,0,0]
        ordered = np.argsort(size)
        sorted_values = size[ordered]
        midpoint = (np.arange(ntrain)+.5)/ntrain
        target = np.interp(u, midpoint, sorted_values)
        nearest = np.clip(np.searchsorted(midpoint,u),0,ntrain-1)
        prev = np.maximum(nearest-1,0)
        use_prev = np.abs(u-midpoint[prev]) < np.abs(u-midpoint[nearest])
        ranks = np.where(use_prev,prev,nearest)
        i1 = ordered[ranks]
        i2 = rng.integers(0,ntrain-1,size=n)
        i2 += (i2 >= i1)
        # Mix OTHER correlated drivers across firms, retain a sampled asset tail.
        weight = rng.uniform(-.25,1.25,size=(n,1,1))
        coeff = weight*c[i1] + (1-weight)*c[i2]
        coeff += rng.standard_normal(coeff.shape)*self.perturb*g['sd'][None]
        asset_noise = rng.standard_normal(n)*(.045*max(float(g['sd'][0,0]), .02))
        adjusted = np.clip(target+asset_noise,sorted_values[0]-.08,sorted_values[-1]+.08)
        coeff = coeff + (adjusted-coeff[:,0,0])[:,None,None]*g['asset_cond_beta'][None]
        coeff[:,0,0] = adjusted
        baseline = np.einsum('tp,npd->ntd',_design_matrix()[:,:2],coeff)
        baseline += g['season'][np.arange(20)%4][None]
        phi = g['phi'];cov = g['cov'];sigma = np.sqrt(np.maximum(1-phi**2,1e-3))
        noise = rng.multivariate_normal(np.zeros(D),cov,size=(n,20))
        residual = np.zeros_like(noise)
        residual[:,0]=noise[:,0]
        for t in range(1,20):
            residual[:,t]=phi*residual[:,t-1]+sigma*noise[:,t]
        return _driver_decode(baseline+residual,self.workcap_beta)

    def _candidates_gpu(self, label, u, generator):
        n = len(u)
        import torch
        dev = torch.device(self.device)
        g = self.groups[label]
        c = g['coeff'];ntrain = g['n_train']
        size = c[:,0,0]
        sorted_values, ordered = torch.sort(size)
        midpoint = (torch.arange(ntrain,device=dev,dtype=torch.float64)+.5)/ntrain
        upper = torch.searchsorted(midpoint,u).clamp(1,ntrain-1)
        low = upper-1
        w = ((u-midpoint[low])/(midpoint[upper]-midpoint[low])).clamp(0,1)
        target = sorted_values[low]*(1-w)+sorted_values[upper]*w
        rank = torch.where((u-midpoint[low]).abs() < (u-midpoint[upper]).abs(),low,upper)
        i1 = ordered[rank]
        i2 = torch.randint(ntrain-1,(n,),generator=generator,device=dev)
        i2 = i2+(i2>=i1).long()
        weights = -.25+1.50*torch.rand((n,1,1),generator=generator,device=dev,dtype=torch.float64)
        coeff=weights*c[i1]+(1-weights)*c[i2]
        coeff=coeff+torch.randn(coeff.shape,device=dev,dtype=torch.float64,generator=generator)*self.perturb*g['sd'][None]
        asset_noise=torch.randn((n,),device=dev,dtype=torch.float64,generator=generator)*(.045*torch.clamp(g['sd'][0,0],min=.02))
        adjusted = (target+asset_noise).clamp(min=sorted_values[0]-.08,max=sorted_values[-1]+.08)
        coeff = coeff + (adjusted-coeff[:,0,0])[:,None,None]*g['asset_cond_beta'][None]
        coeff[:,0,0] = adjusted
        basis=torch.as_tensor(_design_matrix()[:,:2],device=dev,dtype=torch.float64)
        baseline=torch.einsum('tp,npd->ntd',basis,coeff)
        baseline=baseline+g['season'][(torch.arange(20,device=dev)%4)][None]
        chol=torch.linalg.cholesky(g['cov']+torch.eye(D,dtype=torch.float64,device=dev)*1e-7)
        noise=torch.randn((n,20,D),device=dev,dtype=torch.float64,generator=generator)@chol.T
        phi=g['phi'];sigma=torch.sqrt((1-phi**2).clamp(min=1e-3))
        residual=torch.zeros_like(noise);residual[:,0]=noise[:,0]
        for t in range(1,20):
            residual[:,t]=phi*residual[:,t-1]+sigma*noise[:,t]
        return _driver_decode_cuda(baseline+residual,self.workcap_beta)

    def _std_gpu(self, output):
        import torch
        sc=self.scaler
        device=output.device
        scale=torch.as_tensor(sc.scale,device=device,dtype=torch.float64)
        mean=torch.as_tensor(sc.mean,device=device,dtype=torch.float64)
        std=torch.as_tensor(sc.std,device=device,dtype=torch.float64)
        return ((torch.asinh(output/scale)-mean)/std).reshape(len(output),300)

    def sample(self, label, number, seed=None):
        if label not in self.groups:raise ValueError('Unknown fitted class')
        if number <= 0:raise ValueError('number must be > 0')
        n=int(number)
        seed=int(self.seed if seed is None else seed)
        tries=self.max_draw_multiplier
        threshold=self.novelty_threshold[label]
        # IMPORTANT: Preserve the asset-scale quantile distribution under the
        # novelty check. Each desired quantile gets its own K attempts. Global
        # rejection biases sampling toward already-distant, typically LARGE firms.
        if self.device.startswith('cuda'):
            import torch
            dev=torch.device(self.device)
            gen=torch.Generator(device=dev);gen.manual_seed(seed)
            u=(torch.arange(n,device=dev,dtype=torch.float64)
               +torch.rand(n,device=dev,dtype=torch.float64,generator=gen))/n
            u=u[torch.randperm(n,device=dev,generator=gen)]
            u_rep=u[:,None].expand(n,tries).reshape(-1)
            proposals=self._candidates_gpu(label,u_rep,gen)
            distances=torch.cdist(self._std_gpu(proposals),self.references[label]).min(dim=1).values.reshape(n,tries)
            mask=distances>=threshold
            passed=mask.any(dim=1)
            index=torch.where(passed,mask.to(torch.int32).argmax(dim=1),distances.argmax(dim=1))
            failed=int((~passed).sum().item())
            result=proposals.reshape(n,tries,20,15)[torch.arange(n,device=dev),index]
            selected_dist=distances[torch.arange(n,device=dev),index]
            discarded=int((~mask).sum().item())
            mean_tries=float((index.to(torch.float64)+1).mean().item())
            result=result.detach().cpu().numpy()
            closest=float(selected_dist.min().item())
        else:
            from scipy.spatial.distance import cdist
            rng=np.random.default_rng(seed)
            u=(np.arange(n)+rng.random(n))/n
            rng.shuffle(u)
            u_rep=np.repeat(u,tries)
            proposals=self._candidates_cpu(label,u_rep,rng)
            flat=self.scaler.transform(proposals).reshape(n*tries,300).astype(float)
            distances=cdist(flat,self.references[label]).min(axis=1).reshape(n,tries)
            mask=distances>=threshold
            passed=mask.any(axis=1)
            indices=np.where(passed,mask.argmax(axis=1),distances.argmax(axis=1))
            failed=int(np.sum(~passed))
            result=proposals.reshape(n,tries,20,15)[np.arange(n),indices]
            selected_dist=distances[np.arange(n),indices]
            discarded=int(np.sum(~mask))
            mean_tries=float(np.mean(indices+1))
            closest=float(selected_dist.min())
        self.last_sample_diagnostics[label]={
            'candidate_draws':n*tries,
            'accepted':n,
            'rejected_too_close_among_all_candidates':discarded,
            'mean_attempts_to_first_pass':mean_tries,
            'minimum_accepted_nearest_train_distance':closest,
            'nearest_neighbor_threshold_standardized_euclidean':threshold,
            'novelty_guard_fraction_of_train_loo_median':self.novelty_fraction,
            'stratified_quantile_slots':n,
            'quantile_slots_without_threshold_pass_fallback_to_max_distance':failed,
            'threshold_satisfaction_fraction':float((n-failed)/n),
            'note':'Each asset quantile keeps its slot; if all tries are close, choose farthest available and report fallback. No novelty guarantee or differential privacy.',
        }
        return result

    def stats(self):
        out=super().stats()
        out.update({
            'name':'structured_finance_s1_tail_aware',
            'sampler_change':'empirical asset-intercept quantile + joint donor mixing + quantile-preserving proximity-aware selection',
            'asset_quantile_limited_jitter_log_margin':0.08,
            'coefficient_donor_mix_range':[-0.25,1.25],
            'asset_conditional_covariance_shrinkage':0.80,
            'perturb':self.perturb,
            'novelty_fraction':self.novelty_fraction,
            'max_draw_multiplier':self.max_draw_multiplier,
            'train_nearest_threshold_by_class':self.novelty_threshold,
            'warning':'n_risk=10; class tail extrapolation not statistically established',
        })
        return out
