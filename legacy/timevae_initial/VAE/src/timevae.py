"""PyTorch *conditional* TimeVAE variant; risk label controls generation of one joint model.
Inspired by Desai et al.; conditional one-hot injection is this project's design choice.
"""
import torch
from torch import nn
import torch.nn.functional as F


class TimeVAE(nn.Module):
    def __init__(self, seq_len=20, feature_dim=15, latent_dim=8, trend_degree=2,
                 use_base=True, use_trend=True, use_seasonality=True, conditional=True):
        super().__init__()
        if seq_len != 20 or feature_dim != 15:
            raise ValueError('当前实验仅支持20季度×15指标。')
        if not (use_base or use_trend or use_seasonality):
            raise ValueError('解码分支至少开启一个。')
        if not conditional:
            raise ValueError('本次要求联合训练并按类别生成，必须使用条件编码。')
        self.params = dict(seq_len=seq_len,feature_dim=feature_dim,latent_dim=latent_dim,
                           trend_degree=trend_degree,use_base=use_base,
                           use_trend=use_trend,use_seasonality=use_seasonality,conditional=True)
        condition_dim=2
        self.enc=nn.Sequential(nn.Conv1d(feature_dim+condition_dim,32,4,2,1),nn.ReLU(),
                               nn.Conv1d(32,64,4,2,1),nn.ReLU(),nn.Flatten())
        self.mu=nn.Linear(64*5,latent_dim)
        self.logvar=nn.Linear(64*5,latent_dim)
        self.use_base=use_base
        self.use_trend=use_trend
        self.use_seasonality=use_seasonality
        dec_dim=latent_dim+condition_dim
        if use_base:
            self.base_fc=nn.Linear(dec_dim,64*5)
            self.base_dec=nn.Sequential(nn.ConvTranspose1d(64,32,4,2,1),nn.ReLU(),
                                        nn.ConvTranspose1d(32,32,4,2,1),nn.ReLU(),
                                        nn.Conv1d(32,feature_dim,3,padding=1))
        if use_trend:
            t=torch.arange(seq_len,dtype=torch.float32)/seq_len
            self.register_buffer('trend_basis',torch.stack([t**i for i in range(trend_degree+1)],dim=0))
            self.trend_fc=nn.Linear(dec_dim,feature_dim*(trend_degree+1))
        if use_seasonality:
            self.season_fc=nn.Linear(dec_dim,feature_dim*4)
            self.register_buffer('season_idx',torch.arange(seq_len)%4)

    @staticmethod
    def _condition(y, batch_size, device):
        if y is None:
            raise ValueError('条件TimeVAE必须提供y：0无风险、1有风险。')
        if not isinstance(y,torch.Tensor):
            y=torch.as_tensor(y,dtype=torch.long,device=device)
        else:
            y=y.to(device=device,dtype=torch.long)
        if y.ndim != 1 or y.shape[0] != batch_size or not torch.all((y==0)|(y==1)):
            raise ValueError('标签必须是batch长度的一维0/1张量。')
        return F.one_hot(y,num_classes=2).float()

    def encode(self,x,y):
        condition=self._condition(y,x.shape[0],x.device)
        x_and_label=torch.cat([x,condition[:,None,:].expand(-1,20,-1)],dim=2)
        h=self.enc(x_and_label.transpose(1,2))
        return self.mu(h),self.logvar(h).clamp(-12,12)

    def decode(self,z,y):
        batch=z.shape[0]
        c=self._condition(y,batch,z.device)
        joined=torch.cat([z,c],dim=1)
        out=torch.zeros((batch,20,15),dtype=z.dtype,device=z.device)
        if self.use_base:
            h=self.base_fc(joined).reshape(batch,64,5)
            out=out+self.base_dec(h).transpose(1,2)
        if self.use_trend:
            coef=self.trend_fc(joined).reshape(batch,15,-1)
            out=out+torch.matmul(coef,self.trend_basis).transpose(1,2)
        if self.use_seasonality:
            coef=self.season_fc(joined).reshape(batch,15,4)
            out=out+coef[:,:,self.season_idx].transpose(1,2)
        return out

    def forward(self,x,y):
        mu,logvar=self.encode(x,y)
        z=mu+torch.randn_like(mu)*torch.exp(.5*logvar)
        return self.decode(z,y),mu,logvar

    def loss(self,x,recon,mu,logvar,reconstruction_weight=3.0,beta_kl=.025,temporal_weight=.1):
        mse=F.mse_loss(recon,x)
        trend=F.mse_loss(recon[:,1:]-recon[:,:-1],x[:,1:]-x[:,:-1])
        kl=-.5*(1+logvar-mu.pow(2)-logvar.exp()).sum(dim=1).mean()
        total=reconstruction_weight*mse+temporal_weight*trend+beta_kl*kl
        return total,dict(reconstruction=float(mse.detach()),temporal=float(trend.detach()),kl=float(kl.detach()))

    @torch.no_grad()
    def sample(self,number,device,condition,temperature=1.,batch_size=256):
        if condition not in (0,1):raise ValueError('condition=0（无风险）或1（有风险）。')
        result=[]
        self.eval()
        for i in range(0,number,batch_size):
            sz=min(batch_size,number-i)
            z=torch.randn((sz,self.params['latent_dim']),device=device)*temperature
            labels=torch.full((sz,),condition,dtype=torch.long,device=device)
            result.append(self.decode(z,labels).cpu().numpy())
        return __import__('numpy').concatenate(result,axis=0)
