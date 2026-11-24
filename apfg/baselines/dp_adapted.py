"""ADAPTED high-level DP using the pinned MIT official ConditionalUnet1D.

The observation is one sensor-derived feature vector plus legal-pool summary;
actions are two high-level candidate-feature tokens, not native EE trajectories.
The epsilon DDPM scheduler is independently implemented, without diffusers.
"""
from pathlib import Path
import math
import torch
from torch import nn

from apfg.baselines.schema import OBS_DIM,CAND_DIM,SCHEMA_VERSION
from apfg.baselines._official_dp.conditional_unet1d import ConditionalUnet1D


def condition_features(obs,candidates,mask):
    mask=mask.bool()
    if not mask.any(dim=1).all():raise ValueError('No legal candidate in diffusion condition')
    count=mask.sum(dim=1,keepdim=True).clamp_min(1)
    mean=(candidates*mask.unsqueeze(-1)).sum(dim=1)/count
    maximum=candidates.masked_fill(~mask.unsqueeze(-1),-torch.inf).max(dim=1).values
    return torch.cat([obs,mean,maximum],dim=-1)


class DiffusionHighLevelPolicy(nn.Module):
    def __init__(self,horizon=2,diffusion_steps=32,channels=(32,64),time_dim=32):
        super().__init__()
        if horizon<2 or horizon%2:raise ValueError('Official two-level U-Net requires a positive even horizon >=2')
        if diffusion_steps<2:raise ValueError('At least two diffusion steps required')
        self.horizon=int(horizon);self.diffusion_steps=int(diffusion_steps);self.channels=tuple(channels);self.time_dim=time_dim
        self.denoiser=ConditionalUnet1D(CAND_DIM,global_cond_dim=OBS_DIM+2*CAND_DIM,
            diffusion_step_embed_dim=time_dim,down_dims=list(channels),kernel_size=3,n_groups=4,cond_predict_scale=True)
        # Same squared-cosine schedule family as the pinned official DP config.
        # Unlike a 32-step unscaled linear schedule, its final state is nearly
        # pure Gaussian noise, matching the reverse sampler's initial state.
        grid=torch.linspace(0,1,diffusion_steps+1,dtype=torch.float64)
        cosine=torch.cos((grid+.008)/1.008*math.pi/2).square()
        beta=(1-cosine[1:]/cosine[:-1]).clamp(1e-5,.999).float()
        alpha=1-beta;abar=torch.cumprod(alpha,dim=0)
        self.register_buffer('beta',beta);self.register_buffer('alpha',alpha);self.register_buffer('abar',abar)
        self.register_buffer('action_center',torch.zeros(CAND_DIM));self.register_buffer('action_scale',torch.ones(CAND_DIM))

    def fit_action_normalizer(self,actions):
        flat=actions.reshape(-1,CAND_DIM).to(self.action_center.device)
        lower=flat.min(dim=0).values;upper=flat.max(dim=0).values
        self.action_center.copy_((lower+upper)/2);self.action_scale.copy_(((upper-lower)/2).clamp_min(.1))

    def normalize(self,actions):return (actions-self.action_center)/self.action_scale
    def unnormalize(self,actions):return actions*self.action_scale+self.action_center

    def noise_loss(self,obs,candidates,mask,actions,sequence_mask=None,generator=None):
        actions=self.normalize(actions);batch=actions.shape[0]
        time=torch.randint(self.diffusion_steps,(batch,),device=actions.device,generator=generator)
        noise=torch.randn(actions.shape,device=actions.device,dtype=actions.dtype,generator=generator)
        abar=self.abar[time].reshape(batch,1,1)
        noisy=abar.sqrt()*actions+(1-abar).sqrt()*noise
        prediction=self.denoiser(noisy,time,global_cond=condition_features(obs,candidates,mask))
        losses=(prediction-noise).square().mean(dim=-1)
        if sequence_mask is None:return losses.mean()
        return (losses*sequence_mask).sum()/sequence_mask.sum().clamp_min(1)

    @torch.no_grad()
    def sample(self,obs,candidates,mask,seed=0):
        self.eval();condition=condition_features(obs,candidates,mask)
        generator=torch.Generator(device=obs.device).manual_seed(int(seed))
        trajectory=torch.randn((obs.shape[0],self.horizon,CAND_DIM),device=obs.device,generator=generator)
        for t in range(self.diffusion_steps-1,-1,-1):
            time=torch.full((obs.shape[0],),t,device=obs.device,dtype=torch.long)
            epsilon=self.denoiser(trajectory,time,global_cond=condition)
            # DDPM epsilon prediction, exact posterior variance, consecutive
            # reverse steps. No classifier or label fallback is mixed in.
            previous_abar=self.abar[t-1] if t else torch.ones((),device=obs.device)
            clean=((trajectory-(1-self.abar[t]).sqrt()*epsilon)/self.abar[t].sqrt()).clamp(-1,1)
            mean=(self.beta[t]*previous_abar.sqrt()*clean+(1-previous_abar)*self.alpha[t].sqrt()*trajectory)/(1-self.abar[t])
            if t:
                variance=self.beta[t]*(1-previous_abar)/(1-self.abar[t])
                noise=torch.randn(trajectory.shape,device=obs.device,generator=generator)
                trajectory=mean+variance.clamp_min(1e-20).sqrt()*noise
            else:trajectory=mean
        return self.unnormalize(trajectory)

    @torch.no_grad()
    def act(self,state,seed=0):
        device=self.beta.device
        mask=torch.as_tensor(state['mask'],dtype=torch.bool,device=device).unsqueeze(0)
        if not mask.any():return -1
        obs=torch.as_tensor(state['obs'],dtype=torch.float32,device=device).unsqueeze(0)
        candidates=torch.as_tensor(state['candidates'],dtype=torch.float32,device=device).unsqueeze(0)
        action=self.sample(obs,candidates,mask,seed=seed)[:,0]
        distances=((self.normalize(candidates)-self.normalize(action).unsqueeze(1))**2).mean(dim=-1)
        distances=distances.masked_fill(~mask,torch.inf)
        return int(distances.argmin(dim=1).item())

    def save(self,path,metadata=None):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        torch.save({'schema_version':SCHEMA_VERSION,'method':'DP_HIGH_LEVEL_ADAPTED_OFFICIAL_UNET',
            'horizon':self.horizon,'diffusion_steps':self.diffusion_steps,'channels':list(self.channels),'time_dim':self.time_dim,
            'model_state':self.state_dict(),'metadata':metadata or {}},path)

    @classmethod
    def load(cls,path,device='cpu'):
        checkpoint=torch.load(path,map_location=device,weights_only=True)
        if checkpoint['schema_version']!=SCHEMA_VERSION:raise ValueError('DP checkpoint schema mismatch')
        model=cls(checkpoint['horizon'],checkpoint['diffusion_steps'],tuple(checkpoint['channels']),checkpoint['time_dim']).to(device)
        model.load_state_dict(checkpoint['model_state']);model.eval();model.checkpoint_metadata=checkpoint.get('metadata',{})
        return model
