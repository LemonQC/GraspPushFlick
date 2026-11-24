"""FROM_SCRATCH candidate-conditioned supervised behavior cloning baseline."""
from pathlib import Path
import torch
from torch import nn

from apfg.baselines.schema import OBS_DIM,CAND_DIM,SCHEMA_VERSION


class BCPolicy(nn.Module):
    def __init__(self,hidden=64):
        super().__init__();self.hidden=hidden
        self.scorer=nn.Sequential(nn.Linear(OBS_DIM+CAND_DIM,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,1))

    def forward(self,obs,candidates,mask):
        expanded=obs.unsqueeze(1).expand(-1,candidates.shape[1],-1)
        logits=self.scorer(torch.cat([expanded,candidates],dim=-1)).squeeze(-1)
        return logits.masked_fill(~mask.bool(),torch.finfo(logits.dtype).min)

    @torch.no_grad()
    def act(self,state,seed=None):
        device=next(self.parameters()).device
        mask=torch.as_tensor(state['mask'],dtype=torch.bool,device=device).unsqueeze(0)
        if not mask.any():return -1
        obs=torch.as_tensor(state['obs'],dtype=torch.float32,device=device).unsqueeze(0)
        candidates=torch.as_tensor(state['candidates'],dtype=torch.float32,device=device).unsqueeze(0)
        self.eval();return int(self(obs,candidates,mask).argmax(dim=-1).item())

    def save(self,path,metadata=None):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        torch.save({'schema_version':SCHEMA_VERSION,'method':'BC_FROM_SCRATCH','hidden':self.hidden,
                    'model_state':self.state_dict(),'metadata':metadata or {}},path)

    @classmethod
    def load(cls,path,device='cpu'):
        checkpoint=torch.load(path,map_location=device,weights_only=True)
        if checkpoint['schema_version']!=SCHEMA_VERSION:raise ValueError('BC checkpoint schema mismatch')
        model=cls(checkpoint['hidden']).to(device);model.load_state_dict(checkpoint['model_state']);model.eval()
        model.checkpoint_metadata=checkpoint.get('metadata',{})
        return model
