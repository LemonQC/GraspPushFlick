"""From-scratch candidate-conditioned Double DQN for the shared Track A API.

The network consumes only sensor-derived numeric observations/candidates. It
has no scene-family, object-pose, evaluator, or candidate-index embedding.
This is not an unmodified SB3 DQN or an official Atari reproduction.
"""
from dataclasses import asdict,dataclass
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from apfg.baselines.schema import OBS_DIM,CAND_DIM,MAX_CANDIDATES,SCHEMA_VERSION


@dataclass
class DQNConfig:
    hidden_dim:int=96
    learning_rate:float=0.001
    gamma:float=0.98
    batch_size:int=8
    min_replay:int=2
    replay_capacity:int=1024
    target_update_interval:int=20
    gradient_clip:float=10.
    epsilon_start:float=1.
    epsilon_end:float=.1
    epsilon_decay_transitions:int=100

    def __post_init__(self):
        if not (self.hidden_dim>0 and self.learning_rate>0 and 0<=self.gamma<=1):raise ValueError('Invalid DQN network/optimizer parameters')
        if not (0<self.batch_size and 0<self.min_replay<=self.replay_capacity and self.target_update_interval>0):raise ValueError('Invalid replay/update parameters')
        if not (0<=self.epsilon_end<=self.epsilon_start<=1 and self.epsilon_decay_transitions>0):raise ValueError('Invalid epsilon schedule')


def numeric_state(state):
    """Whitelist tensor fields: extra metadata is never a network input."""
    obs=np.asarray(state['obs'],dtype=np.float32)
    candidates=np.asarray(state['candidates'],dtype=np.float32)
    mask=np.asarray(state['mask'],dtype=bool)
    if obs.shape!=(OBS_DIM,) or candidates.shape!=(MAX_CANDIDATES,CAND_DIM) or mask.shape!=(MAX_CANDIDATES,):
        raise ValueError('State shapes differ from frozen shared schema')
    if not np.isfinite(obs).all() or not np.isfinite(candidates).all():raise ValueError('Non-finite policy input')
    return {'obs':obs.copy(),'candidates':candidates.copy(),'mask':mask.copy()}


def numeric_transition(transition):
    current=numeric_state(transition)
    following=numeric_state({'obs':transition['next_obs'],'candidates':transition['next_candidates'],'mask':transition['next_mask']})
    raw_action=np.asarray(transition['action'])
    if raw_action.shape!=() or raw_action.dtype.kind not in 'iu':raise ValueError('Action must be one integer candidate index')
    action=int(raw_action)
    if not 0<=action<MAX_CANDIDATES or not current['mask'][action]:raise ValueError('Cannot put an illegal selected action into replay')
    reward=float(transition['reward'])
    if not np.isfinite(reward):raise ValueError('Non-finite reward')
    done=np.asarray(transition['done'])
    if done.shape!=() or done.dtype.kind not in 'biu' or int(done) not in (0,1):raise ValueError('Done must be a boolean terminal flag')
    return dict(current,action=action,reward=reward,done=bool(done),
                next_obs=following['obs'],next_candidates=following['candidates'],next_mask=following['mask'])


class ReplayBuffer:
    def __init__(self,capacity=1024):
        if capacity<=0:raise ValueError('Replay capacity must be positive')
        self.capacity=int(capacity);self.position=0;self.storage=[]

    def __len__(self):return len(self.storage)

    def append(self,transition):
        value=numeric_transition(transition)
        if len(self.storage)<self.capacity:self.storage.append(value)
        else:self.storage[self.position]=value
        self.position=(self.position+1)%self.capacity

    def sample(self,count,rng):
        if not 0<count<=len(self):raise ValueError('Invalid replay sample count')
        indices=rng.choice(len(self),size=count,replace=False)
        return {key:np.stack([self.storage[int(i)][key] for i in indices]) for key in self.storage[0]}

    def state_dict(self):
        # Tensors and primitives permit weights_only=True checkpoint loading.
        return {'capacity':self.capacity,'position':self.position,
                'storage':[{k:torch.as_tensor(v).clone() for k,v in row.items()} for row in self.storage]}

    def load_state_dict(self,state):
        self.capacity=int(state['capacity']);self.position=int(state['position'])
        self.storage=[numeric_transition({k:v.cpu().numpy() if isinstance(v,torch.Tensor) else v for k,v in row.items()}) for row in state['storage']]
        if not (len(self.storage)<=self.capacity and 0<=self.position<self.capacity):raise ValueError('Invalid checkpoint replay ring')


class CandidateQ(nn.Module):
    """Shared Q(o,c) weights make candidate reordering permutation-equivariant."""
    def __init__(self,hidden_dim=96):
        super().__init__()
        self.obs_encoder=nn.Sequential(nn.Linear(OBS_DIM,hidden_dim),nn.ReLU())
        self.value=nn.Sequential(nn.Linear(hidden_dim+CAND_DIM,hidden_dim),nn.ReLU(),
                                 nn.Linear(hidden_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,1))

    def forward(self,obs,candidates):
        if obs.ndim!=2 or candidates.ndim!=3 or obs.shape[0]!=candidates.shape[0]:raise ValueError('Expected batched observation/candidate tensors')
        encoded=self.obs_encoder(obs).unsqueeze(1).expand(-1,candidates.shape[1],-1)
        return self.value(torch.cat([encoded,candidates],dim=-1)).squeeze(-1)


def double_dqn_targets(rewards,dones,next_online,next_target,next_mask,gamma):
    """Online legal argmax, target-network evaluation; no invalid bootstrap."""
    available=next_mask.any(dim=1)
    best=next_online.masked_fill(~next_mask,-torch.inf).argmax(dim=1)
    bootstrap=next_target.gather(1,best[:,None]).squeeze(1)
    bootstrap=torch.where(available & ~dones,bootstrap,torch.zeros_like(bootstrap))
    return rewards+float(gamma)*bootstrap


class CandidateDQN:
    def __init__(self,config=None,seed=0,device='cpu'):
        self.config=config if isinstance(config,DQNConfig) else DQNConfig(**(config or {}))
        self.seed=int(seed);self.device=torch.device(device);self.rng=np.random.default_rng(self.seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed);self.online=CandidateQ(self.config.hidden_dim)
        self.online.to(self.device)
        self.target=CandidateQ(self.config.hidden_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict());self.target.eval()
        for parameter in self.target.parameters():parameter.requires_grad_(False)
        self.optimizer=torch.optim.Adam(self.online.parameters(),lr=self.config.learning_rate)
        self.replay=ReplayBuffer(self.config.replay_capacity)
        self.environment_steps=0;self.updates=0;self.target_syncs=0
        self.training_metadata={}

    @property
    def epsilon(self):
        fraction=min(1.,self.environment_steps/self.config.epsilon_decay_transitions)
        return self.config.epsilon_start+fraction*(self.config.epsilon_end-self.config.epsilon_start)

    def values(self,state):
        state=numeric_state(state)
        with torch.no_grad():
            q=self.online(torch.as_tensor(state['obs'][None],device=self.device),
                          torch.as_tensor(state['candidates'][None],device=self.device))[0].cpu().numpy()
        return np.where(state['mask'],q,-np.inf)

    def act(self,state,epsilon=0):
        state=numeric_state(state);legal=np.flatnonzero(state['mask'])
        if len(legal)==0:return None
        if epsilon is None:epsilon=self.epsilon
        if not 0<=epsilon<=1:raise ValueError('Epsilon must be in [0,1]')
        if self.rng.random()<epsilon:return int(self.rng.choice(legal))
        return int(np.argmax(self.values(state)))

    def observe(self,transition):
        self.replay.append(transition);self.environment_steps+=1

    add_transition=observe

    def update(self,batch_size=None):
        if len(self.replay)<self.config.min_replay:return None
        count=min(int(batch_size or self.config.batch_size),len(self.replay))
        data=self.replay.sample(count,self.rng)
        tensors={k:torch.as_tensor(v,device=self.device) for k,v in data.items()}
        q=self.online(tensors['obs'].float(),tensors['candidates'].float()).gather(1,tensors['action'].long()[:,None]).squeeze(1)
        with torch.no_grad():
            target=double_dqn_targets(tensors['reward'].float(),tensors['done'].bool(),
                self.online(tensors['next_obs'].float(),tensors['next_candidates'].float()),
                self.target(tensors['next_obs'].float(),tensors['next_candidates'].float()),
                tensors['next_mask'].bool(),self.config.gamma)
        loss=F.smooth_l1_loss(q,target)
        self.optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=nn.utils.clip_grad_norm_(self.online.parameters(),self.config.gradient_clip)
        self.optimizer.step();self.updates+=1
        synced=self.updates%self.config.target_update_interval==0
        if synced:
            self.target.load_state_dict(self.online.state_dict());self.target_syncs+=1
        return {'update':self.updates,'loss':float(loss.detach().cpu()),'td_abs_mean':float((q.detach()-target).abs().mean().cpu()),
            'q_mean':float(q.detach().mean().cpu()),'target_mean':float(target.mean().cpu()),
            'gradient_norm_before_clip':float(norm),'batch_size':count,'replay_size':len(self.replay),
            'environment_steps':self.environment_steps,'epsilon':self.epsilon,'target_sync':int(synced)}

    def save(self,path,include_replay=True):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        checkpoint={'format':'apfg_candidate_double_dqn_v1','schema_version':SCHEMA_VERSION,
            'implementation':'FROM_SCRATCH candidate-conditioned Double DQN, not SB3/official Atari reproduction',
            'config':asdict(self.config),'seed':self.seed,'online':self.online.state_dict(),'target':self.target.state_dict(),
            'optimizer':self.optimizer.state_dict(),'environment_steps':self.environment_steps,'updates':self.updates,
            'target_syncs':self.target_syncs,'numpy_rng_json':json.dumps(self.rng.bit_generator.state),
            'training_metadata':self.training_metadata,'replay':self.replay.state_dict() if include_replay else None}
        temporary=path.with_name(path.name+'.partial');torch.save(checkpoint,temporary);temporary.replace(path)
        return path

    @classmethod
    def load(cls,path,device='cpu'):
        checkpoint=torch.load(Path(path),map_location='cpu',weights_only=True)
        if checkpoint.get('format')!='apfg_candidate_double_dqn_v1' or checkpoint.get('schema_version')!=SCHEMA_VERSION:
            raise ValueError('Incompatible DQN checkpoint/schema')
        agent=cls(checkpoint['config'],checkpoint['seed'],device)
        agent.online.load_state_dict(checkpoint['online']);agent.target.load_state_dict(checkpoint['target'])
        agent.optimizer.load_state_dict(checkpoint['optimizer'])
        for state in agent.optimizer.state.values():
            for key,value in state.items():
                if isinstance(value,torch.Tensor):state[key]=value.to(agent.device)
        agent.environment_steps=int(checkpoint['environment_steps']);agent.updates=int(checkpoint['updates'])
        agent.target_syncs=int(checkpoint['target_syncs']);agent.rng.bit_generator.state=json.loads(checkpoint['numpy_rng_json'])
        agent.training_metadata=checkpoint.get('training_metadata',{})
        if checkpoint.get('replay') is not None:agent.replay.load_state_dict(checkpoint['replay'])
        return agent
