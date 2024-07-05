import math
import numpy as np
import torch as th
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import BasePolicy, BaseModel
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    create_mlp,
    get_actor_critic_arch,
)
from gymnasium import spaces
from typing import Any, Dict, Generator, List, Optional, Tuple, Union, Type

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2, dtype=th.float32):
    betas = np.linspace(
        beta_start, beta_end, timesteps
    )
    return th.tensor(betas, dtype=dtype)

def extract(a, t, x_shape): # alpha, t, x
    b, *_ = t.shape # get batch number
    # out = a.gather(-1, t) # chose value based on the given indexes: t, i.e., out = alpha(t)
    out = a[t]# chose value based on the given indexes: t, i.e., out = alpha(t)
    output = out.reshape(x_shape[0], x_shape[1], 1) # unsqueeze and place with 1, and make the shape has the same dimension as x
    return output # reshape alpha(t) into the shame shape: batch, x_shape

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t): # t: 256, 10 or 1, 1
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = th.exp(th.arange(half_dim, device=device) * -emb)
        emb = t[:, :, None] * emb[None, :] # this will create a new axis, 1, 1, 8
        emb = th.cat((emb.sin(), emb.cos()), dim=-1) # 1, 1, 16 or 256, 10, 16
        return emb

class Networks(nn.Module): # This network is for generating x_t-1 or epsilon_theta

    def __init__(self,
                 action_dim,
                 state_feat_dim,
                 time_dim=16):

        super(Networks, self).__init__()

        self.action_dim = action_dim
        self.state_feat_dim = state_feat_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Tanh(),
            nn.Linear(time_dim * 2, time_dim),
        )

        inputime_dim = self.state_feat_dim + self.action_dim + time_dim

        self.mid_layer = nn.Sequential(nn.Linear(inputime_dim, 64),
                                       nn.Tanh(),
                                       nn.Linear(64, 64),
                                       nn.Tanh())

        self.final_layer = nn.Linear(64, self.action_dim)

    def initialization(self):
        nn.init.xavier_uniform_(self.time_mlp, gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.mid_layer, gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.final_layer, gain=nn.init.calculate_gain('tanh'))

    def forward(self, action, time, state):
        '''
        shape:
            action: n_env, n_action, action_dim
            state: n_env, 1, state_feat_dim
            time: n_env, n_action, time_dim
        '''
        t_embed = self.time_mlp(time)

        x = th.cat([action, t_embed, state], dim=-1).float() # action = 6, 100, 1
        action = self.mid_layer(x)

        return self.final_layer(action)

class Diffusion_Policy(nn.Module): # forward method here is generate a sample

    def __init__(self, 
                 state_feat_dim, 
                 action_dim, 
                 model,
                 n_timesteps=10):
        super(Diffusion_Policy, self).__init__()

        self.state_feat_dim = state_feat_dim
        self.action_dim = action_dim
        self.model = model(action_dim=self.action_dim, state_feat_dim=self.state_feat_dim)

        betas = linear_beta_schedule(n_timesteps) # beta

        alphas = 1. - betas # alpha_t
        alphas_cumprod = th.cumprod(alphas, axis=0) # alpha_bar_t

        # alphas_prev here
        alphas_cumprod_prev = th.cat([th.ones(1), alphas_cumprod[:-1]]) # alpha_bar_t-1

        self.n_timesteps = int(n_timesteps)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', th.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', th.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', th.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', th.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', th.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             th.log(th.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)) # 
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        self.gaussian_distribution = Normal(0.0, 1.0)

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, time, noise): # get x_0 prediction from x_t and noise

        # x_0 = x_t/th.sqrt(1. / alphas_cumprod)) - th.sqrt(1. / alphas_cumprod - 1)), reverse formula
        return (
                extract(self.sqrt_recip_alphas_cumprod, time, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, time, x_t.shape) * noise 
        ) 

    def q_posterior(self, x_start, x_t, time):
        
        posterior_mean = (
                extract(self.posterior_mean_coef1, time, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, time, x_t.shape) * x_t
        ) # or actually we could use the noise, which is 
        
        # corresponding to function
        # posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)) 
        # posterior_mean_coef2 = (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
                       
        posterior_variance = extract(self.posterior_variance, time, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, time, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, time, state_feat):
        # (batch_size, self.n_actions, self.action_dim)
        # generate prediction of x_0, noise = eta_theta here
        x_recon = self.predict_start_from_noise(x, time=time, noise=self.model(x, time, state_feat)) 
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, time=time) # function(1), x_t-1=f(x_t, noise_theta)

        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, x, time, state_feat): # action, time_step, state
        b, *_, device = *x.shape, x.device # batch, _, device

        # model_mean is the mean of x_t-1
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, time=time, state_feat=state_feat)

        # here is random noise to be added as
        noise = th.randn_like(x)
        # no noise when t == 0
        # nonzero_mask = (1 - (time == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        nonzero_mask = (1 - (time == 0).float()).reshape(x.shape[0], x.shape[1], 1)
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise # x_t-1

    def p_sample_loop(self, state_feat, shape, n_actions):
        device = self.betas.device
        batch_size = shape[0] 

        # (batch_size, self.n_actions, self.action_dim)
        x = th.randn(shape, device=device) # gaussian random with action shape

        for i in reversed(range(0, self.n_timesteps)):
            time = th.full((batch_size, n_actions), i, device=device, dtype=th.long) # fill batch size with i
            x = self.p_sample(x, time, state_feat)

        return x

    def sample(self, state_feat, n_actions):
        batch_size = state_feat.shape[0] # get batch size
        shape = (batch_size, n_actions, self.action_dim) # make output shape same as action shape
        action = self.p_sample_loop(state_feat, shape, n_actions) # get x from all diffusion steps
        return action

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = th.randn_like(x_start) # batch, n_actions, action_dim

        a = 1
        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, state_feat, t, advantages, n_actions): # TODO, expand the noise to have multiple choice
        noise = th.randn_like(x_start) # batch, n_actions, action_dim

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise) # this process should also involved with n_actions
        eta_theta = self.model(x_noisy, t, state_feat) # eta_theta

        assert noise.shape == eta_theta.shape
        loss = th.mean(advantages * (eta_theta - noise) ** 2)
        loss = F.mse_loss(eta_theta, noise)

        return loss

    def loss(self, x, state_feat, advantages, n_actions): 
        x = x.unsqueeze(1).expand(-1, n_actions, -1)
        state_feat = state_feat.unsqueeze(1).expand(-1, n_actions, -1)
        batch_size = len(x) # ground truth, .unsqueeze(1).expand(-1, n_actions, -1)
        t = th.randint(0, self.n_timesteps, (batch_size, n_actions), device=x.device).long()
        return self.p_losses(x, state_feat, t, advantages, n_actions)

    def forward(self, state_feat, n_actions=1): 
        state_feat = state_feat.unsqueeze(1).expand(-1, n_actions, -1)
        actions = self.sample(state_feat, n_actions)
        if n_actions == 1:
            return actions.squeeze() # here is the action
        else:
            return actions
    
class ContinuousCritic(BaseModel):

    features_extractor: BaseFeaturesExtractor

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        action_dim = get_action_dim(self.action_space)

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks: List[nn.Module] = []
        for idx in range(n_critics):
            q_net_list = create_mlp(features_dim + action_dim, 1, net_arch, activation_fn)
            q_net = nn.Sequential(*q_net_list)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs, actions):
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        qvalue_input = th.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_multi_forward(self, obs, actions, n_actions):
        with th.no_grad():
            features = self.extract_features(obs, self.features_extractor)
        features = features.unsqueeze(1).expand(-1, n_actions, -1)
        # here are the Q(s',a') for all a' generated by diffusion policy network
        q_values = self.q_networks[0](th.cat([features, actions], dim=-1)) # batch, n_actions, 1
        values = th.mean(q_values,dim=1).unsqueeze(1) # batch, n_actions, 1
        advantages = q_values - values
        return advantages
    
    def q1_multi_forward_advantages(self, obs, actions, n_actions):
        features = self.extract_features(obs, self.features_extractor)
        features = features.unsqueeze(1).expand(-1, n_actions, -1)
        # here are the Q(s',a') for all a' generated by diffusion policy network
        q_values = self.q_networks[0](th.cat([features, actions], dim=-1)) # batch, n_actions, 1
        values = th.mean(q_values, dim=1).unsqueeze(1) # batch, n_actions, 1
        advantages = q_values - values
        return advantages