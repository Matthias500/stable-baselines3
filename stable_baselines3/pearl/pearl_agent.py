#Source: https://github.com/katerakelly/oyster

import numpy as np

import torch
from torch import nn as nn
import torch.nn.functional as F

import stable_baselines3.pearl.pytorch_util_pearl as ptu
from stable_baselines3.common.policies import BaseModel
from stable_baselines3.common.utils import get_device
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import gym
from gym import spaces
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    create_mlp,
)
from abc import ABC, abstractmethod

def _product_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared


def _mean_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of mean of gaussians
    '''
    mu = torch.mean(mus, dim=0)
    sigma_squared = torch.mean(sigmas_squared, dim=0)
    return mu, sigma_squared


def _natural_to_canonical(n1, n2):
    ''' convert from natural to canonical gaussian parameters '''
    mu = -0.5 * n1 / n2
    sigma_squared = -0.5 * 1 / n2
    return mu, sigma_squared


def _canonical_to_natural(mu, sigma_squared):
    ''' convert from canonical to natural gaussian parameters '''
    n1 = mu / sigma_squared
    n2 = -0.5 * 1 / sigma_squared
    return n1, n2

class PEARLEncoder(nn.Module):
    def __init__(self, input_dim: int, feature_dim: int, net_arch: List[int] = [200,200,200], activation_fn: Type[nn.Module] = nn.ReLU, device: Union[torch.device, str] = "auto"):
        super(PEARLEncoder, self).__init__()

        self.device = get_device(device)
        encoder = []
        #encoder.append(nn.Flatten())
        last_layer_dim_shared = input_dim
        for layer in net_arch:
            encoder.append(nn.Linear(last_layer_dim_shared, layer))  # add linear of size layer
            encoder.append(activation_fn())
            last_layer_dim_shared = layer
        encoder.append(nn.Linear(last_layer_dim_shared, feature_dim))
        self.input_dim = input_dim
        self.output_size = feature_dim
        self.encoder = nn.Sequential(*encoder).to(self.device)

    def forward(self, features: torch.Tensor):
        out = self.encoder(features[0])
        for i in range(1,features.size(0)):
            out = torch.cat((out, self.encoder(features[i])),0)
        return self.encoder(features)


class PEARLAgent(BaseModel):
    def __init__(self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        features_extractor: Optional[nn.Module] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        encoder_net_arch: List[int] = [200, 200, 200],
        activation_fn: Type[nn.Module] = nn.ReLU,
        latent_dim: int = 2,
        encoder_lr: float = 0.0003,
    ):
        super(PEARLAgent, self).__init__(
            observation_space=observation_space,
            action_space=action_space,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=None,
            features_extractor=None,
            normalize_images=True,
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs=None,
        )
        self.encoder_net_arch = encoder_net_arch
        self.activation_function = activation_fn
        self.latent_dim = latent_dim
        self.input_dim = 2 * observation_space.shape[0] + action_space.shape[0] + 1
        self.output_dim = 2 * self.latent_dim
        self.encoder_lr = encoder_lr

        self._build()

        self.use_ib = True

        # initialize buffers for z dist and z
        # use buffers so latent context can be saved along with model weights
        self.register_buffer('z', torch.zeros(1, latent_dim))
        self.register_buffer('z_means', torch.zeros(1, latent_dim))
        self.register_buffer('z_vars', torch.zeros(1, latent_dim))
        self.clear_z()

    def _build(self):
        self.context_encoder = PEARLEncoder(
            input_dim=self.input_dim,
            feature_dim=self.output_dim,
            activation_fn=self.activation_function,
            net_arch=self.encoder_net_arch,
        )
        self.context_optimizer = self.optimizer_class(self.context_encoder.encoder.parameters(), self.encoder_lr)
    def clear_z(self, num_tasks=1):
        '''
        reset q(z|c) to the prior
        sample a new z from the prior
        '''
        # reset distribution over z to the prior
        mu = torch.zeros(num_tasks, self.latent_dim).to(self.context_encoder.device)
        if self.use_ib:
            var = torch.ones(num_tasks, self.latent_dim).to(self.context_encoder.device)
        else:
            var = torch.zeros(num_tasks, self.latent_dim).to(self.context_encoder.device)
        self.z_means = mu
        self.z_vars = var
        # sample a new z from the prior
        self.sample_z()
        # reset the context collected so far
        self.context = None

    def detach_z(self):
        ''' disable backprop through z '''
        self.z = self.z.detach()


    def update_context(self, inputs):
        ''' append single transition to the current context '''
        o, a, r, no, d, info = inputs
        o = ptu.from_numpy(o[None, None, ...])
        a = ptu.from_numpy(a[None, None, ...])
        r = ptu.from_numpy(np.array([r])[None, None, ...])
        no = ptu.from_numpy(no[None, None, ...])

        data = torch.cat([o, a, r, no], dim=2)

        if self.context is None:
            self.context = data
        else:
            self.context = torch.cat([self.context, data], dim=1)

    def compute_kl_div(self):
        ''' compute KL( q(z|c) || r(z) ) '''
        prior = torch.distributions.Normal(torch.zeros(self.latent_dim).to(self.context_encoder.device), torch.ones(self.latent_dim).to(self.context_encoder.device))
        posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
        kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
        kl_div_sum = torch.sum(torch.stack(kl_divs))
        return kl_div_sum

    def infer_posterior(self, context):
        ''' compute q(z|c) as a function of input context and sample new z from it'''
        params = self.context_encoder(context)
        #print("Encoder Output:", params.size())
        params = params.view(context.size(0), -1, self.context_encoder.output_size)
        #print("Reordering:", params.size())
        # with probabilistic z, predict mean and variance of q(z | c)
        if self.use_ib:
            mu = params[..., :self.latent_dim]
            #print("Mu:", mu.size())
            sigma_squared = F.softplus(params[..., self.latent_dim:])
            #print("var:", sigma_squared.size())
            sigma_squared = torch.clamp(sigma_squared, min=1e-7)
            added = sigma_squared[0] + sigma_squared[1]
            mu1sigma2 = torch.mul(mu[0], sigma_squared[1])
            mu2sgima1 = torch.mul(mu[1], sigma_squared[0])
            self.z_means = torch.mul((mu1sigma2 + mu2sgima1), torch.reciprocal(added))
            self.z_vars = 1./ (torch.reciprocal(sigma_squared[0]) + torch.reciprocal(sigma_squared[1]))
            for i in range(2,context.size(0)):
                self.z_vars = torch.clamp(self.z_vars, min=1e-7)
                added = self.z_vars + sigma_squared[i]
                mu1sigma2 = torch.mul(self.z_means, sigma_squared[i])
                mu2sgima1 = torch.mul(mu[i], self.z_vars)
                self.z_means = torch.mul((mu1sigma2 + mu2sgima1), torch.reciprocal(added))
                self.z_vars = 1. / (torch.reciprocal(self.z_vars) + torch.reciprocal(sigma_squared[i]))
            #print(self.z_vars)
            #print(self.z_means)
            #print(len([torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]))
            #z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
            #print(zip(torch.unbind(mu), torch.unbind(sigma_squared)))
            #print("Z_Params:",z_params[0])
            #print("length:", len(z_params))
            #self.z_means = torch.stack([p[0] for p in z_params])
            #self.z_vars = torch.stack([p[1] for p in z_params])
            #print("z_means:", self.z_means.size())
            #print("z_vars:",self.z_vars.size())
        # sum rather than product of gaussians structure
        else:
            self.z_means = torch.mean(params, dim=1)
        self.sample_z()
        #print("Sampled STuff", self.z[0])
        #print("z_means:", self.z_means)
        #print("z_vars:",self.z_vars)
        #self.clear_z()
        #print(self.compute_kl_div())

    def sample_z(self):
        if self.use_ib:
            posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
            z = [d.rsample() for d in posteriors]
            self.z = torch.stack(z)
        else:
            self.z = self.z_means


    def forward(self, context):
        ''' given context, get statistics under the current policy of a set of observations '''
        self.infer_posterior(context)
        self.sample_z()

        task_z = self.z

        return task_z

    def log_diagnostics(self, eval_statistics):
        '''
        adds logging data about encodings to eval_statistics
        '''
        z_mean = np.mean(np.abs(ptu.get_numpy(self.z_means[0])))
        z_sig = np.mean(ptu.get_numpy(self.z_vars[0]))
        eval_statistics['Z mean eval'] = z_mean
        eval_statistics['Z variance eval'] = z_sig

    @property
    def networks(self):
        return [self.context_encoder]


class PEARLAgent_old(nn.Module):

    def __init__(self,
         latent_dim,
         context_encoder,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.context_encoder = context_encoder

        self.use_ib = True

        # initialize buffers for z dist and z
        # use buffers so latent context can be saved along with model weights
        self.register_buffer('z', torch.zeros(1, latent_dim))
        self.register_buffer('z_means', torch.zeros(1, latent_dim))
        self.register_buffer('z_vars', torch.zeros(1, latent_dim))

        self.clear_z()

    def clear_z(self, num_tasks=1):
        '''
        reset q(z|c) to the prior
        sample a new z from the prior
        '''
        # reset distribution over z to the prior
        mu = ptu.zeros(num_tasks, self.latent_dim)
        if self.use_ib:
            var = ptu.ones(num_tasks, self.latent_dim)
        else:
            var = ptu.zeros(num_tasks, self.latent_dim)
        self.z_means = mu
        self.z_vars = var
        # sample a new z from the prior
        self.sample_z()
        # reset the context collected so far
        self.context = None

    def detach_z(self):
        ''' disable backprop through z '''
        self.z = self.z.detach()


    def update_context(self, inputs):
        ''' append single transition to the current context '''
        o, a, r, no, d, info = inputs
        o = ptu.from_numpy(o[None, None, ...])
        a = ptu.from_numpy(a[None, None, ...])
        r = ptu.from_numpy(np.array([r])[None, None, ...])
        no = ptu.from_numpy(no[None, None, ...])

        data = torch.cat([o, a, r, no], dim=2)

        if self.context is None:
            self.context = data
        else:
            self.context = torch.cat([self.context, data], dim=1)

    def compute_kl_div(self):
        ''' compute KL( q(z|c) || r(z) ) '''
        prior = torch.distributions.Normal(ptu.zeros(self.latent_dim), ptu.ones(self.latent_dim))
        posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
        kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
        kl_div_sum = torch.sum(torch.stack(kl_divs))
        return kl_div_sum

    def infer_posterior(self, context):
        ''' compute q(z|c) as a function of input context and sample new z from it'''
        params = self.context_encoder(context)
        params = params.view(context.size(0), -1, self.context_encoder.output_size)
        # with probabilistic z, predict mean and variance of q(z | c)
        if self.use_ib:
            mu = params[..., :self.latent_dim]
            sigma_squared = F.softplus(params[..., self.latent_dim:])
            z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
            self.z_means = torch.stack([p[0] for p in z_params])
            self.z_vars = torch.stack([p[1] for p in z_params])
        # sum rather than product of gaussians structure
        else:
            self.z_means = torch.mean(params, dim=1)
        self.sample_z()

    def sample_z(self):
        if self.use_ib:
            posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
            z = [d.rsample() for d in posteriors]
            self.z = torch.stack(z)
        else:
            self.z = self.z_means


    def forward(self, obs, context):
        ''' given context, get statistics under the current policy of a set of observations '''
        self.infer_posterior(context)
        self.sample_z()

        task_z = self.z

        return task_z

    def log_diagnostics(self, eval_statistics):
        '''
        adds logging data about encodings to eval_statistics
        '''
        z_mean = np.mean(np.abs(ptu.get_numpy(self.z_means[0])))
        z_sig = np.mean(ptu.get_numpy(self.z_vars[0]))
        eval_statistics['Z mean eval'] = z_mean
        eval_statistics['Z variance eval'] = z_sig

    @property
    def networks(self):
        return [self.context_encoder]
