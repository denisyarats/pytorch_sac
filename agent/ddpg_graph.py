import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#import apex

import utils
from nets import DoubleQCritic, DeterministicActor2


class DDPGAgent:
    def __init__(self, name, obs_dim, action_dim, device, lr, nstep,
                 batch_size, critic_target_tau, num_expl_steps, lerp,
                 critic_use_ln, critic_hidden_dims, critic_spectral_norms, actor_use_ln, actor_hidden_dims,
                 actor_spectral_norms, update_every_steps, stddev_schedule,
                 stddev_clip, use_tb):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.lerp = lerp

        # models
        self.actor = DeterministicActor2(obs_dim, action_dim, actor_use_ln, actor_hidden_dims,
                                        actor_spectral_norms).to(device)

        self.critic = DoubleQCritic(obs_dim, action_dim, critic_use_ln, critic_hidden_dims,
                                    critic_spectral_norms).to(device)
        self.critic_target = DoubleQCritic(obs_dim, action_dim,
                                           critic_use_ln, critic_hidden_dims,
                                           critic_spectral_norms).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        opt_ty = apex.optimizers.FusedAdam if False else torch.optim.Adam
        self.actor_opt = opt_ty(self.actor.parameters(), lr=lr)
        self.critic_opt = opt_ty(self.critic.parameters(), lr=lr)
        
        self.obs = None
        self.action = None
        self.reward = None
        self.discount = None
        self.next_obs = None
        self.graph = None

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        #stddev = utils.schedule(self.stddev_schedule, step)
        mu = self.actor(obs)
        if eval_mode:
            action = mu
        else:
            action = mu + torch.randn_like(mu) * 0.2 #dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def create_update_graph(self, obs, action, reward, discount, next_obs):
        self.obs = obs
        self.action = action
        self.reward = reward
        self.discount = discount
        self.next_obs = next_obs
        
        critic_state = self.critic.state_dict()
        actor_state = self.actor.state_dict()
        critic_opt_state = self.critic_opt.state_dict()
        actor_opt_state = self.actor_opt.state_dict()
        
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self.update_critic_static()
                self.update_actor_static()
                
        torch.cuda.current_stream().wait_stream(s)
        self.critic.load_state_dict(critic_state)
        self.actor.load_state_dict(actor_state)
        self.critic_opt.load_state_dict(critic_opt_state)
        self.actor_opt.load_state_dict(actor_opt_state)
        
        self.graph = torch.cuda.CUDAGraph()
        self.critic_opt.zero_grad(set_to_none=True)
        self.actor.zero_grad(set_to_none=True)
        with torch.cuda.graph(self.graph):
            critic_loss = self.update_critic_static()
            actor_loss = self.update_actor_static()
            
        return critic_loss, actor_loss
                
    
    def update_critic_static(self):

        with torch.no_grad():
            #stddev = utils.schedule(self.stddev_schedule, step)
            mean = self.actor(self.next_obs)
            next_action = mean + torch.randn_like(mean) * 0.2
            #next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(self.next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = self.reward + (self.discount * target_V)

        Q1, Q2 = self.critic(self.obs, self.action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)


        # optimize critic
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        return critic_loss
    
    def update_impl(self, obs, action, reward, discount, next_obs):
        if self.graph is None:
            self.create_update_graph(obs, action, reward, discount, next_obs)
        self.obs.copy_(obs)
        self.action.copy_(action)
        self.reward.copy_(reward)
        self.discount.copy_(discount)
        self.next_obs.copy_(next_obs)
        
        x = self.graph.replay()
        
        metrics = dict()
        if self.use_tb:
            metrics['critic_loss'] = critic_loss.item()
            metrics['actor_loss'] = actor_loss.item()
            
        return metrics
    
    def update_actor_static(self):
        action = self.actor(self.obs)
        #action = dist.sample(clip=self.stddev_clip)
        #log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(self.obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()


        return actor_loss
        

    def update_actor(self, obs, step):
        metrics = dict()
        
        if self.actor_g is None:
            self.create_update_actor_graph(obs)
            
        self.obs.copy_(obs)
        
        actor_loss = self.actor_g.replay()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            #metrics['actor_logprob'] = log_prob.mean().item()
            #metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        #metrics.update(
        #    self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        #metrics.update(self.update_actor(obs, step))
        metrics.update(self.update_impl(obs, action, reward, discount, next_obs))

        # update critic target
        with torch.no_grad():
            if self.lerp:
                utils.soft_update_params(self.critic, self.critic_target,
                                         self.critic_target_tau)
            else:
                utils.soft_update_params_old(self.critic, self.critic_target,
                                         self.critic_target_tau)
                

        return metrics
