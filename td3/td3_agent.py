"""
This code mainly follows a td3 YouTube tutorial found at:
https://www.youtube.com/watch?v=ZhFO8EWADmY&t=1895s
Channel name: Machine Learning with Phil

Any modifiations are made by the AABL Lab.
"""

import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from buffer import ReplayBuffer
from td3_networks import CriticNetwork, ActorNetwork

class Agent():
    def __init__(self, alpha, beta, input_dims, tau, env,
            gamma=0.99, update_actor_interval=2, warmup=1000,
            n_actions=2, max_size=1000000, layer1_size=400,
            layer2_size=300, batch_size=100, noise=0.1):

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low
        self.input_dims = input_dims
        self.tau = tau
        self.env = env

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size =batch_size

        self.learn_step_count = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_interval = update_actor_interval

        self.actor = ActorNetwork(alpha, input_dims, layer1_size, 
                layer2_size, n_actions=n_actions, name="actor")      
        self.critic_1 = CriticNetwork(beta, input_dims, layer1_size,
                layer2_size, n_actions=n_actions, name="critic_1")
        self.critic_2 = CriticNetwork(beta, input_dims, layer1_size,
                layer2_size, n_actions=n_actions, name="critic_2")

        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size,
                layer2_size, n_actions=n_actions, name="target_actor")
        self.target_critic_1 = CriticNetwork(beta, input_dims, layer1_size,
                layer2_size, n_actions=n_actions, name="target_critic_1")
        self.target_critic_2 = CriticNetwork(beta, input_dims, layer1_size,
                layer2_size, n_actions=n_actions, name="target_critic_2")

        self.noise = noise
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        if self.time_step < self.warmup:
            mu = torch.tensor(np.random.normal(scale=self.noise,
                                                size=(self.n_actions)))
        else:
            state = torch.tensor(observation, dtype=torch.float).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)
        # add noise:
        mu_prime = mu + torch.tensor(np.random.normal(scale=self.noise),dtype=torch.float).to(self.actor.device)
        
        # clamp to ensure action is in env action boundaries:
        mu_prime = torch.clamp(mu_prime, self.min_action[0], self.max_action[0])
        self.time_step += 1

        return mu_prime.cpu().detach().numpy()

    # Custom get_dist function for interactive RL
    def get_dist(self, observation, with_noise=True, skip_warmup=True, as_numpy=True, with_grad=True):
        if with_grad:
            if not skip_warmup:
                if self.time_step < self.warmup:
                    mu = torch.tensor(np.random.normal(scale=self.noise,
                                                        size=(self.n_actions)))
                else:
                    state = torch.tensor(observation, dtype=torch.float).to(self.actor.device)
                    mu = self.actor.forward(state).to(self.actor.device)
            else:
                state = torch.tensor(observation, dtype=torch.float).to(self.actor.device)
                mu = self.actor.forward(state).to(self.actor.device)

            if with_noise:
                mu_prime = mu + torch.tensor(np.random.normal(scale=self.noise),dtype=torch.float).to(self.actor.device)
            else:
                mu_prime = mu
            
            self.time_step += 1

            if as_numpy:
                return mu_prime.cpu().detach().numpy()
            else:
                # return as torch tensor
                return mu_prime.cpu().detach()
        else:
            with torch.no_grad():
                if not skip_warmup:
                    if self.time_step < self.warmup:
                        mu = torch.tensor(np.random.normal(scale=self.noise,
                                                            size=(self.n_actions)))
                    else:
                        state = torch.tensor(observation, dtype=torch.float).to(self.actor.device)
                        mu = self.actor.forward(state).to(self.actor.device)
                else:
                    state = torch.tensor(observation, dtype=torch.float).to(self.actor.device)
                    mu = self.actor.forward(state).to(self.actor.device)

                if with_noise:
                    mu_prime = mu + torch.tensor(np.random.normal(scale=self.noise),dtype=torch.float).to(self.actor.device)
                else:
                    mu_prime = mu
                
                self.time_step += 1

                if as_numpy:
                    return mu_prime.cpu().detach().numpy()
                else:
                    # return as torch tensor
                    return mu_prime.cpu().detach()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        # note the divice should be the same for any network
        reward = torch.tensor(reward, dtype=torch.float).to(self.critic_1.device)
        done = torch.tensor(done).to(self.critic_1.device)
        state_ = torch.tensor(new_state, dtype=torch.float).to(self.critic_1.device)
        state = torch.tensor(state, dtype=torch.float).to(self.critic_1.device)
        action = torch.tensor(action, dtype=torch.float).to(self.critic_1.device)

        # learning:
        target_actions = self.target_actor.forward(state_)

        # add noise/smoothing
        target_actions = torch.clamp(target_actions, self.min_action[0], self.max_action[0])

        q1_ = self.target_critic_1.forward(state_, target_actions)
        q2_ = self.target_critic_2.forward(state_, target_actions)

        q1 = self.critic_1.forward(state, action)
        q2 = self.critic_2.forward(state, action)

        # mask with done values (0, 1). If done is true, q1 and q1 should be zero
        q1_[done] = 0.0
        q2_[done] = 0.0

        # for dimensionality 
        q1_ = q1_.view(-1)
        q2_ = q2_.view(-1)

        critic_value_ = torch.min(q1_, q2_)
        
        target = reward + self.gamma*critic_value_
        target = target.view(self.batch_size, 1)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)

        critic_loss = q1_loss + q2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        if self.learn_step_count % self.update_actor_interval != 0:
            return

        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state))
        actor_loss = -torch.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        # parameters need to initially be the same for the various networks
        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        critic_1 = dict(critic_1_params)
        critic_2 = dict(critic_2_params)
        actor = dict(actor_params)
        target_actor = dict(target_actor_params)
        target_critic_1 = dict(target_critic_1_params)
        target_critic_2 = dict(target_critic_2_params)

        for name in critic_1:
            critic_1[name] = tau*critic_1[name].clone() + (1-tau)*target_critic_1[name].clone()

        for name in critic_2:
            critic_2[name] = tau*critic_2[name].clone() + (1-tau)*target_critic_2[name].clone()

        # soft-update rule
        for name in actor:
            actor[name] = tau*actor[name].clone() + (1-tau)*target_actor[name].clone()

        self.target_critic_1.load_state_dict(critic_1)
        self.target_critic_2.load_state_dict(critic_2)
        self.target_actor.load_state_dict(actor)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()