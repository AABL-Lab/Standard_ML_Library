import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from Standard_ML_Library.SAC.discrete.discrete_buffer import DiscreteReplayBuffer
from Standard_ML_Library.SAC.discrete.discrete_actor_network import DiscreteActorNetwork
from Standard_ML_Library.SAC.discrete.discrete_critic_network import DiscreteCriticNetwork
from Standard_ML_Library.SAC.discrete.discrete_skill_network import DiscreteSkillNetwork
from Standard_ML_Library.SAC.discrete.networks import ValueNetwork

from IPython import embed    

# from pytorchvis import draw_graph
from torchviz import make_dot

# torch.autograd.set_detect_anomaly(True)

class DiscreteDIAYNAgent():
    def __init__(self, alpha=0.0007, beta=0.0007, obs_dims=[8],
            env=None, gamma=0.99, n_actions=2, n_skills=5, max_size=1000000, tau=0.005,
            layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2, auto_entropy=False, 
            entr_lr=None, zero_entropy=False, reparam_noise=1e-6, verbose=False):
        
        input_dims = (obs_dims[0] + n_skills, *obs_dims[1:])
        print("input dimensions", input_dims)
        self.verbose = False
        self.gamma = gamma
        self.tau = tau
        self.memory = DiscreteReplayBuffer(max_size, obs_dims)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.n_skills = n_skills
        self.env = env
        self.auto_entropy = auto_entropy
        self.actor = DiscreteActorNetwork(alpha, input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size, n_actions=n_actions,
                    name='actor')
        self.critic_1 = DiscreteCriticNetwork(beta, input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size, n_actions=n_actions,
                    name='critic_1')
        self.critic_2 = DiscreteCriticNetwork(beta, input_dims,fc1_dims=layer1_size, fc2_dims=layer2_size, n_actions=n_actions,
                    name='critic_2')
        
        self.future_value_1 = DiscreteCriticNetwork(beta, input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size, n_actions=n_actions,
                    name='future_value_1')
        self.future_value_2 = DiscreteCriticNetwork(beta, input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size, n_actions=n_actions,
                    name='future_value_2')

        self.value = ValueNetwork(beta, input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size, name='value')
        self.target_value = ValueNetwork(beta, input_dims,fc1_dims=layer1_size, fc2_dims=layer2_size, name='target_value')

        self.skill = DiscreteSkillNetwork(alpha, obs_dims, fc1_dims=layer1_size, fc2_dims=layer2_size, n_skills=n_skills,
                    name='skill')
        self.skill_loss_fnc = nn.CrossEntropyLoss()

        self.scale = reward_scale
        self.update_network_parameters(tau=1)
        self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32) if env else 0.
        self.log_alpha = torch.zeros(1, requires_grad=True, device="cpu")
        if entr_lr is None:
            self.entr_lr = alpha
        else:
            self.entr_lr = entr_lr
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.entr_lr)
        self.entropy = 1. if not zero_entropy else 0.

        self.created_dot_graphs = True

    def choose_action(self, observation):
        actions, _ = self.actor.sample_action(observation)
        return actions

    def remember(self, state, action, reward, new_state, done, skill=None):
        self.memory.store_transition(state, action, reward, new_state, done, skill)

    def copy_model_over(self, model_source, model_target, tau=None):
        # perform a soft update of one model from another model
        if tau is None:
            tau = self.tau

        target_params = model_target.named_parameters()
        params = model_source.named_parameters()

        target_state_dict = dict(target_params)
        state_dict = dict(params)

        for name in state_dict:
            state_dict[name] = tau*state_dict[name].clone() + \
                    (1-tau)*target_state_dict[name].clone()

        model_target.load_state_dict(state_dict)

    def update_network_parameters(self, tau=None):
        # TODO: Add skill network equivalent 
        self.copy_model_over(self.value, self.target_value, tau)
        self.copy_model_over(self.critic_1, self.future_value_1, tau)
        self.copy_model_over(self.critic_2, self.future_value_2, tau)
    
    def save_models(self, suffix=''):
        self.actor.save_checkpoint(suffix)
        self.value.save_checkpoint(suffix)
        self.target_value.save_checkpoint(suffix)
        self.future_value_1.save_checkpoint(suffix)
        self.future_value_2.save_checkpoint(suffix)
        self.critic_1.save_checkpoint(suffix)
        self.critic_2.save_checkpoint(suffix)
        self.skill.save_checkpoint(suffix)

    def load_models(self, suffix=''):
        print('.... loading models ....')
        self.actor.load_checkpoint(suffix)
        self.value.load_checkpoint(suffix)
        self.target_value.load_checkpoint(suffix)
        self.future_value_1.load_checkpoint(suffix)
        self.future_value_2.load_checkpoint(suffix)
        self.critic_1.load_checkpoint(suffix)
        self.critic_2.load_checkpoint(suffix)
        self.skill.load_checkpoint(suffix)

    def learn(self, update_params=True):
        # if self.memory.mem_cntr < self.batch_size:
        #     return -1, -1, -1, -1

        state, action, _, new_state, done, skill = \
                self.memory.sample_buffer(self.batch_size)

        done = torch.tensor(done).to(self.actor.device)
        obs_ = torch.tensor(new_state, dtype=torch.float).to(self.actor.device) # next observation
        obs = torch.tensor(state, dtype=torch.float).to(self.actor.device)
        action = torch.tensor(action, dtype=torch.float).to(self.actor.device)
        skill = torch.tensor(skill, dtype=torch.int).to(self.actor.device)

        state = torch.cat((torch.tensor(obs), skill), dim=1)
        state_ = torch.cat((torch.tensor(obs_), skill), dim=1) # next state

        # embed()
        value = self.value(state).flatten()
        value_ = self.target_value(state_).flatten()
        value_[done] = 0.0 # by convention the value of the next state after the final state is 0

        q1_new_policy = self.critic_1.forward(state)
        q2_new_policy = self.critic_2.forward(state)

        action_probs, log_probs, action_value = self.actor.forward(state)
        # with torch.no_grad():
        critic_value = torch.min(q1_new_policy, q2_new_policy) 

        self.value.optimizer.zero_grad()
        value_target = critic_value - self.entropy*log_probs
        # JSS point (iii) from the discrete sac paper talks about directly computing the expectation. 
        #   It's the likelihood of taking an action in a state multiplied by the value of that s-a pair. Summed over all actions for each state (i think)
        value_expectation = torch.sum(action_probs * value_target, dim=1)
        value_loss = 0.5 * F.mse_loss(value, value_expectation)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        # Skill discriminator loss and intrinsic reward
        self.skill.optimizer.zero_grad()
        z_hat = torch.argmax(skill, dim=1)
        skill_pred = self.skill(obs_)[0]
        skill_pred_logsftmx = F.log_softmax(skill_pred, 1)
        _, pred_z = torch.max(skill_pred_logsftmx, dim=1, keepdim=True)
        rewards = skill_pred_logsftmx[torch.arange(skill_pred.shape[0]), z_hat] - math.log(1/self.n_skills)
        rewards = rewards.reshape(-1, 1)
        skill_loss = self.skill_loss_fnc(skill_pred, z_hat)
        skill_loss.backward()
        self.skill.optimizer.step()

        self.actor.optimizer.zero_grad()
        actor_loss = self.entropy*log_probs - critic_value
        actor_loss = torch.mean(torch.sum(action_probs * actor_loss, dim=1))
        # actor_loss = F.mse_loss(action_value, critic_value)
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        # make_dot(actor_loss)

        # Get the discounted expected future reward for each q-state
        with torch.no_grad():
            # action_probs_, log_probs_, _ = self.actor.forward(state_)
            # q_values_ = torch.min(self.future_value_1.forward(state_), self.future_value_2.forward(state_))
            # qf_ = torch.mul(action_probs_, (q_values_ - self.entropy * log_probs_))
            # qf_ = qf_.sum(dim=1)
            # qf_[done] = 0.
            # q_hat = self.scale * reward + self.gamma * qf_
            q_hat = self.scale * rewards + self.gamma * value_

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q1_new_policy_action_value = q1_new_policy.gather(1, action.long()).view(-1)
        q2_new_policy_action_value = q2_new_policy.gather(1, action.long()).view(-1)

        # embed()
        critic_1_loss = 0.5 * F.mse_loss(q1_new_policy_action_value, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_new_policy_action_value, q_hat)
        critic_loss = critic_1_loss + critic_2_loss

        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()
        
        if not self.created_dot_graphs: 
            make_dot(critic_loss).render('critic_loss.gv', view=True)
            make_dot(actor_loss).render('actor_loss.gv', view=True)
            make_dot(value_loss).render('value_loss.gv', view=True)
            self.created_dot_graphs = True

        if (self.auto_entropy):
            # JSS: (iv) similarly to (iii) we directly compute the expectation
            self.alpha_optim.zero_grad()
            weighted_log_probs = torch.sum(action_probs * log_probs, dim=1) # from authors repo
            alpha_loss = -(self.log_alpha * (weighted_log_probs + self.target_entropy).detach()).mean() # log_alpha instead of 
            alpha_loss.backward()


        # self.actor.optimizer.zero_grad()

        # watch = [("fc1W", self.actor.fc1.weight),
        #         ("fc2W", self.actor.fc2.weight),
        #         ("fc3W", self.actor.fc3.weight),
        #         ("fc1B", self.actor.fc1.bias),
        #         ("fc2B", self.actor.fc2.bias),
        #         ("fc3B", self.actor.fc3.bias)]
        # a = draw_graph(actor_loss, watch)
        # embed()
        # a = draw_graph(actor_loss)
        # embed()
        # actor_loss.backward(retain_graph=True)
        # critic_loss.backward(retain_graph=True)
        # actor_loss.backward(retain_graph=True)
        # print("a_loss ", actor_loss)
        # print(self.actor.fc2.weight.grad)
        # if abs(torch.sum(self.actor.fc3.weight.grad)) <= 0.:
        #     print("zero grad")
        #     embed()
        # else: print("actor loss ", actor_loss)
        # critic_loss.backward()
        
        # embed()
        
        # self.actor.optimizer.step()

        # self.critic_1.optimizer.step()
        # self.critic_2.optimizer.step()


        if (self.auto_entropy): 
            self.alpha_optim.step()
            self.entropy = self.log_alpha.exp()

        if update_params:
            self.update_network_parameters()

        # losses = [actor_loss.float(), value_loss.float(), critic_loss.float(), alpha_loss.float()]

        # embed()
        # return actor_loss.detach().numpy(), 0., critic_loss.detach().numpy(), alpha_loss.detach().numpy() if self.auto_entropy else 0
        return actor_loss.detach().numpy(), value_loss.detach().numpy(), critic_loss.detach().numpy(), alpha_loss.detach().numpy() if self.auto_entropy else 0, skill_loss.detach().numpy()

    def log(self, to_print):
        if (self.verbose):
            print(to_print)