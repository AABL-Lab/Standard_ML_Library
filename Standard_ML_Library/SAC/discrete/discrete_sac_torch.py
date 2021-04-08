import os
import torch 
import torch.nn.functional as F
import numpy as np
from Standard_ML_Library.SAC.discrete.buffer import ReplayBuffer
from Standard_ML_Library.SAC.discrete.discrete_actor_network import DiscreteActorNetwork
from Standard_ML_Library.SAC.discrete.discrete_critic_network import DiscreteCriticNetwork
from Standard_ML_Library.SAC.discrete.networks import ValueNetwork

from IPython import embed    

class DiscreteAgent():
    def __init__(self, alpha=0.0007, beta=0.0007, input_dims=[8],
            env=None, gamma=0.99, n_actions=2, max_size=1000000, tau=0.005,
            layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2, auto_entropy=False, 
            entr_lr=None, reparam_noise=1e-6, verbose=False):
        self.verbose = False
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.env = env
        self.auto_entropy = auto_entropy
        self.actor = DiscreteActorNetwork(alpha, input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size, n_actions=n_actions,
                    name='actor')
        self.critic_1 = DiscreteCriticNetwork(beta, input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size, n_actions=n_actions,
                    name='critic_1')
        self.critic_2 = DiscreteCriticNetwork(beta, input_dims,fc1_dims=layer1_size, fc2_dims=layer2_size, n_actions=n_actions,
                    name='critic_2')
        self.value = ValueNetwork(beta, input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size, name='value')
        self.target_value = ValueNetwork(beta, input_dims,fc1_dims=layer1_size, fc2_dims=layer2_size, name='target_value')

        self.scale = reward_scale
        self.update_network_parameters(tau=1)
        self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
        self.log_alpha = torch.zeros(1, requires_grad=True, device="cpu")
        if entr_lr is None:
            self.entr_lr = alpha
        else:
            self.entr_lr = entr_lr
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.entr_lr)
        self.entropy = 0

    def choose_action(self, observation):
        state = torch.Tensor([observation]).to(self.actor.device)

        actions, _ = self.actor.sample_action(state)

        # actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions #actions.cpu().detach().numpy()#[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + \
                    (1-tau)*target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)
    
    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def learn(self, update_params=True):
        if(self.auto_entropy):
            if self.memory.mem_cntr < self.batch_size:
                return

            self.log("entropy %f" % self.entropy)
            state, action, reward, new_state, done = \
                    self.memory.sample_buffer(self.batch_size)

            reward = torch.tensor(reward, dtype=torch.float).to(self.actor.device)
            done = torch.tensor(done).to(self.actor.device)
            state_ = torch.tensor(new_state, dtype=torch.float).to(self.actor.device)
            state = torch.tensor(state, dtype=torch.float).to(self.actor.device)
            action = torch.tensor(action, dtype=torch.float).to(self.actor.device)

            value = self.value(state).view(-1)
            value_ = self.target_value(state_).view(-1)
            value_[done] = 0.0

            # actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
            # embed()
            action_values, log_probs = self.actor.forward(state)
            action_probs = F.softmax(action_values, dim=-1)

            # log_probs = log_probs.view(-1)
            # q1_new_policy = self.critic_1.forward(state, actions)
            # q2_new_policy = self.critic_2.forward(state, actions)

            q1_new_policy = self.critic_1.forward(state)
            q2_new_policy = self.critic_2.forward(state)

            critic_value = torch.min(q1_new_policy, q2_new_policy) 
            # critic_value = critic_value.view(-1)

            self.value.optimizer.zero_grad()
            value_target = critic_value - self.entropy*log_probs

            # JSS point (iii) from the discrete sac paper talks about directly computing the expectation. 
            #   It's the likelihood of taking an action in a state multiplied by the value of that s-a pair. Summed over all actions for each state (i think)
            value_expectation = torch.sum(action_probs * value_target, dim=1)

            value_loss = 0.5 * F.mse_loss(value, value_expectation)
            # value_loss = 0.5 * F.mse_loss(value, value_target)
            self.log("value loss: %f" % value_loss)

            value_loss.backward(retain_graph=True)
            self.value.optimizer.step()

            # actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
            # _, log_probs = self.actor.forward(state)
            # actions, log_action_probs = self.actor.sample_action(state)
            # log_probs = log_probs.view(-1)
            # q1_new_policy = self.critic_1.forward(state, actions)
            q1_new_policy = self.critic_1.forward(state)
            # q2_new_policy = self.critic_2.forward(state, actions)
            q2_new_policy = self.critic_2.forward(state)
            critic_value = torch.min(q1_new_policy, q2_new_policy)
            # critic_value = critic_value.view(-1)
            
            # JSS (v) directly compute the expectation rather than doing reparameterization to pass the gradients through
            actor_loss = self.entropy*log_probs - critic_value
            actor_loss = torch.sum(action_probs * actor_loss)
            self.log("actor loss: %f" % actor_loss)

            # embed()
 
            self.actor.optimizer.zero_grad()

            # print(torch.max(self.actor.fc1.weight))
            # print(torch.min(self.actor.fc1.weight))
            # print(torch.max(self.actor.fc1.weight.grad))
            # print(torch.min(self.actor.fc1.weight.grad))
            # print("backward")
            actor_loss.backward(retain_graph=True)
            # print(torch.max(self.actor.fc1.weight))
            # print(torch.min(self.actor.fc1.weight))
            # print(torch.max(self.actor.fc1.weight.grad))
            # print(torch.min(self.actor.fc1.weight.grad))


            self.actor.optimizer.step()
            
            self.critic_1.optimizer.zero_grad()
            self.critic_2.optimizer.zero_grad()
            q_hat = self.scale*reward + self.gamma*value_
            # q1_old_policy = self.critic_1.forward(state).sum(dim=1)
            # q2_old_policy = self.critic_2.forward(state).sum(dim=1)
            q1_old_policy = self.critic_1.forward(state).mean(dim=1)
            q2_old_policy = self.critic_2.forward(state).mean(dim=1)
            # q1_old_policy = self.critic_1.forward(state, action).view(-1)
            # q2_old_policy = self.critic_2.forward(state, action).view(-1)
            critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
            critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)
            critic_loss = critic_1_loss + critic_2_loss
            self.log("critic loss: %f" % critic_loss)

            critic_loss.backward()
            self.critic_1.optimizer.step()
            self.critic_2.optimizer.step()

            # alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            # JSS: (iv) similarly to (iii) we directly compute the expectation
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach())
            alpha_loss = torch.sum(action_probs * alpha_loss)
            self.log("alpha loss: %f" % alpha_loss)

            # embed()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.entropy = self.log_alpha.exp()
            # print("entropy: %f" % self.entropy)

            if update_params:
                self.update_network_parameters()
        else:
            raise NotImplementedError

    def log(self, to_print):
        if (self.verbose):
            print(to_print)
