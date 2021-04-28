import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np

from IPython import embed

# Policy network
## Unlike the continuous version of SAC, we don't need to produce a distribution anymore and can softmax the last layer of the policy network
## point (ii) from discrete SAC

class DiscreteSkillNetwork(nn.Module):
    def __init__(self, alpha, input_dims, n_skills, fc1_dims=256, 
            fc2_dims=256, name='skill', chkpt_dir='tmp/sac'):
        super(DiscreteSkillNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_skills = n_skills
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_skills)
        # self.soft = nn.Softmax()

        # self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        # self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        prob = F.leaky_relu(self.fc1(state))
        prob = F.leaky_relu(self.fc2(prob))   
        skill_values = F.leaky_relu(self.fc3(prob))

        # action_probs = self.soft(action_values)
        skill_probs = F.softmax(skill_values, dim=1)
        
        z = skill_probs == 0.0
        z = z.float() * 1e-8
        log_skill_probabilities = torch.log(skill_probs + z)

        # if float("-inf") in log_action_probabilities:
        #     embed()

        return skill_probs , log_skill_probabilities, skill_values

    def save_checkpoint(self, suffix=''):
        torch.save(self.state_dict(), self.checkpoint_file+suffix)

    def load_checkpoint(self, suffix=''):
        self.load_state_dict(torch.load(self.checkpoint_file+suffix))