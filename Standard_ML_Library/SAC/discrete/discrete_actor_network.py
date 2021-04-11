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

class DiscreteActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, n_actions, fc1_dims=256, 
            fc2_dims=256, name='actor', chkpt_dir='tmp/sac'):
        super(DiscreteActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        # self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        # self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        prob = F.relu(self.fc1(state))
        prob = F.relu(self.fc2(prob))
        action_values = F.relu(self.fc3(prob))

        action_probs = F.softmax(action_values, dim=-1)

        z = action_values == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)

        return action_probs , log_action_probabilities

    def sample_action(self, state, reparameterize=False):
        with torch.no_grad():
            action_probs, log_probs = self.forward(state)
            a_idx = np.random.choice([i for i in range(self.n_actions)], p=action_probs.detach().numpy()[0])

        # print(action_probs)
        return a_idx, log_probs

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))