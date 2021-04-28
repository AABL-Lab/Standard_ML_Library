"""
This code mainly follows a Soft-Actor Critic YouTube tutorial found at:
https://www.youtube.com/watch?v=ioidsRlf79o&t=2649s
Channel name: Machine Learning with Phil

Any modifiations are made by the AABL Lab.
"""

import numpy as np
import torch
from IPython import embed

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

class DiscreteReplayBuffer():
    def __init__(self, max_size, input_shape, n_skills=5):
        # self.n_actions = n_actions
        self.mem_size = max_size
        self.mem_cntr = 0
        self.memory = np.zeros(max_size)
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        # self.action_memory = np.zeros((self.mem_size, n_actions))
        self.action_memory = np.zeros((self.mem_size, 1))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
        self.skill_memory = np.zeros((self.mem_size, n_skills), dtype=np.int)
        self.n_skills=n_skills

    def store_transition(self, state, action, reward, state_, done, skill=None):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        # self.action_memory[index][action] = 1
        # self.action_memory[index] = get_one_hot(np.array([action]), self.n_actions)
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        if (skill is not None):
            self.skill_memory[index] = torch.nn.functional.one_hot(skill, self.n_skills)[0]


        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]
        skills = self.skill_memory[batch]

        return states, actions, rewards, states_, dones, skills


