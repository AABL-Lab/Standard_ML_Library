import gym

import numpy as np
#%matplotlib inline

import torch

from tensorboardX import SummaryWriter

from PS_utils import sample_normal
from sac_torch import Agent

from mpi_utils.normalizer import normalizer
from torch.distributions.normal import Normal


def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'goal2': obs['desired_goal2'].shape[0],
            'goal3': obs['desired_goal3'].shape[0],
            'goal4': obs['desired_goal4'].shape[0],
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high[0],
            }
    # params['max_timesteps'] = env._max_episode_steps
    params['max_timesteps'] = 400
    return params

def preproc_inputs(env_params, obs, g, g2, g3, g4):
    o_norm = normalizer(size=env_params['obs'])
    g_norm = normalizer(size=env_params['goal'])
    g2_norm = normalizer(size=env_params['goal2'])
    g3_norm = normalizer(size=env_params['goal3'])
    g4_norm = normalizer(size=env_params['goal4'])
    obs_norm =  o_norm.normalize(obs)
    g_norm = g_norm.normalize(g)
    g2_norm = g2_norm.normalize(g2)
    g3_norm = g3_norm.normalize(g3)
    g4_norm = g4_norm.normalize(g4)
    # concatenate the stuffs
    inputs = np.concatenate([obs_norm, g_norm, g2_norm, g3_norm, g4_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
    return inputs

def sample_normal(actor, observation, with_noise=False, max_action=2):
    def get_dist_env(agent, observation):
        observation = torch.Tensor([observation]).to('cpu')
        mu1, sigma1 = agent.actor.get_dist(observation)
        mu1 = mu1[0].detach().numpy()
        sigma1 = sigma1[0].detach().numpy()
        mu = mu1
        sigma = sigma1
        mu = torch.from_numpy(mu)
        sigma = torch.from_numpy(sigma)
        #print(mu, sigma)
        return Normal(mu, sigma), mu1, sigma1

    dist, mu, sigma = get_dist_env(actor, observation)
    if with_noise:
        sample = dist.rsample().numpy()
    else:
        sample = dist.sample().numpy()
    #print(sample)
    sample = max_action * np.tanh(sample)
    return sample, dist, mu, sigma

env = gym.make('FetchPush-v1')

env_params = get_env_params(env)
env._max_episode_steps = 100

# Note: these credit assignment intervals impact how the agent behaves a lot.
# Because of this sensitivity the model is overall very sensitive.
# There is a frame time delay of .1 so teaching is not boring. Could make the 
# the agent much better if played at around 10 frames per second (not sure of  current fps)
interval_min = 0
interval_max = .01

episodes=1000
USE_CUDA = torch.cuda.is_available()
lr = .001
replay_buffer_size = 100000
learn_buffer_interval = 200  # interval to learn from replay memory
batch_size = 200
print_interval = 1000
log_interval = 1000
learning_start = 100
win_break = True
queue_size = 2000

n_actions = 2
max_action = 1

observation_space_shape = np.array([37])

actor = Agent(alpha=lr, beta=lr, input_dims=observation_space_shape, env=env,  # learns from env
            n_actions=n_actions, auto_entropy=False)

#oracle = Oracle(torch.load("PS_sac_oracle.pkl"))

frame = env.reset()


episode_rewards = []
all_rewards = []
sum_rewards=[]
losses = []
episode_num = 0
is_win = False
env_interacts = 0

cnt=0
start_f=0
end_f=0

rewards = []
e = 0
render = False

for i in range(episodes):
    start_f=end_f
    loss = 0
    #env.render()
    observation = env.reset()
    obs = observation['observation']
    ag = observation['achieved_goal']
    g = observation['desired_goal']
    g2 = observation['desired_goal2']
    g3 = observation['desired_goal3']
    g4 = observation['desired_goal4']
    ep_rewards = 0
    feedback_value = 0
    
    while(True):
        end_f+=1
        env_interacts+=1
        input_tensor = preproc_inputs(env_params, obs, g, g2, g3, g4)
        input_tensor = input_tensor[0].numpy()
        action, dist, mu, sigma = sample_normal(actor, input_tensor, with_noise=False, max_action=max_action)
        #action = actor.choose_action(observation)
        old_observation = observation
        old_input_tensor = input_tensor
        observation, reward, done, _ = env.step(np.append(action,[0,0]))
        obs = observation['observation']
        ag = observation['achieved_goal']
        g = observation['desired_goal']
        g2 = observation['desired_goal2']
        g3 = observation['desired_goal3']
        g4 = observation['desired_goal4']
        input_tensor = preproc_inputs(env_params, obs, g, g2, g3, g4)
        input_tensor = input_tensor[0].numpy()
        actor.remember(old_input_tensor, action, reward, input_tensor, done)
        episode_rewards.append(reward)
        ep_rewards += reward

        actor.learn()

        if done:
            """print("frames: %5d, reward: %5f, loss: %4f, episode: %4d" % (end_f-start_f, np.sum(episode_rewards),
                                                                                    loss, episode_num))"""                                                                          
            print(ep_rewards)
            print("Episode:", str(i))
            rewards.append(ep_rewards)
            ep_rewards = 0
            all_rewards.append(episode_rewards)
            sum_rewards.append(np.sum(episode_rewards))
            episode_rewards = []
            episode_num += 1
            avg_reward = float(np.mean(episode_rewards[-10:]))
            frame = env.reset()
            break
data1 = rewards

np.save("push_3Delay", rewards)