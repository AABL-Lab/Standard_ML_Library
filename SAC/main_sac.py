"""
This code mainly follows a Soft-Actor Critic YouTube tutorial found at:
https://www.youtube.com/watch?v=ioidsRlf79o&t=2649s
Channel name: Machine Learning with Phil

Any modifiations are made by the AABL Lab.
"""

import pybullet_envs
import gym
import numpy as np
from sac_torch import Agent
from gym import wrappers

if __name__ == '__main__':
    env = gym.make('BipedalWalker-v3')
    agent = Agent(alpha=0.000314854, beta=0.000314854, input_dims=env.observation_space.shape, env=env, batch_size=128,
            tau=.02, max_size=50000, layer1_size=400, layer2_size=300, n_actions=env.action_space.shape[0], reward_scale=1, auto_entropy=True)
    n_games = 1000
    rewards = []
    # uncomment this line and do a mkdir tmp && mkdir video if you want to
    # record video of the agent playing the game.
    #env = wrappers.Monitor(env, 'tmp/video', video_callable=lambda episode_id: True, force=True)
    filename = 'LunarLanderContinuous.png'

    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')

    env_interacts = 0
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            env_interacts+=1
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                if env_interacts > 1000:
                    if env_interacts % 128 == 0:
                        agent.learn(update_params=True)
                    else:
                        agent.learn()
            observation = observation_
            #env.render()
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            #if not load_checkpoint:
            #    agent.save_models()

        rewards.append(score)

        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)
    
    np.save("BP_sac_2000", rewards)
