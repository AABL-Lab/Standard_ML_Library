"""
This code mainly follows a td3 YouTube tutorial found at:
https://www.youtube.com/watch?v=ZhFO8EWADmY&t=1895s
Channel name: Machine Learning with Phil

Any modifiations are made by the AABL Lab.
"""

import gym
import numpy as np
from td3_agent import Agent

if __name__ == '__main__':
    env = gym.make('BipedalWalker-v3')
    agent = Agent(alpha=0.0003, beta=0.0003, input_dims=env.observation_space.shape, tau=0.005,
            env=env, batch_size=100, layer1_size=400, layer2_size=300, n_actions=env.action_space.shape[0])
    
    n_games = 2000
    plot_filename = 'plots/' + "BipedalWalker_" + str(n_games) + '_games.png'

    best_score = env.reward_range[0]
    score_history = []

    #agent.load_models()

    best_reward = -1000

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            score += reward
            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models
        
        print("episode ", i, "score %.1f" % score, "average score %.1f" % avg_score)

        if score > best_reward:
            best_reward = score
            print("NEW BEST: ", score)
        