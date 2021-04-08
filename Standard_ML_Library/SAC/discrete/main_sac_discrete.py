import gym
from gym import wrappers
import numpy as np

import gym_novel_gridworlds
from gym_novel_gridworlds.wrappers import SaveTrajectories, LimitActions, RandomizeInventory
from gym_novel_gridworlds.observation_wrappers import LidarInFront, AgentMap

from SAC.discrete.discrete_sac_torch import DiscreteAgent


from IPython import embed

if __name__ == '__main__':
    env_name = 'NovelGridworld-Bow-v0'

    env = gym.make(env_name)
    env.unbreakable_items.add('crafting_table') # Make crafting table unbreakable for easy solving of task.
    env = LimitActions(env, {'Forward', 'Left', 'Right', 'Break', 'Craft_bow'}) # limit actions for easy training
    # env = RandomizeInventory(env, string_range=(2,3), stick_range=(2,3))
    env = RandomizeInventory(env, string_range=(3,3), stick_range=(3,3))
    env = LidarInFront(env) # generate the observation space using LIDAR sensors
    env.reward_done = 100
    # env.reward_intermediate = 50 # TODO: what is this

    a = 0.000314854
    b = 0.000314854
    entr_lr = 0.00003
    s0 = 100#400
    s1 = 30#300

    agent = DiscreteAgent(alpha=a, beta=b, input_dims=env.observation_space.shape, env=env, batch_size=128,
            tau=.02, max_size=50000, layer1_size=s0, layer2_size=s1, n_actions=env.action_space.n, reward_scale=1, auto_entropy=True, entr_lr=entr_lr)
    n_games = 10000
    max_steps_per_game = 100
    rewards = []
    # uncomment this line and do a mkdir tmp && mkdir video if you want to
    # record video of the agent playing the game.
    #env = wrappers.Monitor(env, 'tmp/video', video_callable=lambda episode_id: True, force=True)
    filename = env_name + '.png'

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
        # while not done:
        for _ in range(max_steps_per_game):
            env_interacts+=1
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            # print("action %d reward %f" % (action, reward))
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                if env_interacts > 1000:
                    if env_interacts % 128 == 0:
                        agent.learn(update_params=True)
                    else:
                        agent.learn()
            observation = observation_
            if done: break
            if (load_checkpoint):
                env.render()
                
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
               agent.save_models()

        rewards.append(score)

        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score, 'entropy %.5f' % agent.entropy)
    
    np.save("polycraft_discrete_sac", rewards)
