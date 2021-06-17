import os
from os.path import abspath, dirname, realpath
import torch

from agent import Agent
import gym
import pybullet_envs

import time
from tensorboardX import SummaryWriter

ROOT = dirname(abspath(realpath(__file__)))  # path to the ee619 directory

def main():
    now = time.localtime()
    dir_name = '{0:04d}-{1:02d}-{2:02d}_{3:02d}-{4:02d}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)
    summary = SummaryWriter(os.path.join(ROOT, 'logs/{}'.format(dir_name)))
    output_dir = os.path.join(ROOT, 'trained_models/{}'.format(dir_name))
    os.makedirs(output_dir, exist_ok=True)

    env = gym.make('Walker2DBulletEnv-v0')
    env.seed(0)
    env.render()
    agent = Agent()

    # seed = 0
    # repeat = 10000

    best_reward = 0.0

    seed_ = 0
    while True:
    # for seed_ in range(seed, seed + repeat):
        observation = env.reset()
        agent.ounoise.reset()
        done = False

        actor_loss = 0.0
        critic_loss = 0.0
        reward_sum = 0.0

        agent.decay_epsilon()

        step = 0
        while not done:
            action = agent.act(observation, is_training=True)
            next_observation, reward, done, _ = env.step(action)

            agent.push_memory(observation, action, reward, next_observation, done)
            loss_a, loss_c = agent.train()
            actor_loss += loss_a
            critic_loss += loss_c
            reward_sum += reward

            observation = next_observation
            step += 1

        summary.add_scalar('actor/model_loss', actor_loss/step, seed_)
        summary.add_scalar('critic/model_loss', critic_loss/step, seed_)
        summary.add_scalar('reward', reward_sum, seed_)

        if reward_sum >= best_reward:
            torch.save(agent.actor.model.state_dict(), '{}/actor.pkl'.format(output_dir, seed_))
            torch.save(agent.critic.model.state_dict(), '{}/critic.pkl'.format(output_dir, seed_))
            torch.save(agent.actor.target_model.state_dict(), '{}/actor_t.pkl'.format(output_dir, seed_))
            torch.save(agent.critic.target_model.state_dict(), '{}/critic_t.pkl'.format(output_dir, seed_))

            with open(os.path.join(ROOT, 'logs/{}.txt'.format(dir_name)), 'a') as f:
                f.write("(Episode {}: Reward {}) The best model parameters were saved.\n".format(seed_, reward_sum))

            best_reward = reward_sum

        seed_ += 1
        
if __name__ == '__main__':
    main()