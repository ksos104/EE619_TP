import os
import torch

from agent import Agent
import gym
import pybullet_envs

import time
from tensorboardX import SummaryWriter

def main():
    now = time.localtime()
    dir_name = '{0:04d}-{1:02d}-{2:02d}_{3:02d}-{4:02d}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)
    summary = SummaryWriter('./logs/{}'.format(dir_name))
    output_dir = './trained_models/{}'.format(dir_name)
    os.makedirs(output_dir, exist_ok=True)

    env = gym.make('Walker2DBulletEnv-v0')
    agent = Agent()

    seed = 0
    repeat = 100000

    best_reward = 0.0

    for seed_ in range(seed, seed + repeat):
        env.seed(seed_)

        # observation = torch.from_numpy(env.reset())
        # if torch.cuda.is_available():
        #     observation = observation.cuda()
        observation = env.reset()
        done = False

        actor_loss = 0.0
        critic_loss = 0.0
        reward_sum = 0.0

        step = 0
        while not done:
            # action = agent.act(observation.cpu().detach().numpy(), is_training=True)  
            action = agent.act(observation, is_training=True)
            next_observation, reward, done, _ = env.step(action)
            
            # next_observation = torch.from_numpy(next_observation)
            # if torch.cuda.is_available():
            #     next_observation = next_observation.cuda()

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
            torch.save(agent.actor.model.state_dict(), '{}/actor.pkl'.format(output_dir))
            torch.save(agent.critic.model.state_dict(), '{}/critic.pkl'.format(output_dir))
        
if __name__ == '__main__':
    main()