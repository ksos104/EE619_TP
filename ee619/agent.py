"""Agent for a Walker2DBullet environment."""
import os
from os.path import abspath, dirname, realpath

from gym.spaces.box import Box
import numpy as np
import torch

from collections import deque
import random

ROOT = dirname(abspath(realpath(__file__)))  # path to the ee619 directory

import sys
sys.path.append(ROOT)
from actor import Actor
from critic import Critic

class Agent:
    """Agent for a Walker2DBullet environment."""
    def __init__(self):
        self._action_space = Box(-1, 1, (6,))
        self._action_space.seed(0)

        self.observation_size = 22
        self.action_size = self._action_space.shape[0]

        self.gamma = 0.99
        self.tau = 1e-3
        self.epsilon = 1.0
        self.epsilon_decay = 1e-3
        self.epsilon_min = 1e-3
        self.ounoise = OUNoise(self.action_size)

        self.batch_size = 64
        self.actor_hidden_units = (400, 300)
        self.critic_hidden_units = (400, 300)

        self.actor_learning_rate = 1e-4
        self.critic_learning_rate = 1e-3

        self.actor = Actor(observation_size=self.observation_size, action_size=self.action_size, hidden_units=self.actor_hidden_units, learning_rate=self.actor_learning_rate, tau=self.tau)
        self.critic = Critic(observation_size=self.observation_size, action_size=self.action_size, hidden_units=self.critic_hidden_units, learning_rate=self.critic_learning_rate, tau=self.tau, gamma=self.gamma)

        self.memory_size = 1e+6
        self.memory = deque()

    def train(self):
        if len(self.memory) > self.batch_size:
            observations, actions, rewards, next_observations, done = self.get_sample()

            observations = torch.Tensor(observations)
            actions = torch.Tensor(actions)
            rewards = torch.Tensor(rewards).unsqueeze(dim=1)
            next_observations = torch.Tensor(next_observations)
            done = (np.array(done) == False) * 1
            done = torch.Tensor(done).unsqueeze(dim=1)
            
            if torch.cuda.is_available():
                observations = observations.cuda()
                actions = actions.cuda()
                next_observations = next_observations.cuda()
                rewards = rewards.cuda()
                done = done.cuda()

            critic_loss = self.train_critic(observations, actions, rewards, next_observations, done)
            actor_loss = self.train_actor(observations)
            self.update_target_models()

            return actor_loss, critic_loss
        
        return 0, 0

    def train_actor(self, observations):
        actions = self.actor.model(observations)
        q_values = self.critic.model(observations, actions)

        actor_loss = self.actor.update_model(q_values)

        return actor_loss

    def train_critic(self, observations, actions, rewards, next_observations, done):
        next_actions = self.actor.target_model(next_observations).detach()

        critic_loss = self.critic.update_model(observations, actions, rewards, next_observations, next_actions, done)

        return critic_loss

    def update_target_models(self):
        self.critic.update_target_model()
        self.actor.update_target_model()

    def get_sample(self):
        sample = random.sample(self.memory, self.batch_size)
        observations, actions, rewards, next_observations, done = zip(*sample)
        return observations, actions, rewards, next_observations, done

    def push_memory(self, observation, action, reward, next_observation, done):
        self.memory.append((observation, action, reward, next_observation, done))
        if len(self.memory) > self.memory_size:
            self.memory.popleft()

    def decay_epsilon(self):
        self.epsilon -= self.epsilon_decay

    def act(self, observation: np.ndarray, is_training=False):
        """Decides which action to take for the given observation."""
        observation = torch.from_numpy(observation)
        if torch.cuda.is_available():
            observation = observation.cuda()

        action = self.actor.model(observation).cpu().detach().numpy()
        if is_training:
            action = action + max(self.epsilon, self.epsilon_min) * self.ounoise.noise()
        
        action = np.clip(action, -1.0, 1.0)

        return action

    def load(self):
        """Loads network parameters if there are any.

        Example:
            path = join(ROOT, 'model.pth')
            self.policy.load_state_dict(torch.load(path))
        """
        actor_model = os.path.join(ROOT, 'actor.pkl')
        critic_model = os.path.join(ROOT, 'critic.pkl')
        # actor_target_model = os.path.join(ROOT, 'actor_t.pkl')
        # critic_target_model = os.path.join(ROOT, 'critic_t.pkl')

        self.actor.model.load_state_dict(torch.load(actor_model))
        self.critic.model.load_state_dict(torch.load(critic_model))
        # self.actor.target_model.load_state_dict(torch.load(actor_target_model))
        # self.critic.target_model.load_state_dict(torch.load(critic_target_model))

class OUNoise:
    def __init__(self, action_size, mu=0, theta=0.15, sigma=0.2):
        self.action_size = action_size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_size) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_size) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx

        return self.state