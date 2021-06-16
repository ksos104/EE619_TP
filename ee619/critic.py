import torch
import torch.nn as nn
import numpy as np
    
class Critic_Network(nn.Module):
    """
    Critic Network

    input: state, action
    output: q value
    """
    def __init__(self, observation_size, action_size, hidden_units):
        super(Critic_Network, self).__init__()
        self.fc1 = nn.Linear(observation_size, hidden_units[0])
        self.fc2 = nn.Linear(hidden_units[0] + action_size, hidden_units[1])
        self.fc3 = nn.Linear(hidden_units[1], 1)
        self.relu = nn.ReLU()

    def forward(self, observations, actions):
        input_ob = self.relu(self.fc1(observations))
        input = torch.cat((input_ob, actions), dim=1)

        output = self.relu(self.fc2(input))
        output = self.fc3(output)

        return output

class Critic(object):
    def __init__(self, observation_size, action_size, hidden_units, learning_rate, tau, gamma):
        super(Critic, self).__init__()

        self.observation_size = observation_size
        self.action_size = action_size
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        self.model = Critic_Network(observation_size=self.observation_size, action_size=self.action_size, hidden_units=self.hidden_units)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.target_model = Critic_Network(observation_size=self.observation_size, action_size=self.action_size, hidden_units=self.hidden_units)

        for target_weight, weight in zip(self.target_model.parameters(), self.model.parameters()):
            target_weight.data.copy_(weight.data)

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.target_model = self.target_model.cuda()

    def update_model(self, observations, actions, rewards, next_observations, next_actions, done):
        with torch.autograd.set_detect_anomaly(True):
            next_q_values = self.target_model(next_observations, next_actions).detach()

            target_q_values = rewards + done * self.gamma * next_q_values

            q_values = self.model(observations, actions)
            loss_function = nn.MSELoss()
            critic_loss = loss_function(target_q_values, q_values)

            self.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            self.optimizer.step()

            return critic_loss

    def update_target_model(self):
        for target_weight, weight in zip(self.target_model.parameters(), self.model.parameters()):
            target_weight.data.copy_(target_weight.data * (1.0 - self.tau) + weight.data * self.tau)
            