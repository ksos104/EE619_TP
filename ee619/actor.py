import torch
import torch.nn as nn
import numpy as np

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor_Network(nn.Module):
    """
    Actor Network

    input: state
    output: action
    """
    def __init__(self, observation_size, action_size, hidden_units, eps=3e-2):
        super(Actor_Network, self).__init__()
        self.fc1 = nn.Linear(observation_size, hidden_units[0])
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3 = nn.Linear(hidden_units[1], action_size)
        self.fc3.weight.data.uniform_(-eps, eps)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.relu(self.fc1(input))
        output = self.relu(self.fc2(output))
        output = self.tanh(self.fc3(output))

        return output

class Actor(object):
    def __init__(self, observation_size, action_size, hidden_units, learning_rate, tau):
        super(Actor, self).__init__()

        self.observation_size = observation_size
        self.action_size = action_size
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.tau = tau

        self.model = Actor_Network(observation_size=self.observation_size, action_size=self.action_size, hidden_units=self.hidden_units)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.target_model = Actor_Network(observation_size=self.observation_size, action_size=self.action_size, hidden_units=self.hidden_units)

        for target_weight, weight in zip(self.target_model.parameters(), self.model.parameters()):
            target_weight.data.copy_(weight.data)

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.target_model = self.target_model.cuda()

    def update_model(self, q_values):
        with torch.autograd.set_detect_anomaly(True):
            actor_loss = (-q_values).mean()
            self.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.optimizer.step()

            return actor_loss

    def update_target_model(self):
        for target_weight, weight in zip(self.target_model.parameters(), self.model.parameters()):
            target_weight.data.copy_(target_weight.data * (1.0 - self.tau) + weight.data * self.tau)
            