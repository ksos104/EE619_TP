import torch
import torch.nn as nn

class Actor_Network(nn.Module):
    """
    Actor Network

    input: state
    output: action
    """
    def __init__(self, observation_size, action_size, hidden_units):
        super(Actor_Network, self).__init__()
        self.fc1 = nn.Linear(observation_size, hidden_units[0])
        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc3 = nn.Linear(hidden_units[1], action_size)
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

    def update_model(self, q_value):
        with torch.autograd.set_detect_anomaly(True):
            actor_loss = -1 * q_value.clone()
            self.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.optimizer.step()

            return actor_loss

    def update_target_model(self):
        with torch.no_grad():
            model_dict = self.model.state_dict()
            target_model_dict = self.target_model.state_dict()

            for key in model_dict.keys():
                target_model_dict[key] = self.tau * model_dict[key] + (1 - self.tau) * target_model_dict[key]
            