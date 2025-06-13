import torch
import torch.nn as nn
import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, M=10, N=2, state_dim=4, hidden_dim=64, std_init=[2.0, 1.0]):
        super().__init__()

        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.sequence_len = N + 1 + M

        self.encoder_gru = nn.GRU(
            input_size=self.state_dim,
            hidden_size=self.hidden_dim,
            batch_first=True
        )

        self.actor_fc = nn.Sequential(
            layer_init(nn.Linear(self.hidden_dim, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 64)),
            nn.ReLU()
        )
        self.actor_output = layer_init(nn.Linear(64, 2), std=0.01)
        self.actor_logstd = nn.Parameter(torch.tensor(std_init))

        self.critic_fc = nn.Sequential(
            layer_init(nn.Linear(self.hidden_dim, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
    
    def actor_parameters(self):
        return list(self.encoder_gru.parameters()) + list(self.actor_fc.parameters()) + list(self.actor_output.parameters()) + [self.actor_logstd]

    def critic_parameters(self):
        return list(self.critic_fc.parameters())


    def encode_sequence(self, x):
        _, h = self.encoder_gru(x)
        return h[-1]

    def forward_actor(self, x):
        z = self.encode_sequence(x)
        x = self.actor_fc(z)
        mu = self.actor_output(x)
        return mu

    def forward_critic(self, x):
        z = self.encode_sequence(x)
        return self.critic_fc(z)

    def get_value(self, x):
        return self.forward_critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.forward_actor(x)
        std = self.actor_logstd.exp().expand_as(logits)
        dist = torch.distributions.Normal(logits, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.forward_critic(x)
        return action, log_prob, entropy, value

    def print_actor_architecture(self):
        print("Actor Architecture:")
        print(self.encoder_gru)
        print(self.actor_fc)
        print(self.actor_output)