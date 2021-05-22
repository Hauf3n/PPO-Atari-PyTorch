import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from Network import PPO_Network

class PPO_Agent(nn.Module):
    
    def __init__(self, in_channels, num_actions):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_actions = num_actions
        self.network = PPO_Network(in_channels, num_actions)
    
    def forward(self, x):
        policy, value = self.network(x)
        return policy, value
    
    def select_action(self, policy):
        return np.random.choice(range(self.num_actions) , 1, p=policy)[0]