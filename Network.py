import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class PPO_Network(nn.Module):
    # nature paper architecture
    
    def __init__(self, in_channels, num_actions):
        super().__init__()
        self.num_actions = num_actions
        
        network = [
            torch.nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7,512),
            nn.ReLU(),
            nn.Linear(512, num_actions + 1)
        ]
        
        self.network = nn.Sequential(*network)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        policy, value = torch.split(self.network(x),(self.num_actions, 1), dim=1)
        policy = self.softmax(policy)
        return policy, value