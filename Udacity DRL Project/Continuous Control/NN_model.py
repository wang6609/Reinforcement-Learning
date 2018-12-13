import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden1=256, hidden2=128):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

class Critic(nn.Module):
    def __init__(self,state_size, action_size, hidden1=256, hidden2=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden1)
        self.fc2 = nn.Linear(hidden1+action_size, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
      
    def forward(self, xs):
        x, a = xs
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(torch.cat([out,a],1)))
        out = self.fc3(out)
        return out
'''

class Critic(nn.Module):
    def __init__(self,state_size, action_size, hidden1=32, hidden2=16):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size+action_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
      
    def forward(self, sa):
        s,a = sa
        x = F.relu(self.fc1(torch.cat([s,a],1)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
'''