import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden1=128, hidden2=96):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, action_size)
        
        #self.bn_in = nn.BatchNorm1d(state_size)
        #self.bn1 = nn.BatchNorm1d(hidden1)
        #self.bn2 = nn.BatchNorm1d(hidden2)
        
    def forward(self, state):
        
        #x = self.bn_in(state)
        x = F.leaky_relu(self.fc1(state))
        
        #x = self.bn1(x)
        x = F.leaky_relu(self.fc2(x))
        
        #x = self.bn2(x)
        x = torch.tanh(self.fc3(x))
        return x

class Critic(nn.Module):
    def __init__(self,state_size, action_size, hidden1=128, hidden2=96):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden1)
        self.fc2 = nn.Linear(hidden1+action_size, hidden2)
        self.fc3 = nn.Linear(hidden2,1)
        
        #self.bn_in = nn.BatchNorm1d(state_size)
        #self.bn1 = nn.BatchNorm1d(hidden1)
        #self.bn2 = nn.BatchNorm1d(hidden2)
        
    def forward(self, xs):
        x, a = xs
        
        #out = self.bn_in(x)
        out = F.leaky_relu(self.fc1(x))
        
        #out = self.bn1(out)
        out = F.leaky_relu(self.fc2(torch.cat([out,a],1)))
        
        #out = self.bn2(out)
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