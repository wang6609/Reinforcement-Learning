import torch
import torch.nn as nn
import torch.nn.functional as F

#Normal Network architecture
class QNetwork(nn.Module):
    
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        
        super(QNetwork, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units,action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#Dueling Network   
class DuelingQNetwork(nn.Module):
    
    def __init__(self, state_size, action_size, seed,fc1_units=32,fc_value_1_units = 16,fc_advantage_1_units = 16):
        
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.action_size = action_size
        self.state_size = state_size

        self.fc1 = nn.Linear(state_size, fc1_units)
        
        self.fc_value_1 = nn.Linear(fc1_units,fc_value_1_units)
        self.fc_value_2 = nn.Linear(fc_value_1_units,1)
        
        self.fc_advantage_1 = nn.Linear(fc1_units,fc_advantage_1_units)
        self.fc_advantage_2 = nn.Linear(fc_advantage_1_units,action_size)
        
    def forward(self, state):
        
        batch_size = state.size(0)
        
        x = F.relu(self.fc1(state))
        
        value = F.relu(self.fc_value_1(x))
        value = F.relu(self.fc_value_2(value))
        value = value.expand(batch_size,self.action_size)

        advantage = F.relu(self.fc_advantage_1(x))
        advantage = F.relu(self.fc_advantage_2(advantage))

        x = value + (advantage - advantage.mean(1).unsqueeze(1).expand(batch_size, self.action_size))

        return x  