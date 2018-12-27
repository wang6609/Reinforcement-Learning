import numpy as np
import random
import copy
from collections import namedtuple, deque
from NN_model import Critic,Actor

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  
BATCH_SIZE = 48        
GAMMA = 0.99            
TAU = 0.001              
LR_actor =  0.00015            
LR_critic = 0.001
UPDATE_EVERY = 1     

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

criterion = torch.nn.MSELoss()

class Agent():

    def __init__(self, state_size, action_size,seed):

        self.state_size = state_size
        self.action_size = action_size
        
        
        #Actor
        self.actor_local = Actor(state_size, action_size).to(device)
        self.actor_target = Actor(state_size, action_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr = LR_actor) 

        # Critic
        self.critic_local = Critic(state_size, action_size).to(device)
        self.critic_target = Critic(state_size, action_size).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr = LR_critic)

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.noise = OUNoise(action_size, seed)

        self.t_step = 0
        
        
    
    def step(self, state, action, reward, next_state, done):

        if reward > 0:
            for i in range(4):
                self.memory.add(state,action,reward,next_state,done)
              
        self.memory.add(state,action,reward,next_state,done)
        
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
    
    def reset(self):
        self.noise.reset()
        
    def act(self, state, add_noise = True):

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state)
        self.actor_local.train()
        
        if add_noise:
            action += torch.from_numpy(self.noise.sample()).float()
        return np.clip(action, -1, 1).cpu().data.numpy()
        
        
    def learn(self, experiences, gamma):

        states, actions, rewards, next_states, dones = experiences
        
        # Critic update
        next_q_values = self.critic_target([next_states,self.actor_target(next_states)])
        target_q = rewards + gamma*(1 - dones) * next_q_values
        q = self.critic_local([states, actions])

        value_loss = F.mse_loss(q, target_q)
        
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
        #-----------------------------------------------------------------------------------------------------------------#

        # Actor update
        policy_loss = -self.critic_local([states,self.actor_local(states)]) 
        policy_loss = policy_loss.mean()
        
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        self.actor_optimizer.step()
        
        # --------------------update target network-----------------------------------------------------------------------#
        self.soft_update(self.actor_local, self.actor_target, TAU)
        self.soft_update(self.critic_local, self.critic_target, TAU) 
        # ----------------------------------------------------------------------------------------------------------------#
    
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
        

class ReplayBuffer:

    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
    
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state