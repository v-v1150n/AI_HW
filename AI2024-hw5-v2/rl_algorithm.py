import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import YOUR_CODE_HERE
import utils

class PacmanActionCNN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PacmanActionCNN, self).__init__()
        # build your own CNN model
        "*** YOUR CODE HERE ***"
        utils.raiseNotDefined()
        # this is just an example, you can modify this.
        self.conv1 = nn.Conv2d(state_dim, 16, kernel_size=8, stride=4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        
        "*** YOUR CODE HERE ***"
        utils.raiseNotDefined()
        
        return x

class ReplayBuffer:
    # referenced [TD3 official implementation](https://github.com/sfujim/TD3/blob/master/utils.py#L5).
    def __init__(self, state_dim, action_dim, max_size=int(1e5)):
        self.states = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, *action_dim), dtype=np.int64)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.terminated = np.zeros((max_size, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0
        self.max_size = max_size

    def update(self, state, action, reward, next_state, terminated):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.terminated[self.ptr] = terminated
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, batch_size)
        return (
            torch.FloatTensor(self.states[ind]),
            torch.FloatTensor(self.actions[ind]),
            torch.FloatTensor(self.rewards[ind]),
            torch.FloatTensor(self.next_states[ind]),
            torch.FloatTensor(self.terminated[ind]), 
        )

class DQN:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-4,
        epsilon=0.9,
        epsilon_min=0.05,
        gamma=0.99,
        batch_size=64,
        warmup_steps=5000,
        buffer_size=int(1e5),
        target_update_interval=10000,
    ):
        """
        DQN agent has four methods.

        - __init__() as usual
        - act() takes as input one state of np.ndarray and output actions by following epsilon-greedy policy.
        - process() method takes one transition as input and define what the agent do for each step.
        - learn() method samples a mini-batch from replay buffer and train q-network
        """
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.target_update_interval = target_update_interval

        self.network = PacmanActionCNN(state_dim[0], action_dim)
        self.target_network = PacmanActionCNN(state_dim[0], action_dim)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = torch.optim.RMSprop(self.network.parameters(), lr)

        self.buffer = ReplayBuffer(state_dim, (1, ), buffer_size)
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        self.network.to(self.device)
        self.target_network.to(self.device)
        
        self.total_steps = 0
        self.epsilon_decay = (epsilon - epsilon_min) / 1e6
    
    @torch.no_grad()
    def act(self, x, training=True):
        self.network.train(training)
        if training and ((np.random.rand() < self.epsilon) or (self.total_steps < self.warmup_steps)):
            # Random action
            action = np.random.randint(0, self.action_dim)
        else:
            # output actions by following epsilon-greedy policy
            x = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
            
            "*** YOUR CODE HERE ***"
            utils.raiseNotDefined()
            # get q-values from network
            q_value = YOUR_CODE_HERE
            # get action with maximum q-value
            action = YOUR_CODE_HERE
        
        return action
    
    def learn(self):
        "*** YOUR CODE HERE ***"
        utils.raiseNotDefined()
        
        # sample a mini-batch from replay buffer
        state, action, reward, next_state, terminated = map(lambda x: x.to(self.device), self.buffer.sample(self.batch_size))
        
        # get q-values from network
        next_q = YOUR_CODE_HERE
        # td_target: if terminated, only reward, otherwise reward + gamma * max(next_q)
        td_target = YOUR_CODE_HERE
        # compute loss with td_target and q-values
        loss = YOUR_CODE_HERE
        
        # initialize optimizer
        "self.optimizer.YOUR_CODE_HERE"
        # backpropagation
        YOUR_CODE_HERE
        # update network
        "self.optimizer.YOUR_CODE_HERE"
        
        return {YOUR_CODE_HERE} # return dictionary for logging
    
    def process(self, transition):
        "*** YOUR CODE HERE ***"
        utils.raiseNotDefined()
        
        result = {}
        self.total_steps += 1
        
        # update replay buffer
        "self.buffer.YOUR_CODE_HERE"

        if self.total_steps > self.warmup_steps:
            result = self.learn()
            
        if self.total_steps % self.target_update_interval == 0:
            # update target networ
            "self.target_network.YOUR_CODE_HERE"
        
        self.epsilon -= self.epsilon_decay
        return result