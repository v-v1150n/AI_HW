# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # import numpy as np

# # from utils import raiseNotDefined
# # import utils

# # class PacmanActionCNN(nn.Module):
# #     def __init__(self, state_dim, action_dim):
# #         super(PacmanActionCNN, self).__init__()
# #         # 建立 CNN 模型
# #         self.conv1 = nn.Conv2d(state_dim, 32, kernel_size=8, stride=4)
# #         self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
# #         self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
# #         self.fc1 = nn.Linear(64 * 7 * 7, 512)  # 假設輸入圖像大小是 (84, 84)
# #         self.fc2 = nn.Linear(512, action_dim)
    
# #     def forward(self, x):
# #         x = F.relu(self.conv1(x))
# #         x = F.relu(self.conv2(x))
# #         x = F.relu(self.conv3(x))
# #         x = x.view(x.size(0), -1)  # 展平張量
# #         x = F.relu(self.fc1(x))
# #         x = self.fc2(x)
# #         return x

# # class ReplayBuffer:
# #     # 參考 [TD3 official implementation](https://github.com/sfujim/TD3/blob/master/utils.py#L5)
# #     def __init__(self, state_dim, action_dim, max_size=int(1e5)):
# #         self.states = np.zeros((max_size, *state_dim), dtype=np.float32)
# #         self.actions = np.zeros((max_size, *action_dim), dtype=np.int64)
# #         self.rewards = np.zeros((max_size, 1), dtype=np.float32)
# #         self.next_states = np.zeros((max_size, *state_dim), dtype=np.float32)
# #         self.terminated = np.zeros((max_size, 1), dtype=np.float32)

# #         self.ptr = 0
# #         self.size = 0
# #         self.max_size = max_size

# #     def update(self, state, action, reward, next_state, terminated):
# #         self.states[self.ptr] = state
# #         self.actions[self.ptr] = action
# #         self.rewards[self.ptr] = reward
# #         self.next_states[self.ptr] = next_state
# #         self.terminated[self.ptr] = terminated
        
# #         self.ptr = (self.ptr + 1) % self.max_size
# #         self.size = min(self.size + 1, self.max_size)
        
# #     def sample(self, batch_size):
# #         ind = np.random.randint(0, self.size, batch_size)
# #         return (
# #             torch.FloatTensor(self.states[ind]),
# #             torch.FloatTensor(self.actions[ind]),
# #             torch.FloatTensor(self.rewards[ind]),
# #             torch.FloatTensor(self.next_states[ind]),
# #             torch.FloatTensor(self.terminated[ind]), 
# #         )

# # class DQN:
# #     def __init__(
# #         self,
# #         state_dim,
# #         action_dim,
# #         lr=1e-4,
# #         epsilon=0.9,
# #         epsilon_min=0.05,
# #         gamma=0.99,
# #         batch_size=64,
# #         warmup_steps=5000,
# #         buffer_size=int(1e5),
# #         target_update_interval=10000,
# #     ):
# #         """
# #         DQN agent has four methods.

# #         - __init__() as usual
# #         - act() takes as input one state of np.ndarray and output actions by following epsilon-greedy policy.
# #         - process() method takes one transition as input and define what the agent do for each step.
# #         - learn() method samples a mini-batch from replay buffer and train q-network
# #         """
# #         self.action_dim = action_dim
# #         self.epsilon = epsilon
# #         self.gamma = gamma
# #         self.batch_size = batch_size
# #         self.warmup_steps = warmup_steps
# #         self.target_update_interval = target_update_interval

# #         self.network = PacmanActionCNN(state_dim[0], action_dim)
# #         self.target_network = PacmanActionCNN(state_dim[0], action_dim)
# #         self.target_network.load_state_dict(self.network.state_dict())
# #         self.optimizer = torch.optim.RMSprop(self.network.parameters(), lr)

# #         self.buffer = ReplayBuffer(state_dim, (1, ), buffer_size)
# #         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# #         self.network.to(self.device)
# #         self.target_network.to(self.device)
        
# #         self.total_steps = 0
# #         self.epsilon_decay = (epsilon - epsilon_min) / 1e6
    
# #     @torch.no_grad()
# #     def act(self, x, training=True):
# #         self.network.train(training)
# #         if training and ((np.random.rand() < self.epsilon) or (self.total_steps < self.warmup_steps)):
# #             # 隨機動作
# #             action = np.random.randint(0, self.action_dim)
# #         else:
# #             # 根據 epsilon-greedy 策略選擇動作
# #             x = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
# #             q_value = self.network(x)
# #             action = q_value.max(1)[1].item()
        
# #         return action
    
# #     def learn(self):
# #         # 從回放緩衝區中隨機抽取一個 mini-batch
# #         state, action, reward, next_state, terminated = map(lambda x: x.to(self.device), self.buffer.sample(self.batch_size))
        
# #         # 獲取當前狀態的 Q 值
# #         q_values = self.network(state)
# #         next_q_values = self.target_network(next_state)
        
# #         # 計算 TD 目標值
# #         next_q_value = next_q_values.max(1)[0].unsqueeze(1)
# #         td_target = reward + (1 - terminated) * self.gamma * next_q_value
        
# #         # 計算損失
# #         q_value = q_values.gather(1, action.long())
# #         loss = F.mse_loss(q_value, td_target)
       
# #         # 初始化優化器並進行反向傳播
# #         self.optimizer.zero_grad()
# #         loss.backward()
# #         self.optimizer.step()
        
# #         return {"value_loss": loss.item()}
    
# #     def process(self, transition):
# #         state, action, reward, next_state, terminated = transition
# #         self.total_steps += 1
        
# #         # 更新回放緩衝區
# #         self.buffer.update(state, action, reward, next_state, terminated)

# #         result = {}
# #         if self.total_steps > self.warmup_steps:
# #             result = self.learn()
            
# #         if self.total_steps % self.target_update_interval == 0:
# #             # 更新目標網路
# #             self.target_network.load_state_dict(self.network.state_dict())
        
# #         self.epsilon = max(self.epsilon - self.epsilon_decay, 0.05)
# #         return result


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np

# class PacmanActionCNN(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(PacmanActionCNN, self).__init__()
#         self.conv1 = nn.Conv2d(state_dim[0], 32, kernel_size=8, stride=4)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1)  # 新增一層卷積層
#         self.bn4 = nn.BatchNorm2d(256)
#         self.dropout = nn.Dropout(p=0.5)
#         self.fc1 = nn.Linear(256 * 5 * 5, 512)  # 根據新的卷積層數量調整輸入大小
#         self.fc2 = nn.Linear(512, action_dim)

#     def forward(self, x):
#         x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01)
#         x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.01)
#         x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.01)
#         x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.01)  # 新增一層卷積層
#         x = x.view(x.size(0), -1)
#         x = F.leaky_relu(self.dropout(self.fc1(x)), negative_slope=0.01)
#         x = self.fc2(x)
#         return x

# class ReplayBuffer:
#     def __init__(self, state_dim, action_dim, max_size=int(1e5)):  # 調整緩衝區大小
#         self.states = np.zeros((max_size, *state_dim), dtype=np.float32)
#         self.actions = np.zeros((max_size, *action_dim), dtype=np.int64)
#         self.rewards = np.zeros((max_size, 1), dtype=np.float32)
#         self.next_states = np.zeros((max_size, *state_dim), dtype=np.float32)
#         self.terminated = np.zeros((max_size, 1), dtype=np.float32)
#         self.ptr = 0
#         self.size = 0
#         self.max_size = max_size

#     def update(self, state, action, reward, next_state, terminated):
#         self.states[self.ptr] = state
#         self.actions[self.ptr] = action
#         self.rewards[self.ptr] = reward
#         self.next_states[self.ptr] = next_state
#         self.terminated[self.ptr] = terminated
#         self.ptr = (self.ptr + 1) % self.max_size
#         self.size = min(self.size + 1, self.max_size)
        
#     def sample(self, batch_size):
#         ind = np.random.randint(0, self.size, batch_size)
#         return (
#             torch.FloatTensor(self.states[ind]),
#             torch.FloatTensor(self.actions[ind]),
#             torch.FloatTensor(self.rewards[ind]),
#             torch.FloatTensor(self.next_states[ind]),
#             torch.FloatTensor(self.terminated[ind]), 
#         )

# class DQN:
#     def __init__(
#         self,
#         state_dim,
#         action_dim,
#         lr=0.005,  # 調整學習率
#         epsilon=0.9,  # 初始 epsilon
#         epsilon_min=0.05,  # 最小 epsilon
#         epsilon_decay=5e-7,  # 衰減速度
#         gamma=0.99,
#         batch_size=64,
#         warmup_steps=1000,
#         buffer_size=int(1e5),  # 調整緩衝區大小
#         target_update_interval=10000,  # 增加目標網絡更新頻率
#     ):
#         self.action_dim = action_dim
#         self.epsilon = epsilon
#         self.epsilon_min = epsilon_min
#         self.epsilon_decay = epsilon_decay
#         self.gamma = gamma
#         self.batch_size = batch_size
#         self.warmup_steps = warmup_steps
#         self.target_update_interval = target_update_interval

#         self.network = PacmanActionCNN(state_dim, action_dim)
#         self.target_network = PacmanActionCNN(state_dim, action_dim)
#         self.target_network.load_state_dict(self.network.state_dict())
#         self.optimizer = torch.optim.Adam(self.network.parameters(), lr)

#         self.buffer = ReplayBuffer(state_dim, (1,), buffer_size)
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.network.to(self.device)
#         self.target_network.to(self.device)
#         self.total_steps = 0

#     @torch.no_grad()
#     def act(self, x, training=True):
#         self.network.train(training)
#         if training and (np.random.rand() < self.epsilon or self.total_steps < self.warmup_steps):
#             action = np.random.randint(0, self.action_dim)
#         else:
#             x = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
#             q_value = self.network(x)
#             action = q_value.max(1)[1].item()
#         return action

#     def learn(self):
#         state, action, reward, next_state, terminated = map(lambda x: x.to(self.device), self.buffer.sample(self.batch_size))
#         q_values = self.network(state)
#         next_q_values = self.target_network(next_state)
#         next_q_value = next_q_values.max(1)[0].unsqueeze(1)
#         td_target = reward + (1 - terminated) * self.gamma * next_q_value
#         q_value = q_values.gather(1, action.long())
#         loss = F.mse_loss(q_value, td_target)

#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()

#         return {"value_loss": loss.item()}

#     def process(self, transition):
#         state, action, reward, next_state, terminated = transition
#         self.total_steps += 1
#         self.buffer.update(state, action, reward, next_state, terminated)

#         result = {}
#         if self.total_steps > self.warmup_steps:
#             result = self.learn()
        
#         if self.total_steps % self.target_update_interval == 0:
#             self.target_network.load_state_dict(self.network.state_dict())
        
#         self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)
#         return result





import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import YOUR_CODE_HERE
import utils

class PacmanActionCNN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PacmanActionCNN, self).__init__()
        self.conv1 = nn.Conv2d(state_dim, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, action_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ReplayBuffer:
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
            torch.LongTensor(self.actions[ind]),
            torch.FloatTensor(self.rewards[ind]),
            torch.FloatTensor(self.next_states[ind]),
            torch.FloatTensor(self.terminated[ind]), 
        )

class DQN:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=0.0005,
        epsilon=0.9,
        epsilon_min=0.05,
        epsilon_decay=1e-5,
        gamma=0.99,
        batch_size=64,
        warmup_steps=1000,
        buffer_size=int(1e5),
        target_update_interval=10000,
    ):
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.target_update_interval = target_update_interval

        self.network = PacmanActionCNN(state_dim[0], action_dim)
        self.target_network = PacmanActionCNN(state_dim[0], action_dim)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = torch.optim.RMSprop(self.network.parameters(), lr)

        self.buffer = ReplayBuffer(state_dim, (1, ), buffer_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network.to(self.device)
        self.target_network.to(self.device)
        
        self.total_steps = 0
        self.epsilon_decay = (epsilon - epsilon_min) / 1e6
    
    @torch.no_grad()
    def act(self, x, training=True):
        self.network.train(training)
        if training and ((np.random.rand() < self.epsilon) or (self.total_steps < self.warmup_steps)):
            action = np.random.randint(0, self.action_dim)
        else:
            x = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
            q_value = self.network(x)
            action = q_value.argmax().item()
        return action
    
    def learn(self):
        state, action, reward, next_state, terminated = map(lambda x: x.to(self.device), self.buffer.sample(self.batch_size))

        q_values = self.network(state).gather(1, action)
        next_q_values = self.target_network(next_state).max(1)[0].unsqueeze(1)
        td_target = reward + self.gamma * next_q_values * (1 - terminated)
        
        loss = F.smooth_l1_loss(q_values, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {"value_loss": loss.item()}
    
    def process(self, transition):
        state, action, reward, next_state, terminated = transition
        self.buffer.update(state, action, reward, next_state, terminated)
        result = {}
        self.total_steps += 1

        if self.total_steps > self.warmup_steps:
            result = self.learn()
            
        if self.total_steps % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.network.state_dict())
        
        self.epsilon = max(self.epsilon - self.epsilon_decay, 0.05)
        return result
