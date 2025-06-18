import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple

from network import PPO_Network
from parameters import LEARNING_RATE, GAMMA, BATCH_SIZE, TRAIN_EPOCHS, ADVANTAGE_LAMBDA, CLIP_EPSILON


class Episode_Recorder(): 
    def __init__(self,
                 device:torch.device) -> None:
        self.device = device
        self.reset()

    def reset(self):
        """ Clear the trajectory when begin a new episode."""
        self.trajectory = {
            "obs": torch.tensor([], dtype = torch.float32).to(self.device),
            "action": torch.tensor([], dtype = torch.int64).to(self.device),  
            "reward": torch.tensor([], dtype = torch.float32).to(self.device),
            "next_obs": torch.tensor([], dtype = torch.float32).to(self.device),
            "done": torch.tensor([], dtype = torch.float32).to(self.device)
        }
    
    def append(self,
               obs:np.ndarray,
               action:np.ndarray,
               reward:float,
               next_obs:np.ndarray,
               done:bool):
        '''
        Append one step to the trajectory of current episode.
        '''
        # 确保观察值是2D的 [H, W]
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)  # [1, H, W]
        action = torch.tensor([action], dtype=torch.int64).unsqueeze(0).to(self.device)  # [1, action_dim=1]
        reward = torch.tensor([[reward]], dtype=torch.float32).to(self.device)  # [1, 1]
        next_obs = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(self.device)  # [1, H, W]
        done = torch.tensor([[done]], dtype=torch.float32).to(self.device)  # [1, 1]
        
        self.trajectory["obs"] = torch.cat((self.trajectory["obs"], obs))
        self.trajectory["action"] = torch.cat((self.trajectory["action"], action))
        self.trajectory["reward"] = torch.cat((self.trajectory["reward"], reward))
        self.trajectory["next_obs"] = torch.cat((self.trajectory["next_obs"], next_obs))
        self.trajectory["done"] = torch.cat((self.trajectory["done"], done))
        
    def get_trajectory(self
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Get the trajectory of current episode.
        '''
        return self.trajectory["obs"], self.trajectory["action"], \
            self.trajectory["reward"], self.trajectory["next_obs"], self.trajectory["done"]
        

class PPO_Agent():
    def __init__(self,
                 obs_dim:Tuple[int, int],
                 action_dim:int,
                 device:torch.device
                 ) -> None:
        self.network = PPO_Network(obs_dim=obs_dim, action_dim=action_dim)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LEARNING_RATE)
        self.episode_recorder = Episode_Recorder(device)
        self.gamma = GAMMA
        self.batch_size = BATCH_SIZE
        self.train_epoches = TRAIN_EPOCHS
        self.advantage_lambda = ADVANTAGE_LAMBDA
        self.clip_epsilon = CLIP_EPSILON
        self.device = device

        print(f"obs_dim: {obs_dim}, device: {device}")
    
    def get_action(self,
                   obs:np.ndarray) -> Tuple[torch.tensor, torch.tensor]:
        '''
        Get the action from the network.

        return:
            action: 0, 1, 2, 3
        '''
        obs = torch.tensor(obs, dtype = torch.float32).to(self.device)  # shape: (1, obs_dim)
        dist = self.network.pi(obs)  # shape: (1, action_dim)
        dist = F.softmax(dist, dim = -1) 
        action_probs = torch.distributions.Categorical(dist)
        picked_action = action_probs.sample()

        return picked_action.item()
        
    
    def train(self)->None:
        '''
        Train the network.
        '''
        obs, action, reward, next_obs, done = self.episode_recorder.get_trajectory()
        # obs: (batch_size, H, W)
        # action: (batch_size, action_dim), 
        # reward: (batch_size, 1), 
        # next_obs: (batch_size, H, W), 
        # done: (batch_size, 1)
        # print(f"obs.shape: {obs.shape}")
        # print(f"action.shape: {action.shape}")
        # 1. TD_target = r + gamma * V(s')
        with torch.no_grad():  # 不需要计算梯度
            TD_target = reward + self.gamma * self.network.v(next_obs) * (1 - done)
            TD_error = TD_target - self.network.v(obs)

        # check_nan(TD_error, "TD_error")
        # print(f"TD_error: {TD_error}")

        # 2. Advantages GAE
        advantage = self.compute_advantage(TD_error)
        # print(f"advantage: {advantage}")

        # 3. old_log_prob
        with torch.no_grad():
            old_log_prob = self.calculate_log_prob(obs, action).detach()

        # 确保 batch_size 不超过数据量
        batch_size = min(self.batch_size, obs.shape[0])

        # 4. update the network by batch
        for _ in range(self.train_epoches):
            sample_indices = np.random.randint(low=0, 
                                               high=obs.shape[0], 
                                               size=batch_size)
            # print(f"obs[sample_indices]: {obs[sample_indices].shape}")
            critic_loss = torch.mean(F.mse_loss(TD_target[sample_indices].detach(), 
                                                self.network.v(obs[sample_indices])))
            # check_nan(critic_loss, "critic_loss")
            log_prob = self.calculate_log_prob(obs[sample_indices], action[sample_indices])
            ratio = torch.exp(log_prob - old_log_prob[sample_indices])  # pi_theta/pi_theta_old
            
            # check_nan(ratio, "ratio")
            surr1 = ratio * advantage[sample_indices]
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage[sample_indices]
            actor_loss = -torch.mean(torch.min(surr1, surr2))

            total_loss = actor_loss + critic_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(),1.0)
            self.optimizer.step()


    def compute_advantage(self, TD_error:torch.tensor) -> torch.tensor:
        '''
        使用GAE计算优势函数
        Args:
            TD_error: torch.tensor, shape (batch_size, 1)
                TD误差
        Returns:
            advantage: torch.tensor, shape (batch_size, 1)
                计算得到的优势函数值
        '''
        # 初始化优势函数列表
        advantage_list = []
        advantage = torch.tensor(0.0, device=self.device)
        
        # 从后向前计算GAE
        for delta in TD_error.flip(0):  # 使用flip代替[::-1]
            # GAE计算公式：A_t = delta_t + gamma * lambda * A_{t+1}
            advantage = self.gamma * self.advantage_lambda * advantage + delta
            advantage_list.append(advantage)
        
        # 反转列表，恢复原始顺序
        advantage_list.reverse()
        
        # 将列表转换为tensor
        advantage = torch.cat(advantage_list)
        
        return advantage
        
    def calculate_log_prob(self,
                           obs:torch.tensor,
                           action:torch.tensor) -> torch.tensor:
        '''
        计算动作的对数概率
        '''
        dist = F.softmax(self.network.pi(obs), dim=-1)
        log_prob = torch.log(dist.gather(-1, action))
        # print(f"dist: {dist.shape}")
        # print(f"action: {action.shape}")
        
        
        return log_prob
    
               
    
        
    

