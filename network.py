import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

class PPO_Network(nn.Module):
    def __init__(self,
                 obs_dim:Tuple[int, int],  # 用于计算卷积后的特征图大小
                 action_dim:int,          # 动作空间大小
                 num_inputs:int=1,        # 灰度图，单通道
                 device:torch.device=torch.device("cuda"),  # 默认使用cuda
                 ) -> None:
        super(PPO_Network, self).__init__()
        
        # 特征提取网络
        self.conv_layers = nn.Sequential(
            # 第一层：使用较小kernel保留细节
            nn.Conv2d(num_inputs, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 第二层：继续提取特征
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 第三层：深层特征
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 第四层：抽象特征
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        conv_output_size = self.calculate_conv_output_size(obs_dim)
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        
        # 策略网络输出层（Actor）
        self.fc_pi = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        # 价值网络输出层（Critic）
        self.fc_v = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # 将网络移动到指定设备
        self.to(device)
    
    
    def _extract_features(self, x):
        """
        提取特征
        Args:
            x: 输入状态，shape (batch_size, channels, height, width)
        Returns:
            features: 提取的特征，shape (batch_size, 512)
        """
        # 打印输入维度
        # print(f"网络输入维度: {x.shape}")
        if len(x.shape) == 2: 
            # get_action时，输入维度为(H, W)
            x = x.unsqueeze(0).unsqueeze(0)
        elif len(x.shape) == 3:
            # train时，输入维度为(batch_size, H, W)
            x = x.unsqueeze(1)
        
        # 通过卷积层
        x = self.conv_layers(x)
        # 展平
        x = x.view(x.size(0), -1)
        # 通过全连接层
        x = self.fc(x)
        
        return x
    
    def pi(self, x):
        """
        策略网络(Actor)
        Args:
            x: 输入状态
        Returns:
            action_prob: 动作概率
        """
        features = self._extract_features(x)
        x= self.fc_pi(features)
        
        return x
    
    def v(self, x):
        """
        价值网络(Critic)
        Args:
            x: 输入状态
        Returns:
            value: 状态价值
        """
        # print(f"价值网络输入维度: {x.shape}")
        features = self._extract_features(x)
        value = self.fc_v(features)
        # print(f"状态价值维度: {value.shape}")
        return value
    
    def calculate_conv_output_size(self, obs_dim: Tuple[int, int]) -> int:
        """
        计算卷积层输出后的特征图大小
        Args:
            obs_dim: 输入图像的维度 (height, width)
        Returns:
            int: 全连接层的输入维度
        """
        height, width = obs_dim
        # print(height, width)
        
        # 第一层卷积: kernel_size=5, stride=2, padding=2
        # output_size = (input_size + 2*padding - kernel_size) / stride + 1
        height = (height + 2*2 - 5) // 2 + 1
        width = (width + 2*2 - 5) // 2 + 1
        
        # 第二层卷积: kernel_size=3, stride=2, padding=1
        height = (height + 2*1 - 3) // 2 + 1
        width = (width + 2*1 - 3) // 2 + 1
        
        # 第三层卷积: kernel_size=3, stride=2, padding=1
        height = (height + 2*1 - 3) // 2 + 1
        width = (width + 2*1 - 3) // 2 + 1
        
        # 第四层卷积: kernel_size=3, stride=2, padding=1
        height = (height + 2*1 - 3) // 2 + 1
        width = (width + 2*1 - 3) // 2 + 1
        
        # 最后一层有256个输出通道
        total_features = height * width * 256
            
        return total_features
