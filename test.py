from robot import Robot
from map import Map
from env import Env
from agent import PPO_Agent

robot = Robot()
map = Map('eval', 'easy')
env = Env(robot, map)

import torch
import matplotlib.pyplot as plt
import os

# 获取观察空间和动作空间的维度
obs_dim = (192, 256)  # 获取地图的实际尺寸 (height, width)
action_dim = 4  # 角度和距离
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化智能体
agent = PPO_Agent(
    obs_dim=obs_dim,
    action_dim=action_dim, 
    device=device
)

# 加载最新的模型文件
model_dir = 'models'
model_path = os.path.join(model_dir, 'model_20250619_2137.pth')
print(model_path)
agent.network.load_state_dict(torch.load(model_path))
print(f"Loaded model from {model_path}")

# 运行一局游戏并可视化
obs = env.reset()
total_reward=0
done = False

plt.ion()  # 开启交互模式
fig = plt.figure()

while not done:
    # 显示当前状态
    plt.clf()
    plt.imshow(obs, cmap='gray')
    plt.title(f'step: {env.step_count}\n explored rate: {env.explored_rate:.3f}\n total reward: {total_reward}')
    plt.pause(0.03)
    
    # 选择动作
    action = agent.get_action(obs)
    # 执行动作
    obs, reward, done = env.step(action)
    total_reward += reward

plt.ioff()
plt.show()