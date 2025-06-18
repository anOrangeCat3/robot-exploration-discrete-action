import copy
import torch
import datetime

from robot import Robot
from map import Map
from env import Env
from agent import PPO_Agent
from parameters import TRAIN_EPISODE_NUM, EVAL_INTERVAL

class TrainManager():
    def __init__(self,
                 env:Env,
                 agent:PPO_Agent,
                 episode_num:int=TRAIN_EPISODE_NUM,
                 eval_iters:int=EVAL_INTERVAL,
                 ) -> None:
        '''
        初始化训练管理类

        属性：
        train_env: Env
            训练环境
            
        episode_num: int
            训练轮数
        eval_iters: int
            评估间隔
        '''
        self.train_env = env
        self.eval_env = copy.deepcopy(env)
        self.agent = agent
        self.episode_num = episode_num
        self.eval_iters = eval_iters
        self.best_reward = float('-inf')  # 初始化最佳奖励为负无穷
        self.train_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.reward_list = []
    
    def train_episode(self)->None:
        '''
        一轮游戏
        '''
        # 清空episode_recorder记录的轨迹
        self.agent.episode_recorder.reset()
        # 重置环境
        obs = self.train_env.reset()
        done = False
        while not done:
            # 选择动作
            action = self.agent.get_action(obs)
            # 执行动作
            next_obs, reward, done = self.train_env.step(action)
            # 记录轨迹
            self.agent.episode_recorder.append(obs, action, reward, next_obs, done)
            # 更新状态
            obs = next_obs
        # 训练
        self.agent.train()

    def train(self)->None:
        for i in range(self.episode_num):
            self.train_episode()
            if i % EVAL_INTERVAL == 0:
                eval_reward = self.eval()
                print(f"Episode {i}, Eval Reward: {eval_reward}, Steps: {self.eval_env.step_count}")
                self.reward_list.append(eval_reward)
        
        return self.reward_list

    def eval(self)->None:
        obs = self.eval_env.reset()
        done = False
        test_reward = 0
        while not done:
            action = self.agent.get_action(obs)
            next_obs, reward, done = self.eval_env.step(action)
            obs = next_obs
            test_reward += reward

        # 保存最佳模型
        if test_reward > self.best_reward:
            self.best_reward = test_reward
            # 保存网络参数
            torch.save(self.agent.network.state_dict(), f'models/model_{self.train_time}.pth')
            print(f"New best model saved! Reward: {self.best_reward:.2f}")
            
        return test_reward
        

if __name__ == "__main__":
    robot = Robot()
    map = Map('maps/map1.png')
    env = Env(robot, map)

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
    
    # 初始化训练管理器
    train_manager = TrainManager(env, agent)
    reward_list = train_manager.train()
    print(reward_list)