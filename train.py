import torch
import datetime
import os

from robot import Robot
from map import Map
from env import Env
from agent import PPO_Agent
from parameters import TRAIN_EPISODE_NUM, EVAL_INTERVAL, SAVE_MODEL_INTERVAL

class TrainManager():
    def __init__(self,
                 train_env:Env,
                 eval_env:Env,
                 agent:PPO_Agent,
                 episode_num:int=TRAIN_EPISODE_NUM,
                 eval_iters:int=EVAL_INTERVAL,
                 model:str=None
                 ) -> None:
        '''
        初始化训练管理类
        '''
        self.train_env = train_env
        self.eval_env = eval_env
        self.agent = agent
        self.episode_num = episode_num
        self.eval_iters = eval_iters
        self.best_reward = float('-inf')  # 初始化最佳奖励为负无穷
        self.train_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        self.reward_list = []
        if model:
            self.load_model(model)

        
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

        dir_path = 'models/'+self.train_time
        os.makedirs(dir_path, exist_ok=True)
        txt_path = os.path.join(dir_path, 'log.txt')
        with open(txt_path, 'w') as f:
            f.write("Episode\tReward\tSteps\tExplored Rate\n")  # 创建空文件

        i = 0
        while True:
            self.train_episode()
            if (i+1) % EVAL_INTERVAL == 0:
                average_reward, average_step, average_explored_rate = self.eval()
                print(f"Episode {i+1}, Reward: {average_reward:.3f}, Steps: {average_step}, Explored Rate: {average_explored_rate:.3f}")
                with open(txt_path, 'a') as f:
                    f.write(f"{i+1}\t{average_reward:.3f}\t{average_step}\t{average_explored_rate:.3f}\n")
                
                self.reward_list.append(average_reward)
                
                # 保存最佳模型
                # if average_reward > self.best_reward:
                #     self.best_reward = average_reward
                #     self.save_model(average_explored_rate=average_explored_rate,
                #                     model_path=f'models/model_{self.train_time}.pth',
                #                     train_data_path=f'models/train_data_{self.train_time}.json')
                    # print(f"New best model saved! Reward: {self.best_reward:.3f}, Explored Rate: {average_explored_rate:.3f}")
            if (i+1) % SAVE_MODEL_INTERVAL == 0:
                model_path = os.path.join(dir_path, f'model_{i+1}.pth')
                torch.save(self.agent.network.state_dict(), model_path)
                print(f"Model saved to: {model_path}")

            i += 1

    def eval(self)->float:
        average_reward = 0
        average_step = 0
        average_explored_rate = 0
        eval_episode_num = self.eval_env.map.all_map_number
        for _ in range(eval_episode_num):
            average_reward += self.eval_episode()
            average_step += self.eval_env.step_count
            average_explored_rate += self.eval_env.explored_rate

        average_reward /= eval_episode_num
        average_step /= eval_episode_num
        average_explored_rate /= eval_episode_num

        return average_reward, average_step, average_explored_rate

    def eval_episode(self)->None:
        obs = self.eval_env.reset()
        done = False
        test_reward = 0
        while not done:
            action = self.agent.get_action(obs)
            next_obs, reward, done = self.eval_env.step(action)
            obs = next_obs
            test_reward += reward
 
        return test_reward
        
        
    def load_model(self,
                model:str=None
                )->None:
        '''
        加载模型
        '''
        model_dir = 'models'
        model_path = os.path.join(model_dir, model)

        if model_path and os.path.exists(model_path):
            print(f"Loading model from: {model_path}")
            state_dict = torch.load(model_path,map_location=self.agent.device)
            self.agent.network.load_state_dict(state_dict)
            print("Model loaded successfully!")
        elif model_path:
            print(f"Warning: Model file {model_path} not found, starting fresh training")

if __name__ == "__main__":
    robot = Robot()
    train_map = Map('train', 'easy', seed=1)
    eval_map = Map('eval', 'easy')
    train_env = Env(robot, train_map)
    eval_env = Env(robot, train_map)

    # 获取观察空间和动作空间的维度
    obs_dim = (164, 164)  # 获取地图的实际尺寸 (height, width)
    action_dim = 4  # 上下左右
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化智能体
    agent = PPO_Agent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device
    )
    
    # 初始化训练管理器
    train_manager = TrainManager(
        train_env=train_env,
        eval_env=eval_env,
        agent=agent,
        model='20250630_2244/model_600.pth'
    )
    train_manager.train()
    
    