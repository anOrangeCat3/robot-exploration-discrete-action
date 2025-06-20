import numpy as np
from skimage.transform import resize
from typing import Tuple

from parameters import EXPLORATION_RATE_THRESHOLD,EXPLORATION_MAX_STEP
from robot import Robot
from map import Map

class Env:
    '''
    环境类
    用于管理机器人和地图的交互

    包含：
    map: 地图类
    robot: 机器人类
    '''
    def __init__(self,
                 robot:Robot,
                 map:Map,
                 ) -> None:
        '''
        初始化环境

        属性：
        robot: Robot
            机器人
        map: Map
            地图
        '''
        self.robot = robot
        self.map = map
        self.explored_rate = 0
        self.explored_area = 0
        self.step_count = 0
        self.reward_step = 0
        

    def reset(self,)->np.ndarray:
        '''
        重置环境

        返回：
        obs: np.ndarray
            机器人自己的地图
        '''
        self.explored_rate = 0
        self.explored_area = 0
        self.step_count = 0
        self.reward_step = 0
        # 选择地图
        global_map=self.map.reset()

        # 更新机器人自己的地图
        robot_belief_map = self.robot.reset(self.map.robot_start_position, global_map)
        # 更新探索率
        self.update_explored_area_rate()
        # 加上机器人自己的位置
        obs= self.mark_robot_position(robot_belief_map)
        # TODO: resize
        obs = resize(obs, (192, 256))

        return obs
    
    def step(self,
             action:float
             )->Tuple[np.ndarray, float, bool]:
        '''
        执行一步动作

        参数:
        action: 上下左右
            
        返回:
        obs: np.ndarray 认知地图
        reward: float
        done: bool
        '''
        # 更新步数
        self.step_count += 1
        # 机器人移动
        self.robot.move(action)
        # 更新机器人自己的belief_map
        robot_belief_map = self.robot.update_belief_map(self.map.global_map)
        # 加上机器人自己的位置
        obs = self.mark_robot_position(robot_belief_map)
        # TODO: resize
        obs = resize(obs, (192, 256))
        # TODO: 设计奖励
        reward,done = self.calculate_reward()
        
        return obs,reward,done
    
    def calculate_terminated_truncated(self,)->bool:
        '''判断是否结束'''
        # 根据探索率判断是否结束
        terminated = self.explored_rate >= EXPLORATION_RATE_THRESHOLD
        # 达到步数上限结束
        truncated = self.step_count >= EXPLORATION_MAX_STEP

        return terminated,truncated
        

    def calculate_reward(self)->float:
        '''计算奖励'''
        # 更新探索率
        explored_area_change,explored_rate_change=self.update_explored_area_rate()
        # print(f"explored_area_change: {explored_area_change}")
        # 判断是否结束
        terminated,truncated = self.calculate_terminated_truncated()
        done = terminated or truncated
        reward = 0
        # print(f"explored_rate_change: {explored_rate_change},explored_area_change: {explored_area_change}")
        
        # 1. 探索奖励 - 增大奖励信号
        if explored_area_change > 0:
            reward += explored_rate_change*30
            self.reward_step = self.step_count
        # 2. 惩罚长时间不探索新的区域
        no_exploration_steps = self.step_count - self.reward_step
        if no_exploration_steps > 10:
            reward -= ((no_exploration_steps-10)//10)*0.05
        # 3. 完成奖励
        if terminated:
            efficiency_bonus = (1.0 - self.step_count / EXPLORATION_MAX_STEP) * 50
            reward += efficiency_bonus+50
        
        # 4. 运动惩罚 - 轻微惩罚每步，鼓励高效探索
        reward -= 0.01
        # print(f"explored_rate_change: {explored_rate_change}, reward: {reward}")
        return reward,done


    def mark_robot_position(self, robot_belief_map):
        height, width = robot_belief_map.shape

        # 假设 self.robot.position 是 (x, y)
        pos_x, pos_y = self.robot.position
        # 因为numpy默认的坐标原点和图像的坐标原点不一样，所以需要转换
        center_y = int(pos_y)
        center_x = int(pos_x)
        center_y = np.clip(center_y, 0, height-1)
        center_x = np.clip(center_x, 0, width-1)

        y_coords, x_coords = np.ogrid[:height, :width]
        distances = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
        circle_mask = distances <= self.robot.radius
        robot_belief_map[circle_mask] = 1

        return robot_belief_map
    

    def update_explored_area_rate(self)->Tuple[float,float]:
        '''
        更新探索面积
        '''
        old_explored_area = self.explored_area
        old_explored_rate = self.explored_rate
        self.explored_area = self.robot.explored_area
        explored_area_change = self.explored_area - old_explored_area
        self.explored_rate = self.explored_area / self.map.all_passable_area
        explored_rate_change = self.explored_rate - old_explored_rate
        
        return explored_area_change,explored_rate_change
