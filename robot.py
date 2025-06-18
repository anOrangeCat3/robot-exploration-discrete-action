import numpy as np
from typing import Tuple

from parameters import ROBOT_RADIUS, STEP_SIZE
from sensor import Lidar


class Robot:
    '''
    Robot类
    主要用于和地图(environment)交互
    最好和agent分开, 方便训练
    即: agent负责决策(训练), robot负责和环境交互

    包含：
    position: 机器人自身位置
    belief_map: 机器人认知地图, 确认地图上的座标点是否已经探索过 
    lidar: 传感器, sensor类

    功能：(和地图交互)
    移动
    扫描
    更新自身认知地图
    '''
    def __init__(self,
                 ) -> None:
        '''
        初始化机器人

        属性：
        position: np.ndarray
            机器人位置
        lidar: Lidar
            传感器
        belief_map: np.ndarray
            机器人自己的地图
        position_history: np.ndarray
            机器人位置历史
        '''
        self.radius = ROBOT_RADIUS
        self.position = None
        self.belief_map = None
        self.lidar = Lidar()
        self.position_history = np.zeros((0, 2))  # 初始化为空数组，但保持二维结构
        self.explored_area = 0

    def reset(self,
              robot_start_position:np.ndarray,
              global_map:np.ndarray
              )->np.ndarray:
        '''
        每轮游戏开始前重置机器人

        参数:
        robot_start_position: np.ndarray
            机器人初始位置

        global_map: np.ndarray
            本次游戏全局地图

        返回：
        belief_map: np.ndarray
            机器人自己的地图
        '''
        self.position = robot_start_position  # 初始化机器人位置
        self.belief_map=np.ones_like(global_map) * 127  # 初始化机器人自己的地图, 127为未知区域
        self.position_history = robot_start_position  # 初始化为空数组  
        self.explored_area = 0

        # 更新机器人自己的地图
        self.update_belief_map(global_map)

        return self.belief_map.copy()
    

    def update_belief_map(self,
                         global_map:np.ndarray
                         ):
        '''
        更新机器人自己的地图

        参数:
        global_map: np.ndarray
            全局地图

        返回：
        belief_map: np.ndarray
            机器人自己认知的地图
        '''
        # lidar扫描, 获取新信息, 更新belief_map
        self.belief_map=self.lidar.scan(self.position,self.belief_map.copy(),global_map)
        # 计算已经探索区域
        self.explored_area = self.calculate_explored_area()

        return self.belief_map.copy()
    
    def move(self,
             action:int
             )->None:
        '''
        移动机器人

        Args:
            action: int
                0: right
                1: down 
                2: left 
                3: up
        '''
        if action == 0:
            new_position = self.position + np.array([STEP_SIZE, 0])
        elif action == 1:
            new_position = self.position + np.array([0, STEP_SIZE])
        elif action == 2:
            new_position = self.position + np.array([-STEP_SIZE, 0])
        elif action == 3:
            new_position = self.position + np.array([0, -STEP_SIZE])

        # 考虑机器人自身的半径, 不要撞到障碍物
        end_position = self.find_nearest_boundary(new_position)
        # 移动到本次移动的终点
        self.position = end_position
        self.position_history = np.vstack((self.position_history, end_position))

    def find_safe_position_in_known_area(self, position: np.ndarray) -> np.ndarray:
        """
        在已知区域内找到一个安全的位置，考虑机器人半径
        
        Args:
            position: np.ndarray
                目标位置
        Returns:
            safe_position: np.ndarray
                安全的位置
        """
        x, y = position
        
        # 检查目标位置是否安全（考虑机器人半径）
        if self.is_position_safe(x, y):
            return position
        
        # 如果目标位置不安全，在周围寻找最近的安全位置
        h, w = self.belief_map.shape
        
        # 搜索半径，从小到大逐渐扩大
        for search_radius in range(1, self.radius * 2):
            for dx in range(-search_radius, search_radius + 1):
                for dy in range(-search_radius, search_radius + 1):
                    new_x, new_y = x + dx, y + dy
                    
                    # 检查是否在地图范围内
                    if 0 <= new_x < w and 0 <= new_y < h:
                        # 检查是否在已知区域且安全
                        if (self.belief_map[new_y, new_x] == 255 and 
                            self.is_position_safe(new_x, new_y)):
                            return np.array([new_x, new_y])
        
        # 如果找不到安全位置，返回当前位置
        return self.position

    def is_position_safe(self, x: int, y: int) -> bool:
        """
        检查位置是否安全（考虑机器人半径）
        
        参数:
        x, y: int
            要检查的位置坐标
        
        返回:
        is_safe: bool
            位置是否安全
        """
        h, w = self.belief_map.shape
        
        # 检查机器人半径范围内是否有障碍物
        for dx in range(-self.radius, self.radius + 1):
            for dy in range(-self.radius, self.radius + 1):
                # 只检查圆形范围内的点
                if dx*dx + dy*dy <= self.radius*self.radius:
                    check_x, check_y = x + dx, y + dy
                    
                    # 检查是否在地图范围内
                    if 0 <= check_x < w and 0 <= check_y < h:
                        # 如果发现障碍物（值为1）或未知区域（值为127），则不安全
                        if self.belief_map[check_y, check_x] in [1, 127]:
                            return False
                    else:
                        # 超出地图边界也认为不安全
                        return False
        
        return True

    def find_nearest_boundary(self, position: np.ndarray) -> np.ndarray:
        """
        Bresenham算法思想
        找到最近的边界点，考虑机器人自身的半径，且只在雷达范围内检查障碍物

        参数:
        position: np.ndarray
            当前位置

        返回:
        last_safe: np.ndarray
            最近的边界点(考虑到机器人自身半径, 不会与墙壁碰撞)
        """
        x0, y0 = self.position
        x1, y1 = position

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        error = dx - dy
        x, y = x0, y0

        def is_safe(px, py):
            h, w = self.belief_map.shape
            r = self.lidar.lidar_range  # 只检查雷达范围内

            y_min = max(0, py - r)
            y_max = min(h, py + r + 1)
            x_min = max(0, px - r)
            x_max = min(w, px + r + 1)

            yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
            mask = (xx - px) ** 2 + (yy - py) ** 2 <= self.radius ** 2  # 机器人本体半径
            local_map = self.belief_map[y_min:y_max, x_min:x_max]
            return not np.any(local_map[mask] == 1)

        last_safe = np.array([x, y])

        while True:
            if 0 <= x < self.belief_map.shape[1] and 0 <= y < self.belief_map.shape[0]:
                if is_safe(x, y):
                    last_safe = np.array([x, y])
                else:
                    return last_safe
            else:
                return last_safe

            if x == x1 and y == y1:
                return last_safe

            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx

    def calculate_explored_area(self,
                               )->np.ndarray:
        '''
        计算已经探索区域

        return:
        explored_area: np.ndarray
            已经探索区域面积
        '''
        return np.sum(self.belief_map == 255)
        
        
