import numpy as np
from typing import Tuple
from skimage import io


class Map:
    '''
    地图类
    '''
    def __init__(self,
                 map_path:str,
                 ) -> None:
        '''
        初始化地图

        属性：
        map_path: str
            地图文件路径

        global_map: np.ndarray
            全局地图(已经二值化, 仅有可通行(255)/不可通行(1)两种状态)

        robot_start_position: np.ndarray
            机器人初始位置
        '''
        self.map_path = map_path
        self.global_map, self.robot_start_position = self.map_setup()
        self.all_passable_area = self.calculate_all_passable_area()


    def map_setup(self)->Tuple[np.ndarray, np.ndarray]:
        '''
        将地图二值化，并返回机器人初始位置(黄点中心)
        
        参数:
        map_path: 地图路径
        
        返回:
        global_map: 全局地图
        robot_position: 机器人位置
        '''
        # 第二个参数 1 表示以灰度模式读取图像
        # 图像数据通常加载为 0-1 的浮点数，所以这里乘以 255，将图像数据转换为 0-255 的整数范围
        # astype(int) 将图像矩阵的数据类型转换为整型
        global_map = (io.imread(self.map_path, 1) * 255).astype(int)

        # 地图中黄点位置 
        robot_location = np.nonzero(global_map == 208)

        robot_location = np.array([np.array(robot_location)[1, 127], 
                                    np.array(robot_location)[0, 127]])
        
        global_map = (global_map > 150)
        global_map = global_map * 254 + 1   
        
        return global_map, robot_location
    
    def calculate_all_passable_area(self)->np.ndarray:
        '''
        计算所有可通行区域
        '''
        return np.sum(self.global_map == 255)
