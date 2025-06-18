import numpy as np

from parameters import LIDAR_RANGE, LIDAR_ANGLE_STEP

class Lidar:
    '''
    Lidar类

    包含：
    激光雷达的扫描范围
    激光雷达的扫描角度增量(扫描步长)

    功能：
    扫描地图, 与地图交互
    '''
    def __init__(self,
                 )->None:
        self.lidar_range = LIDAR_RANGE
        self.angel_step = LIDAR_ANGLE_STEP

    def scan(self,
             robot_position:np.ndarray,
             robot_belief_map:np.ndarray,
             ground_truth:np.ndarray):
        '''
        以机器人坐标为中心, 扫描一圈
        '''
        x0 = robot_position[0]
        y0 = robot_position[1]
        angle = 0
        while angle < 2 * np.pi:
            x1 = x0+np.cos(angle)*self.lidar_range
            y1 = y0+np.sin(angle)*self.lidar_range

            robot_belief_map=self.boundary_check(x0,y0,x1,y1,ground_truth,robot_belief_map)
            angle += self.angel_step
        
        return robot_belief_map
    
    def boundary_check(self,
                       x0:np.ndarray, 
                       y0:np.ndarray, 
                       x1:np.ndarray, 
                       y1:np.ndarray, 
                       ground_truth:np.ndarray, 
                       robot_belief_map:np.ndarray)->np.ndarray:
        """ Checks if line is blocked by obstacle """
        x0 = x0.round()
        y0 = y0.round()
        x1 = x1.round()
        y1 = y1.round()
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        x, y = x0, y0
        error = dx - dy
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        dx *= 2
        dy *= 2

        collision_flag = 0
        max_collision = 10

        while 0 <= x < ground_truth.shape[1] and 0 <= y < ground_truth.shape[0]:
            k = ground_truth[y, x]
            if k == 1 and collision_flag < max_collision:
                collision_flag += 1
                if collision_flag >= max_collision:
                    break

            if k !=1 and collision_flag > 0:
                break

            if x == x1 and y == y1:
                break

            robot_belief_map[y, x] = k

            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx

        return robot_belief_map


