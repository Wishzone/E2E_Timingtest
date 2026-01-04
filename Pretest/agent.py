import time
import numpy as np
from collections import deque
from pre import Predictor

class Agent:
    def __init__(self):
        # 初始化预测器 (X和Y轴分开)
        self.pred_x = Predictor()
        self.pred_y = Predictor()
        
        # 延迟缓冲区: 存储 (timestamp, x, y)
        self.history_buffer = deque()
        
        # 动作缓冲区: 存储 (timestamp, dx, dy)
        self.action_buffer = deque()
        
        self.DELAY_SEC = 0.070  # 70ms 延迟
        
        # PI控制参数
        self.Kp = 0.3
        self.Ki = 2.0
        
        # 积分误差累积
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.max_integral_contribution = 30.0 

    def get_delayed_observation(self, current_time, real_x, real_y):
        """
        模拟系统延迟，返回70ms前的观测值
        """
        # 存入当前真实值
        self.history_buffer.append((current_time, real_x, real_y))
        
        target_time = current_time - self.DELAY_SEC
        
        best_match = None
        for item in self.history_buffer:
            t, x, y = item
            if t >= target_time:
                best_match = item
                break
        
        if best_match is None and len(self.history_buffer) > 0:
            best_match = self.history_buffer[-1]
        elif best_match is None:
            best_match = (current_time, real_x, real_y)
            
        # 清理过旧数据
        while len(self.history_buffer) > 0 and self.history_buffer[0][0] < current_time - 0.2:
            self.history_buffer.popleft()
            
        return best_match

    def step(self, current_time, real_x, real_y, center_x, center_y):
        """
        执行一步控制
        返回: (move_x, move_y)
        """
        # 1. 获取延迟观测值
        obs_time, obs_x, obs_y = self.get_delayed_observation(current_time, real_x, real_y)
        
        # 2. 预测器更新 (仅平滑)
        self.pred_x.update(obs_time, obs_x)
        self.pred_y.update(obs_time, obs_y)
        
        # 3. 基础位置估计 (基于延迟观测)
        base_est_x = self.pred_x.predict(obs_time)
        base_est_y = self.pred_y.predict(obs_time)
        
        # 4. Smith Predictor 动作补偿
        pending_dx = 0
        pending_dy = 0
        
        # 清理过期动作
        while len(self.action_buffer) > 0 and self.action_buffer[0][0] < obs_time:
            self.action_buffer.popleft()
            
        # 累加未观测到的动作
        for t, dx, dy in self.action_buffer:
            if t > obs_time:
                pending_dx += dx
                pending_dy += dy
        
        # 5. 最终状态估计
        est_x = base_est_x + pending_dx
        est_y = base_est_y + pending_dy
        
        # 6. 计算误差
        error_x = center_x - est_x
        error_y = center_y - est_y
        
        # 7. 积分项更新
        dt = 0.01 # 假设步长
        self.integral_x += error_x * dt
        self.integral_y += error_y * dt
        
        # 抗饱和
        limit_val = self.max_integral_contribution / self.Ki if self.Ki > 0 else 0
        self.integral_x = np.clip(self.integral_x, -limit_val, limit_val)
        self.integral_y = np.clip(self.integral_y, -limit_val, limit_val)
        
        # 死区与积分重置
        if abs(error_x) < 10: 
            error_x = 0
            self.integral_x = 0
        if abs(error_y) < 10: 
            error_y = 0
            self.integral_y = 0
            
        # 8. 计算控制输出
        move_x = error_x * self.Kp + self.integral_x * self.Ki
        move_y = error_y * self.Kp + self.integral_y * self.Ki
        
        # 幅度限制
        max_step = 50
        move_x = np.clip(move_x, -max_step, max_step)
        move_y = np.clip(move_y, -max_step, max_step)
        
        # 9. 整数化并记录
        int_move_x = int(move_x)
        int_move_y = int(move_y)
        
        if abs(int_move_x) > 0 or abs(int_move_y) > 0:
            self.action_buffer.append((current_time, int_move_x, int_move_y))
            
        return int_move_x, int_move_y
