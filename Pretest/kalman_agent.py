import numpy as np
from collections import deque

class KalmanFilter:
    def __init__(self, dt=0.016):
        self.dt = dt
        # 状态向量: [x, y, vx, vy]
        self.x = np.zeros(4)
        
        # 状态协方差矩阵 P
        self.P = np.eye(4) * 100.0
        
        # 状态转移矩阵 F (恒定速度模型)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # 观测矩阵 H (只观测位置)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # 观测噪声协方差 R (传感器噪声)
        self.R = np.eye(2) * 5.0 
        
        # 过程噪声协方差 Q (模型不确定性)
        self.Q = np.eye(4) * 0.1

    def process(self, measurement):
        """
        执行一次卡尔曼滤波迭代：预测 + 更新
        """
        # 1. 预测 (Predict)
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # 2. 更新 (Update)
        z = np.array(measurement)
        y = z - self.H @ self.x  # 创新 (Innovation)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)  # 卡尔曼增益
        
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        
        return self.x[:2] # 返回平滑后的位置

class KalmanAgent:
    def __init__(self):
        # 初始化卡尔曼滤波器
        self.kf = KalmanFilter(dt=0.016) # 假设约60Hz采样
        
        # 延迟缓冲区: 存储 (timestamp, x, y)
        self.history_buffer = deque()
        
        # 动作缓冲区: 存储 (timestamp, dx, dy)
        self.action_buffer = deque()
        
        self.DELAY_SEC = 0.070  # 70ms 延迟
        
        # PI控制参数 (与普通Agent保持一致以便比较)
        self.Kp = 0.3
        self.Ki = 2.0
        
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.max_integral_contribution = 30.0 
        
        self.last_obs_time = None

    def get_delayed_observation(self, current_time, real_x, real_y):
        """
        模拟系统延迟，返回70ms前的观测值
        """
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
            
        while len(self.history_buffer) > 0 and self.history_buffer[0][0] < current_time - 0.2:
            self.history_buffer.popleft()
            
        return best_match

    def step(self, current_time, real_x, real_y, center_x, center_y):
        # 1. 获取延迟观测值
        obs_time, obs_x, obs_y = self.get_delayed_observation(current_time, real_x, real_y)
        
        # 2. 卡尔曼滤波处理
        # 动态调整 dt (如果时间间隔变化较大)
        if self.last_obs_time is not None:
            dt = obs_time - self.last_obs_time
            if dt > 0:
                self.kf.F[0, 2] = dt
                self.kf.F[1, 3] = dt
        
        # 获取平滑后的位置估计 (在 obs_time 时刻)
        smoothed_pos = self.kf.process([obs_x, obs_y])
        base_est_x, base_est_y = smoothed_pos[0], smoothed_pos[1]
        
        self.last_obs_time = obs_time
        
        # 3. Smith Predictor 动作补偿
        pending_dx = 0
        pending_dy = 0
        
        while len(self.action_buffer) > 0 and self.action_buffer[0][0] < obs_time:
            self.action_buffer.popleft()
            
        for t, dx, dy in self.action_buffer:
            if t > obs_time:
                pending_dx += dx
                pending_dy += dy
        
        # 4. 最终状态估计
        est_x = base_est_x + pending_dx
        est_y = base_est_y + pending_dy
        
        # 5. 计算误差与控制
        error_x = center_x - est_x
        error_y = center_y - est_y
        
        dt = 0.01
        self.integral_x += error_x * dt
        self.integral_y += error_y * dt
        
        limit_val = self.max_integral_contribution / self.Ki if self.Ki > 0 else 0
        self.integral_x = np.clip(self.integral_x, -limit_val, limit_val)
        self.integral_y = np.clip(self.integral_y, -limit_val, limit_val)
        
        if abs(error_x) < 10: 
            error_x = 0
            self.integral_x = 0
        if abs(error_y) < 10: 
            error_y = 0
            self.integral_y = 0
            
        move_x = error_x * self.Kp + self.integral_x * self.Ki
        move_y = error_y * self.Kp + self.integral_y * self.Ki
        
        max_step = 50
        move_x = np.clip(move_x, -max_step, max_step)
        move_y = np.clip(move_y, -max_step, max_step)
        
        int_move_x = int(move_x)
        int_move_y = int(move_y)
        
        if abs(int_move_x) > 0 or abs(int_move_y) > 0:
            self.action_buffer.append((current_time, int_move_x, int_move_y))
            
        return int_move_x, int_move_y
