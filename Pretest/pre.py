import numpy as np

class Predictor:
    def __init__(self):
        """
        初始化预测器
        """
        self.history_time = []
        self.history_pos = []
        self.window_size = 6  # 使用最近6个点进行拟合，增加一点平滑度

    def update(self, timestamp, pos):
        """
        更新观测数据
        timestamp: 观测数据对应的时间戳（注意这是延迟后的时间）
        pos: 观测到的位置
        """
        self.history_time.append(timestamp)
        self.history_pos.append(pos)
        
        # 保持窗口大小
        if len(self.history_time) > self.window_size:
            self.history_time.pop(0)
            self.history_pos.pop(0)

    def predict(self, target_time):
        """
        预测目标时间的位置
        target_time: 我们想要预测的时间点（通常是当前系统时间）
        """
        if len(self.history_time) < 2:
            return self.history_pos[-1] if self.history_pos else 0.0
        
        # 使用线性回归拟合最近的点来估计速度和位置
        # y = kx + b
        # 我们将时间归一化以避免大数值计算问题
        t_base = self.history_time[0]
        t_local = np.array(self.history_time) - t_base
        y = np.array(self.history_pos)
        
        # 简单的最小二乘法拟合一次函数 (假设短时间内速度恒定)
        # A = [t, 1]
        A = np.vstack([t_local, np.ones(len(t_local))]).T
        k, b = np.linalg.lstsq(A, y, rcond=None)[0]
        
        # 预测: target_time 对应的本地时间
        target_t_local = target_time - t_base
        predicted_pos = k * target_t_local + b
        
        return predicted_pos
