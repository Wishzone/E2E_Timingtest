import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from pre import Predictor

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows自带的黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 配置参数
DELAY_MS = 70        # 系统延迟 (ms)
DT_MS = 10           # 仿真步长 (ms)
TOTAL_TIME_SEC = 30  # 总仿真时间 (s)

class Simulation:
    def __init__(self):
        self.true_pos = 0.0
        # 历史缓冲区用于模拟延迟：存储 (simulation_time, position)
        self.history_buffer = deque() 
        self.predictor = Predictor()
        
        # 用于绘图的数据
        self.logs = {
            't': [], 
            'true': [], 
            'measured': [], 
            'predicted': []
        }

    def get_delayed_measurement(self, current_sim_time):
        """
        模拟获取延迟后的传感器数据
        返回: (measurement_time, position)
        """
        target_time = current_sim_time - (DELAY_MS / 1000.0)
        
        # 在缓冲区中找到最接近 target_time 的数据
        best_match = (0.0, 0.0)
        found = False
        
        # 遍历缓冲区找到符合延迟时间的数据
        for t, pos in self.history_buffer:
            if t >= target_time:
                best_match = (t, pos)
                found = True
                break
        
        # 如果没找到（比如刚开始），使用最新的数据或者0
        if not found and self.history_buffer:
            best_match = self.history_buffer[-1]
            
        return best_match

    def run(self):
        print(f"开始仿真: 延迟={DELAY_MS}ms, 步长={DT_MS}ms")
        sim_time = 0.0
        
        # 初始化缓冲区，假设开始前位置都在0
        for i in range(int(DELAY_MS/DT_MS) + 5):
            t = - (DELAY_MS/1000.0) + i * (DT_MS/1000.0)
            self.history_buffer.append((t, 0.0))

        # 设置绘图
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 6))
        line_true, = ax.plot([], [], 'b-', label='真实位置 (True)', linewidth=2)
        line_pred, = ax.plot([], [], 'r--', label='预测位置 (Predicted)', alpha=0.7)
        line_meas, = ax.plot([], [], 'g:', label='延迟观测 (Delayed)', alpha=0.5)
        
        ax.set_title(f'70ms 延迟补偿仿真 (PID控制)')
        ax.set_xlabel('时间 (s)')
        ax.set_ylabel('位置 (偏离中心点的距离)')
        ax.legend(loc='upper right')
        ax.grid(True)
        
        # 动态调整坐标轴
        ax.set_xlim(0, 5)
        ax.set_ylim(-5, 5)

        try:
            while sim_time < TOTAL_TIME_SEC:
                # 1. 模拟环境扰动 (鼠标移动)
                # 模拟一个持续的移动趋势(正弦波)加上随机抖动
                disturbance = 3.0 * np.sin(sim_time * 1.0) + np.random.normal(0, 0.5)
                
                # 2. 获取延迟的测量值 (模拟系统只能看到70ms前的数据)
                meas_time, meas_pos = self.get_delayed_measurement(sim_time)
                
                # 3. 预测算法
                # 将过时的测量值喂给预测器
                self.predictor.update(meas_time, meas_pos)
                
                # 预测当前时刻 (sim_time) 的位置
                # 预测器会根据历史轨迹外推这70ms的变化
                pred_pos = self.predictor.predict(sim_time)
                
                # 4. 控制系统 (P控制器)
                # 目标是保持位置为 0
                # Error = Target - Predicted
                error = 0.0 - pred_pos
                
                # 简单的 P 控制系数
                # 如果预测准确，这个控制量应该能抵消扰动并将位置拉回0
                Kp = 1.5 
                control_action = Kp * error
                
                # 5. 更新物理系统
                # 位置变化 = (扰动速度 + 控制速度) * dt
                # 这里假设 disturbance 和 control_action 都是速度量纲
                self.true_pos += (disturbance + control_action) * (DT_MS / 1000.0)
                
                # 6. 记录真实状态到缓冲区 (用于未来的延迟测量)
                self.history_buffer.append((sim_time, self.true_pos))
                
                # 清理过长的缓冲区 (保留最近 200ms 数据即可)
                while len(self.history_buffer) > 0 and self.history_buffer[0][0] < sim_time - 0.2:
                    self.history_buffer.popleft()

                # 记录日志
                self.logs['t'].append(sim_time)
                self.logs['true'].append(self.true_pos)
                self.logs['measured'].append(meas_pos)
                self.logs['predicted'].append(pred_pos)
                
                # 绘图更新 (每 100ms 更新一次画面)
                if int(sim_time * 1000) % 100 == 0:
                    # 保持显示最近5秒的数据
                    t_data = np.array(self.logs['t'])
                    start_idx = max(0, len(t_data) - int(5.0 / (DT_MS/1000.0)))
                    
                    current_t = t_data[start_idx:]
                    
                    line_true.set_data(current_t, self.logs['true'][start_idx:])
                    line_pred.set_data(current_t, self.logs['predicted'][start_idx:])
                    line_meas.set_data(current_t, self.logs['measured'][start_idx:])
                    
                    ax.set_xlim(max(0, sim_time - 5), max(5, sim_time))
                    
                    # 动态调整Y轴
                    y_data = self.logs['true'][start_idx:]
                    if len(y_data) > 0:
                        ymin, ymax = min(y_data), max(y_data)
                        margin = max(1.0, (ymax - ymin) * 0.2)
                        ax.set_ylim(ymin - margin, ymax + margin)
                    
                    plt.pause(0.001)
                
                sim_time += DT_MS / 1000.0
                # time.sleep(0.001) # 如果需要减慢仿真速度可以取消注释
                
        except KeyboardInterrupt:
            print("仿真中断")
        
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    sim = Simulation()
    sim.run()
