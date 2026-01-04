import time
import numpy as np
import math
import random
import threading
import tkinter as tk
from mouse_utils import get_cursor_pos, move_mouse_relative, get_screen_size
from agent import Agent
from kalman_agent import KalmanAgent

class Benchmark:
    def __init__(self):
        self.w, self.h = get_screen_size()
        self.center_x, self.center_y = self.w // 2, self.h // 2
        self.running = True
        self.metrics = []

    def show_marker(self):
        """显示中心红点"""
        def _run():
            root = tk.Tk()
            root.overrideredirect(True)
            root.attributes("-topmost", True)
            root.attributes("-alpha", 0.7)
            size = 8
            root.geometry(f"{size}x{size}+{self.center_x-size//2}+{self.center_y-size//2}")
            label = tk.Label(root, bg="red")
            label.pack(fill="both", expand=True)
            root.mainloop()
        t = threading.Thread(target=_run, daemon=True)
        t.start()

    def reset_mouse(self):
        """强制将鼠标移回中心"""
        cx, cy = get_cursor_pos()
        move_mouse_relative(self.center_x - cx, self.center_y - cy)
        time.sleep(0.5)

    def run_phase(self, agent, phase_name, duration, disturbance_func):
        print(f"\n>>> 开始测试阶段: {phase_name} (持续 {duration}s)")
        self.reset_mouse()
        
        start_time = time.time()
        phase_errors = []
        
        while time.time() - start_time < duration:
            loop_start = time.time()
            
            # 1. 施加扰动 (Disturbance)
            # disturbance_func 返回 (dx, dy)
            dist_dx, dist_dy = disturbance_func(time.time() - start_time)
            if dist_dx != 0 or dist_dy != 0:
                move_mouse_relative(int(dist_dx), int(dist_dy))
            
            # 2. 获取当前状态
            real_x, real_y = get_cursor_pos()
            
            # 3. 算法控制 (Agent Step)
            # Agent 内部负责处理延迟模拟
            ctrl_dx, ctrl_dy = agent.step(loop_start, real_x, real_y, self.center_x, self.center_y)
            
            # 4. 执行控制
            if ctrl_dx != 0 or ctrl_dy != 0:
                move_mouse_relative(ctrl_dx, ctrl_dy)
            
            # 5. 记录误差
            # 重新获取位置以获得更准确的误差（可选，或者直接用 real_x/y）
            # 这里我们用 real_x/y 作为本帧的评估点
            error = np.sqrt((real_x - self.center_x)**2 + (real_y - self.center_y)**2)
            phase_errors.append(error)
            
            # 6. 维持循环频率 ~60Hz
            elapsed = time.time() - loop_start
            if elapsed < 0.016:
                time.sleep(0.016 - elapsed)
                
            # 安全退出
            if real_x < 10 and real_y < 10:
                print("检测到安全退出信号")
                self.running = False
                return None

        # 计算阶段得分
        rmse = np.sqrt(np.mean(np.array(phase_errors)**2))
        avg_err = np.mean(phase_errors)
        max_err = np.max(phase_errors)
        
        # 简单的分数计算：满分100，误差越大扣分越多
        # 假设 RMSE < 10px 为满分， > 500px 为0分
        score = max(0, 100 - (rmse - 10) * 0.5)
        if score > 100: score = 100
        
        print(f"[{phase_name}] 结果:")
        print(f"  RMSE: {rmse:.2f} px")
        print(f"  Avg Error: {avg_err:.2f} px")
        print(f"  Max Error: {max_err:.2f} px")
        print(f"  得分: {score:.1f}")
        
        return {"name": phase_name, "rmse": rmse, "score": score}

    def run_suite(self):
        self.show_marker()
        print(f"屏幕中心: ({self.center_x}, {self.center_y})")
        print("测试将在3秒后开始，请不要触碰鼠标...")
        time.sleep(3)
        
        # 定义要测试的算法
        agents = [
            ("PID + 线性回归", Agent()),
            ("PID + 卡尔曼滤波", KalmanAgent())
        ]
        
        all_results = {}
        
        # 定义扰动函数
        
        # 1. 匀速直线运动 (Constant Velocity)
        # 模拟鼠标持续向右下方漂移
        def drift_disturbance(t):
            return 3, 2  # 每帧移动 (3, 2)
            
        # 2. 加速运动 (Acceleration / Sine Wave)
        # 模拟正弦波往复运动
        def sine_disturbance(t):
            # x轴正弦速度: v = A * cos(wt)
            # 这里的返回值是位移 dx = v * dt
            # 假设 dt = 0.016
            freq = 2.0
            amp = 15.0
            dx = amp * math.cos(freq * t)
            dy = amp * math.sin(freq * t)
            return dx, dy
            
        # 3. 随机运动 (Random Walk)
        def random_disturbance(t):
            return random.randint(-10, 10), random.randint(-10, 10)

        # 执行测试
        for agent_name, agent in agents:
            print(f"\n{'='*20}")
            print(f"正在测试算法: {agent_name}")
            print(f"{'='*20}")
            
            agent_results = []
            
            if self.running:
                res = self.run_phase(agent, "匀速漂移对抗", 5.0, drift_disturbance)
                if res: agent_results.append(res)
                
            if self.running:
                time.sleep(1)
                res = self.run_phase(agent, "正弦波对抗", 5.0, sine_disturbance)
                if res: agent_results.append(res)
                
            if self.running:
                time.sleep(1)
                res = self.run_phase(agent, "随机抖动对抗", 5.0, random_disturbance)
                if res: agent_results.append(res)
            
            all_results[agent_name] = agent_results

        # 总分对比
        if all_results:
            print("\n" + "="*60)
            print("最终对比报告")
            print("="*60)
            
            for agent_name, results in all_results.items():
                print(f"\n算法: {agent_name}")
                print("-" * 30)
                total_score = 0
                for r in results:
                    print(f"{r['name']:<15} | RMSE: {r['rmse']:>6.2f} | 得分: {r['score']:>5.1f}")
                    total_score += r['score']
                
                avg_score = total_score / len(results) if results else 0
                print("-" * 30)
                print(f"综合评分: {avg_score:.1f} / 100")
            print("="*60)

if __name__ == "__main__":
    try:
        bm = Benchmark()
        bm.run_suite()
    except KeyboardInterrupt:
        print("\n测试已中断")
