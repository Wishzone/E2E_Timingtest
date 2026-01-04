import time
import ctypes
import numpy as np
from collections import deque
from pre import Predictor
import threading
import tkinter as tk
import csv
from datetime import datetime

# Windows API 常量
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000

class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

def get_cursor_pos():
    pt = POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y

def move_mouse_relative(dx, dy):
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_MOVE, int(dx), int(dy), 0, 0)

def get_screen_size():
    w = ctypes.windll.user32.GetSystemMetrics(0)
    h = ctypes.windll.user32.GetSystemMetrics(1)
    return w, h

def show_center_marker(center_x, center_y):
    """在屏幕中心显示一个红点"""
    def _run():
        root = tk.Tk()
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.attributes("-alpha", 0.7)  # 半透明
        # 红色圆点(实际上是方块)
        size = 6
        root.geometry(f"{size}x{size}+{center_x-size//2}+{center_y-size//2}")
        label = tk.Label(root, bg="red")
        label.pack(fill="both", expand=True)
        root.mainloop()
    
    t = threading.Thread(target=_run, daemon=True)
    t.start()

def save_metrics(data):
    if not data:
        return

    # 计算统计指标
    distances = [d['distance'] for d in data]
    avg_dist = np.mean(distances)
    max_dist = np.max(distances)
    rmse = np.sqrt(np.mean(np.array(distances)**2))
    
    print("\n" + "="*30)
    print("测试结果统计")
    print("="*30)
    print(f"平均误差距离: {avg_dist:.2f} px")
    print(f"最大误差距离: {max_dist:.2f} px")
    print(f"RMSE (均方根误差): {rmse:.2f} px")
    print(f"总样本数: {len(data)}")
    print("="*30)
    
    # 保存到文件
    filename = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    try:
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['timestamp', 'x', 'y', 'error_x', 'error_y', 'distance'])
            writer.writeheader()
            writer.writerows(data)
        print(f"详细数据已保存至: {filename}")
    except Exception as e:
        print(f"保存数据失败: {e}")

def main():
    # 获取屏幕中心
    w, h = get_screen_size()
    center_x, center_y = w // 2, h // 2
    print(f"屏幕分辨率: {w}x{h}, 中心点: ({center_x}, {center_y})")
    
    # 显示中心标记
    show_center_marker(center_x, center_y)
    
    print("程序将在3秒后开始控制鼠标...")
    print("【安全警告】如需停止，请将鼠标快速移动到屏幕左上角 (0,0)")
    time.sleep(3)

    # 初始化预测器 (X和Y轴分开)
    pred_x = Predictor()
    pred_y = Predictor()
    
    # 延迟缓冲区
    # 存储 (timestamp, x, y)
    history_buffer = deque()
    
    # 动作缓冲区：存储已发送但尚未被观测到的控制指令
    # 存储 (timestamp, dx, dy)
    action_buffer = deque()
    
    DELAY_SEC = 0.070  # 70ms 延迟
    
    # 控制参数
    Kp = 0.3  # 比例增益
    Ki = 2.0  # 积分增益：用于消除稳态误差
    
    # 积分误差累积
    integral_x = 0.0
    integral_y = 0.0
    # 积分限幅 (Anti-windup)，防止积分项过大导致过冲
    # 限制积分项能贡献的最大速度
    max_integral_contribution = 3000.0 
    
    # 性能指标记录
    metrics_data = []
    start_time = time.time()
    
    try:
        while True:
            current_time = time.time()
            
            # 1. 获取真实位置 (模拟传感器读取)
            real_x, real_y = get_cursor_pos()
            
            # 记录指标
            curr_error_x = real_x - center_x
            curr_error_y = real_y - center_y
            curr_dist = np.sqrt(curr_error_x**2 + curr_error_y**2)
            
            # 只记录开始控制后的数据（比如前3秒不记录，或者这里直接记录）
            # 这里我们记录所有数据
            metrics_data.append({
                'timestamp': current_time - start_time,
                'x': real_x,
                'y': real_y,
                'error_x': curr_error_x,
                'error_y': curr_error_y,
                'distance': curr_dist
            })
            
            # 安全退出检测
            if real_x < 10 and real_y < 10:
                print("检测到鼠标在左上角，程序退出。")
                break
                
            # 2. 存入延迟缓冲区
            history_buffer.append((current_time, real_x, real_y))
            
            # 3. 获取延迟后的观测值
            # 寻找最接近 current_time - DELAY_SEC 的数据
            target_time = current_time - DELAY_SEC
            
            delayed_data = None
            # 从旧到新遍历，找到第一个时间 >= target_time 的
            best_match = None
            for item in history_buffer:
                t, x, y = item
                if t >= target_time:
                    best_match = item
                    break
            
            # 如果没找到（缓冲区数据都比目标时间旧），取最新的
            if best_match is None and len(history_buffer) > 0:
                best_match = history_buffer[-1]
            elif best_match is None:
                best_match = (current_time, real_x, real_y)
                
            obs_time, obs_x, obs_y = best_match
            
            # 清理过旧的数据 (保留最近 200ms)
            while len(history_buffer) > 0 and history_buffer[0][0] < current_time - 0.2:
                history_buffer.popleft()
                
            # 4. 预测算法更新
            pred_x.update(obs_time, obs_x)
            pred_y.update(obs_time, obs_y)
            
            # 5. 预测当前位置
            # 基础预测：不再进行时间外推，只进行平滑处理
            # 我们只信任观测到的位置 + 我们自己发出的动作
            # 如果进行外推，会将我们之前的控制动作误判为外部速度，导致震荡
            base_est_x = pred_x.predict(obs_time)
            base_est_y = pred_y.predict(obs_time)
            
            # 关键修正：加上所有“已发送但未被观测到”的控制量 (Smith Predictor 思想)
            # 观测值 obs_time 之后的所有动作，其效果尚未体现在 obs_x/y 中
            pending_dx = 0
            pending_dy = 0
            
            # 清理过期的动作记录
            while len(action_buffer) > 0 and action_buffer[0][0] < obs_time:
                action_buffer.popleft()
                
            # 累加未观测到的动作
            for t, dx, dy in action_buffer:
                if t > obs_time:
                    pending_dx += dx
                    pending_dy += dy
            
            # 最终估计位置 = 基础预测 + 待生效的控制量
            est_x = base_est_x + pending_dx
            est_y = base_est_y + pending_dy
            
            # 6. 计算控制量
            # 目标是中心点
            # Error = Target - Estimated_Current
            error_x = center_x - est_x
            error_y = center_y - est_y
            
            # 积分项更新 (dt 约为 0.01s)
            dt = 0.01
            integral_x += error_x * dt
            integral_y += error_y * dt
            
            # 积分抗饱和 (Clamping)
            # 限制积分项的数值，防止其无限增长
            limit_val = max_integral_contribution / Ki
            integral_x = np.clip(integral_x, -limit_val, limit_val)
            integral_y = np.clip(integral_y, -limit_val, limit_val)
            
            # 死区处理：如果误差很小，清零积分并停止移动，避免在中心点附近抖动
            if abs(error_x) < 10: 
                error_x = 0
                integral_x = 0 # 进入死区后清除积分，避免过冲
            if abs(error_y) < 10: 
                error_y = 0
                integral_y = 0
            
            # PI控制 (比例 + 积分)
            # 积分项会随着时间积累误差，从而提供额外的拉力来消除稳态距离
            move_x = error_x * Kp + integral_x * Ki
            move_y = error_y * Kp + integral_y * Ki
            
            # 限制单次移动幅度，防止飞出
            max_step = 50
            move_x = np.clip(move_x, -max_step, max_step)
            move_y = np.clip(move_y, -max_step, max_step)
            
            # 7. 执行移动 (相对移动)
            # 只有当误差足够大时才移动，避免抖动
            if abs(move_x) > 1 or abs(move_y) > 1:
                int_move_x = int(move_x)
                int_move_y = int(move_y)
                move_mouse_relative(int_move_x, int_move_y)
                # 记录实际发送的整数值，确保模型与现实一致
                action_buffer.append((time.time(), int_move_x, int_move_y))
            
            # 控制循环频率
            time.sleep(0.01) # ~100Hz
            
    except KeyboardInterrupt:
        print("\n程序已停止")
    finally:
        save_metrics(metrics_data)

if __name__ == "__main__":
    main()
