import cv2
import time
import numpy as np
import sys
try:
    from pynput import mouse
except ImportError:
    print("请先安装 pynput 库: pip install pynput")
    sys.exit(1)

# --- 配置 ---
IMAGE_PATH = 'test.png'
TIMEOUT = 5.0
MOVEMENT_THRESHOLD = 10

# 全局变量
start_time = 0
end_time = 0
is_moved = False
start_mouse_pos = None

def on_move(x, y):
    global end_time, is_moved, start_time, start_mouse_pos
    
    if start_time == 0 or start_mouse_pos is None:
        return

    if is_moved:
        return False

    # 计算相对于起始位置的移动距离
    dist = ((x - start_mouse_pos[0])**2 + (y - start_mouse_pos[1])**2)**0.5
    if dist > MOVEMENT_THRESHOLD:
        end_time = time.time()
        is_moved = True
        return False

def main():
    global start_time, is_moved, start_mouse_pos

    # 读取图片
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"错误: 无法找到 {IMAGE_PATH}")
        sys.exit(1)

    # 准备窗口
    window_name = 'Latency Test'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # 显示黑屏
    black_img = np.zeros_like(img)
    cv2.imshow(window_name, black_img)
    cv2.waitKey(1)
    
    print("=== 自动化延迟测试 ===")
    print("等待 2 秒...")

    # 启动监听
    listener = mouse.Listener(on_move=on_move)
    listener.start()
    mouse_controller = mouse.Controller()

    # 固定等待 2 秒 (保持窗口响应)
    start_wait = time.time()
    while time.time() - start_wait < 2.0:
        cv2.waitKey(10)

    # 记录基准位置
    start_mouse_pos = mouse_controller.position

    # 显示图片并计时
    cv2.imshow(window_name, img)
    cv2.waitKey(1) # 强制刷新
    start_time = time.time()

    # 等待响应
    while not is_moved:
        if time.time() - start_time > TIMEOUT:
            break
        cv2.waitKey(1)

    listener.stop()
    cv2.destroyAllWindows()

    if is_moved:
        latency_ms = (end_time - start_time) * 1000
        print(f"\n>>> 延迟: {latency_ms:.2f} ms <<<")
    else:
        print("\n>>> 超时 <<<")

if __name__ == "__main__":
    main()
