import cv2
import numpy as np
from config import VIS_W, VIS_H, THERMAL_W, THERMAL_H


class BoxSmoother:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 4)
        self.kf.transitionMatrix = np.eye(4, dtype=np.float32)
        self.kf.measurementMatrix = np.eye(4, dtype=np.float32)
        # 调小过程噪声，让框更稳
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.005
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.is_initialized = False

    def update(self, box):
        x, y, w, h = map(float, box)
        measurement = np.array([[x], [y], [w], [h]], dtype=np.float32)

        if not self.is_initialized:
            self.kf.statePost = measurement
            self.is_initialized = True
            return box

        self.kf.predict()
        estimated = self.kf.correct(measurement)
        return tuple(map(int, estimated.flatten()))

    def reset(self):
        self.is_initialized = False


class SentryDetector:
    def __init__(self):
        self.cooldown_frames = 0
        self.max_cooldown = 60
        self.is_triggered = False
        self.target_box = None
        self.smoother = BoxSmoother()

        # [优化] 超大膨胀核：专门解决"只框中指"的问题
        # 30x30 的核可以跨越整个手掌的宽度，把手指强行连成一体
        self.dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))

        # [优化] 增强去噪：过滤背景里的随机事件点
        self.noise_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    def detect(self, thermal_raw, event_mask):
        triggered_this_frame = False
        final_box = None

        # 1. 强力去除事件噪点
        clean_event_mask = cv2.morphologyEx(event_mask, cv2.MORPH_OPEN, self.noise_kernel)

        # 2. 热成像检测 (极低阈值，抓冷物体)
        # 降低到 25 度，尽可能捕捉手背的微弱热量
        temp_thresh_val = (25.0 + 273.15) * 64.0
        _, t_thresh = cv2.threshold(thermal_raw, temp_thresh_val, 65535, cv2.THRESH_BINARY)
        t_mask = t_thresh.astype(np.uint8)

        # 放大到可见光分辨率
        t_mask_big = cv2.resize(t_mask, (VIS_W, VIS_H), interpolation=cv2.INTER_NEAREST)

        # 3. 融合 (Union)
        combined_mask = cv2.bitwise_or(t_mask_big, clean_event_mask)

        # [核心] 闭运算+膨胀：填补空洞，连接断肢
        # 先闭运算填补手指间的缝隙
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, self.dilate_kernel)
        # 再膨胀一圈，确保框住边缘
        combined_mask = cv2.dilate(combined_mask, self.dilate_kernel, iterations=1)

        # 4. 寻找最大连通域
        cnts, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if cnts:
            c = max(cnts, key=cv2.contourArea)
            # 提高面积阈值，避免误报
            if cv2.contourArea(c) > 3000:
                triggered_this_frame = True
                x, y, w, h = cv2.boundingRect(c)

                # Padding: 稍微往外扩一点
                pad = 20
                x = max(0, x - pad)
                y = max(0, y - pad)
                w = min(VIS_W - x, w + 2 * pad)
                h = min(VIS_H - y, h + 2 * pad)

                final_box = (x, y, w, h)

        # 5. 滤波输出
        if triggered_this_frame and final_box is not None:
            self.is_triggered = True
            self.cooldown_frames = self.max_cooldown
            self.target_box = self.smoother.update(final_box)
        else:
            if self.cooldown_frames > 0:
                self.cooldown_frames -= 1
            else:
                self.is_triggered = False
                self.target_box = None
                self.smoother.reset()

        return self.is_triggered, self.target_box