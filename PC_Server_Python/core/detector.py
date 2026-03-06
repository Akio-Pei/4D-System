import cv2
import numpy as np
from config import VIS_W, VIS_H


class BoxSmoother:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 4)
        self.kf.transitionMatrix = np.eye(4, dtype=np.float32)
        self.kf.measurementMatrix = np.eye(4, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05
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
        self.state = "NONE"
        self.target_box = None
        self.smoother = BoxSmoother()

        # 事件流变成了极细的线，我们需要一个中等大小的核把它“充气”连接成一个面来算面积
        self.event_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        self.thermal_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))

        self.t_thresh_c = 25.0
        self.e_area_min = 1000
        self.max_cooldown = 15

    def update_params(self, t_val, e_val, cooldown):
        self.t_thresh_c = t_val
        # 线条轮廓面积很小，UI传入的阈值在此缩小适配
        self.e_area_min = e_val / 4.0
        self.max_cooldown = cooldown

    def pad_box(self, box, p=20):
        x, y, w, h = box
        return (max(0, x - p), max(0, y - p), min(VIS_W - (x - p), w + 2 * p), min(VIS_H - (y - p), h + 2 * p))

    def detect(self, aligned_temp_c, event_mask):
        # 1. 提取事件轮廓 (Blue)
        # 将线框闭合并膨胀成实心色块以便捕捉
        e_closed = cv2.morphologyEx(event_mask, cv2.MORPH_CLOSE, self.event_kernel)
        e_dilated = cv2.dilate(e_closed, self.event_kernel, iterations=1)
        cnts_e, _ = cv2.findContours(e_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        box_e = None
        if cnts_e:
            ce = max(cnts_e, key=cv2.contourArea)
            if cv2.contourArea(ce) > self.e_area_min:
                box_e = cv2.boundingRect(ce)

        # 2. 提取热力轮廓 (Green)
        # 输入已经是物理对齐好的摄氏度矩阵，直接用 UI 的 t_thresh_c 进行布尔截断！
        t_mask = (aligned_temp_c > self.t_thresh_c).astype(np.uint8) * 255
        t_closed = cv2.morphologyEx(t_mask, cv2.MORPH_CLOSE, self.thermal_kernel)
        cnts_t, _ = cv2.findContours(t_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        box_t = None
        if cnts_t:
            ct = max(cnts_t, key=cv2.contourArea)
            if cv2.contourArea(ct) > 1500:
                box_t = cv2.boundingRect(ct)

        # 3. 状态判定与框融合
        current_state = "NONE"
        final_box = None

        if box_t and box_e:
            tx, ty, tw, th = box_t
            ex, ey, ew, eh = box_e
            ix1, iy1 = max(tx, ex), max(ty, ey)
            ix2, iy2 = min(tx + tw, ex + ew), min(ty + th, ey + eh)

            if ix1 < ix2 and iy1 < iy2:
                current_state = "BOTH"
                mx1, my1 = min(tx, ex), min(ty, ey)
                mx2, my2 = max(tx + tw, ex + ew), max(ty + th, ey + eh)
                final_box = self.pad_box((mx1, my1, mx2 - mx1, my2 - my1))
            else:
                current_state = "THERMAL"
                final_box = self.pad_box(box_t)
        elif box_t:
            current_state = "THERMAL"
            final_box = self.pad_box(box_t)
        elif box_e:
            current_state = "EVENT"
            final_box = self.pad_box(box_e)

        if current_state != "NONE":
            self.cooldown_frames = self.max_cooldown
            self.state = current_state
            self.target_box = self.smoother.update(final_box)
        else:
            if self.cooldown_frames > 0:
                self.cooldown_frames -= 1
            else:
                self.state = "NONE"
                self.target_box = None
                self.smoother.reset()

        return self.state, self.target_box