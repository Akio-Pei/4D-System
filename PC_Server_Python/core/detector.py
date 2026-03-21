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

        self.event_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        self.thermal_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))

        self.t_thresh_c = 25.0
        self.e_area_min = 1000
        self.max_cooldown = 15

        # 🟢 巨物防御：目标框最大面积不能超过屏幕的 60%
        self.max_area_ratio = 0.6

    def update_params(self, t_val, e_val, cooldown):
        self.t_thresh_c = t_val
        self.e_area_min = e_val / 4.0
        self.max_cooldown = cooldown

    def pad_box(self, box, p=20):
        x, y, w, h = box
        # 🟢 严谨的数学锁：先算起点的绝对安全坐标，再算宽高，绝不溢出！
        nx = max(0, int(x - p))
        ny = max(0, int(y - p))
        nw = min(VIS_W - nx, int(w + 2 * p))
        nh = min(VIS_H - ny, int(h + 2 * p))
        return (nx, ny, nw, nh)

    def detect(self, aligned_temp_c, event_mask):
        box_e = None
        total_pixels = VIS_W * VIS_H

        # 🟢 1. 提取事件轮廓 (带全局防抖保护)
        event_pixel_count = cv2.countNonZero(event_mask)

        # 如果满屏噪点超过 30%，判定为【手持摄像机整体抖动】，直接屏蔽本帧事件信号！
        if event_pixel_count < total_pixels * 0.3:
            e_closed = cv2.morphologyEx(event_mask, cv2.MORPH_CLOSE, self.event_kernel)
            e_dilated = cv2.dilate(e_closed, self.event_kernel, iterations=1)
            cnts_e, _ = cv2.findContours(e_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if cnts_e:
                ce = max(cnts_e, key=cv2.contourArea)
                area = cv2.contourArea(ce)
                # 剔除小噪点，同时也剔除占据屏幕大半的假目标
                if self.e_area_min < area < (total_pixels * self.max_area_ratio):
                    box_e = cv2.boundingRect(ce)

        # 🟢 2. 提取热力轮廓 (带巨物防御)
        t_mask = (aligned_temp_c > self.t_thresh_c).astype(np.uint8) * 255
        t_closed = cv2.morphologyEx(t_mask, cv2.MORPH_CLOSE, self.thermal_kernel)
        cnts_t, _ = cv2.findContours(t_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        box_t = None
        if cnts_t:
            ct = max(cnts_t, key=cv2.contourArea)
            area_t = cv2.contourArea(ct)
            if 1500 < area_t < (total_pixels * self.max_area_ratio):
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