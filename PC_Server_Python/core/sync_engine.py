import cv2
import numpy as np
import time
import os
import csv
import json  # <--- [新增] 用来存参数
import datetime
from collections import deque
from PyQt6.QtCore import QThread, pyqtSignal
from algorithms.vignetting import VignettingCorrector
from algorithms.event_sim import PseudoEventGen
from algorithms.alignment import ImageAligner
from core.detector import SentryDetector
from config import THERMAL_W, THERMAL_H, VIS_W, VIS_H


class SyncEngine(QThread):
    # 信号: Fusion, Thermal, Event, ROI, Depth, Info
    update_signal = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict)
    log_signal = pyqtSignal(str)

    def __init__(self, q_vis, q_therm):
        super().__init__()
        self.q_vis, self.q_therm = q_vis, q_therm
        self.running = True
        self.mode = "LOCKED"
        self.checker_mode = False

        try:
            self.algo_vign = VignettingCorrector()
            self.algo_align = ImageAligner()
            self.algo_evt = PseudoEventGen(width=VIS_W, height=VIS_H)
            self.detector = SentryDetector()
        except:
            pass

        self.cache_t_raw = np.zeros((THERMAL_H, THERMAL_W), dtype=np.uint16)
        self.event_buffer = deque(maxlen=20)
        self.fps_cnt = 0
        self.curr_fps = 0.0
        self.fps_timer = time.time()

        # === 录制系统状态 ===
        self.is_recording = False
        self.rec_dir = ""
        self.csv_writer = None
        self.rec_file = None
        self.rec_frame_id = 0

        # 3D 交互
        self.depth_yaw = 0.0
        self.depth_pitch = 0.5
        self.depth_inverse = False

    def set_mode(self, mode):
        self.mode = mode
        self.log_signal.emit(f">>> MODE: {mode}")

    def update_align_params(self, dx=0, dy=0, d_scale=None, set_scale=None, set_angle=None, toggle_checker=False):
        try:
            nx = self.algo_align.x + dx
            ny = self.algo_align.y + dy
            ns = self.algo_align.scale
            if d_scale: ns *= d_scale
            if set_scale: ns = set_scale
            na = self.algo_align.angle
            if set_angle is not None: na = set_angle
            if toggle_checker: self.checker_mode = not self.checker_mode
            self.algo_align.update_params(x=nx, y=ny, scale=ns, angle=na, opacity=0.5)
        except:
            pass

    def update_depth_rotation(self, dx, dy):
        self.depth_yaw += dx * 0.02
        self.depth_pitch += dy * 0.02
        self.depth_pitch = np.clip(self.depth_pitch, -1.5, 1.5)

    def reset_3d_view(self):
        self.depth_yaw = 0.0
        self.depth_pitch = 0.5
        self.log_signal.emit("[3D] View Reset")

    def toggle_depth_inverse(self):
        self.depth_inverse = not self.depth_inverse
        state = "INVERTED" if self.depth_inverse else "NORMAL"
        self.log_signal.emit(f"[3D] Depth {state}")

    def rotate_image(self, image, angle):
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(image, M, (w, h))

    # === [核心功能] 智能录制启动 ===
    def start_recording(self):
        if self.is_recording: return
        self.is_recording = True
        self.rec_frame_id = 0

        # 重置算法状态，确保第一帧干净
        self.algo_evt.reset()

        # 创建带时间戳的数据集目录
        ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.rec_dir = os.path.join("auto_captures", f"target_{ts_str}")

        # HexPlane 需要的数据结构
        os.makedirs(os.path.join(self.rec_dir, "visual"), exist_ok=True)
        os.makedirs(os.path.join(self.rec_dir, "thermal"), exist_ok=True)
        os.makedirs(os.path.join(self.rec_dir, "event_hq"), exist_ok=True)

        # 🔥 [关键修改] 保存当前的对齐参数！
        # 这样 tools/prepare_hexplane.py 就能知道怎么对齐了
        try:
            align_params = {
                "x": self.algo_align.x,
                "y": self.algo_align.y,
                "scale": self.algo_align.scale,
                "angle": self.algo_align.angle
            }
            with open(os.path.join(self.rec_dir, "align_params.json"), "w") as f:
                json.dump(align_params, f, indent=4)
            self.log_signal.emit(f"[REC] Saved alignment params.")
        except Exception as e:
            print(f"Error saving align params: {e}")

        # 创建对齐索引表
        self.rec_file = open(os.path.join(self.rec_dir, "alignment.csv"), "w", newline='')
        self.csv_writer = csv.writer(self.rec_file)
        self.csv_writer.writerow(["frame_id", "timestamp_vis", "timestamp_therm"])

        self.log_signal.emit(f"[REC] AUTO-TRIGGERED: {ts_str}")

    def stop_recording(self):
        if not self.is_recording: return
        self.is_recording = False
        if self.rec_file:
            self.rec_file.close()
            self.rec_file = None
        self.log_signal.emit(f"[REC] STOPPED. Saved {self.rec_frame_id} frames.")

    def draw_idle_roi(self, width, height):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.rectangle(img, (0, 0), (width - 1, height - 1), (40, 40, 40), 1)
        text = "NO TARGET"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        th = 1
        (tw, th_px), _ = cv2.getTextSize(text, font, scale, th)
        cv2.putText(img, text, ((width - tw) // 2, (height + th_px) // 2), font, scale, (100, 100, 100), th)
        return img

    def get_smart_crop(self, image, box, out_size=(120, 120)):
        bx, by, bw, bh = box
        ih, iw = image.shape[:2]
        pad_x, pad_y = int(bw * 0.5), int(bh * 0.5)
        x1, y1 = max(0, bx - pad_x), max(0, by - pad_y)
        x2, y2 = min(iw, bx + bw + pad_x), min(ih, by + bh + pad_y)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0: return np.zeros((out_size[1], out_size[0], 3), dtype=np.uint8)
        ch, cw = crop.shape[:2]
        tw, th = out_size
        scale = min(tw / cw, th / ch)
        new_w, new_h = int(cw * scale), int(ch * scale)
        resized = cv2.resize(crop, (new_w, new_h))
        canvas = np.zeros((th, tw, 3), dtype=np.uint8)
        dx, dy = (tw - new_w) // 2, (th - new_h) // 2
        canvas[dy:dy + new_h, dx:dx + new_w] = resized
        return canvas

    def render_3d_depth_fusion(self, thermal_raw, event_mask, width=120, height=120):
        grid_w, grid_h = 320, 200
        t_small = cv2.resize(thermal_raw, (grid_w, grid_h), interpolation=cv2.INTER_CUBIC)
        z_therm = t_small.astype(np.float32)
        z_therm = (z_therm - z_therm.min()) / (z_therm.max() - z_therm.min() + 1e-5)
        e_small = cv2.resize(event_mask, (grid_w, grid_h), interpolation=cv2.INTER_NEAREST)

        # 3D 融合逻辑
        raw_mask = np.zeros_like(z_therm)
        raw_mask[(z_therm > 0.15) | (e_small > 0)] = 1.0
        fill_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        body_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, fill_kernel)
        body_mask = cv2.GaussianBlur(body_mask, (15, 15), 0)

        z_final = (body_mask * 0.5) + (z_therm * body_mask * 0.5)

        if self.depth_inverse:
            z_final = 1.0 - z_final
            z_final[body_mask < 0.1] = 0

        x = np.linspace(-1.6, 1.6, grid_w)
        y = np.linspace(-1.0, 1.0, grid_h)
        xv, yv = np.meshgrid(x, y)
        c, s = np.cos(self.depth_pitch), np.sin(self.depth_pitch)
        Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        c, s = np.cos(self.depth_yaw), np.sin(self.depth_yaw)
        Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        R = Ry @ Rx
        points_3d = np.stack([xv, yv, -z_final * 0.8], axis=-1).reshape(-1, 3)
        points_rot = points_3d @ R.T
        dist = 2.5
        x_proj = points_rot[:, 0] / (points_rot[:, 2] + dist)
        y_proj = points_rot[:, 1] / (points_rot[:, 2] + dist)
        u = (x_proj * width * 0.7 + width / 2).astype(np.int32)
        v = (y_proj * width * 0.7 + height / 2).astype(np.int32)

        img = np.zeros((height, width, 3), dtype=np.uint8)
        colors = cv2.applyColorMap((z_final.reshape(-1) * 255).astype(np.uint8), cv2.COLORMAP_MAGMA).reshape(-1, 3)
        valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)

        if np.any(valid):
            z_vals = points_rot[valid, 2]
            sort_idx = np.argsort(z_vals)[::-1]
            img[v[valid][sort_idx], u[valid][sort_idx]] = colors[valid][sort_idx]

        return img

    def run(self):
        self.log_signal.emit("[CORE] ENGINE STARTED (HEXPLANE READY)")

        while self.running:
            try:
                # 1. 采集数据
                if len(self.q_vis) == 0: self.msleep(1); continue
                while len(self.q_vis) > 1: self.q_vis.popleft()
                v_ts, v_fid, v_raw = self.q_vis.popleft()

                # 图像处理
                v_corr = self.algo_vign.process(v_raw)
                e_mask_fast = self.algo_evt.process_fast(v_corr)
                self.event_buffer.append((e_mask_fast, v_ts, v_corr))

                v_bg = cv2.cvtColor(v_corr, cv2.COLOR_GRAY2BGR)
                e_disp = np.zeros_like(v_bg)
                e_disp[e_mask_fast > 0] = [0, 255, 0]

                if len(self.q_therm) > 0:
                    while len(self.q_therm) > 1: self.q_therm.pop()
                    t_ts, t_fid, t_raw = self.q_therm.pop()
                    self.cache_t_raw = t_raw
                else:
                    t_ts = v_ts

                # 2. 检测与录制
                triggered, target_box = self.detector.detect(self.cache_t_raw, e_mask_fast)

                if triggered:
                    self.start_recording()
                else:
                    self.stop_recording()

                # 3. 满血录制
                if self.is_recording:
                    e_mask_hq = self.algo_evt.process_hq(v_corr)
                    fname = f"{self.rec_frame_id:06d}.png"
                    cv2.imwrite(os.path.join(self.rec_dir, "visual", fname), v_corr)
                    cv2.imwrite(os.path.join(self.rec_dir, "thermal", fname), self.cache_t_raw)
                    cv2.imwrite(os.path.join(self.rec_dir, "event_hq", fname), e_mask_hq)
                    self.csv_writer.writerow([self.rec_frame_id, v_ts, t_ts])
                    self.rec_frame_id += 1

                # 4. 融合渲染 (这里只是为了界面显示)
                raw_f = self.cache_t_raw.astype(np.float32)
                temp_c = raw_f / 64.0 - 273.15
                t_min, t_max = np.min(temp_c), np.max(temp_c)
                if t_max - t_min < 2.0: t_max = t_min + 5.0
                t_norm = ((temp_c - t_min) / (t_max - t_min) * 255).astype(np.uint8)
                t_color = cv2.applyColorMap(t_norm, cv2.COLORMAP_JET)

                tx, ty, tw, th, angle, _ = self.algo_align.get_transform_params()
                t_scaled = cv2.resize(t_color, (tw, th))
                t_rotated = self.rotate_image(t_scaled, angle)
                x1 = max(0, tx); y1 = max(0, ty)
                x2 = min(VIS_W, tx + tw); y2 = min(VIS_H, ty + th)
                vw = x2 - x1; vh = y2 - y1
                final_fusion = v_bg.copy()
                if vw > 0 and vh > 0:
                    v_crop = v_bg[y1:y2, x1:x2]
                    ox = x1 - tx; oy = y1 - ty
                    if oy + vh <= t_rotated.shape[0] and ox + vw <= t_rotated.shape[1]:
                        t_crop = t_rotated[oy:oy + vh, ox:ox + vw]
                        if self.checker_mode:
                            mask = ((np.indices((vh, vw))[0] // 32 + np.indices((vh, vw))[1] // 32) % 2 == 0)
                            mask = np.dstack([mask] * 3)
                            blend = np.where(mask, v_crop, t_crop)
                            final_fusion[y1:y2, x1:x2] = blend
                        else:
                            blend = cv2.addWeighted(v_crop, 0.6, t_crop, 0.7, 0)
                            final_fusion[y1:y2, x1:x2] = blend
                        color = (0, 0, 255) if self.is_recording else (0, 255, 0)
                        if self.mode != "LOCKED":
                            cv2.rectangle(final_fusion, (x1, y1), (x2, y2), color, 2)
                        else:
                            final_fusion = blend

                depth_display = self.render_3d_depth_fusion(self.cache_t_raw, e_mask_fast, 120, 120)
                if target_box is not None:
                    roi_display = self.get_smart_crop(v_bg, target_box, (120, 120))
                    cv2.rectangle(roi_display, (0, 0), (119, 119), (0, 0, 255), 1)
                else:
                    roi_display = self.draw_idle_roi(120, 120)

                self.fps_cnt += 1
                if time.time() - self.fps_timer >= 1.0:
                    self.curr_fps = self.fps_cnt; self.fps_cnt = 0; self.fps_timer = time.time()
                status_txt = "REC ●" if self.is_recording else "SCAN"
                info = {"fps": self.curr_fps, "mode": status_txt, "rec": "REC" if self.is_recording else ""}
                self.update_signal.emit(final_fusion, t_color, e_disp, roi_display, depth_display, info)

            except Exception as e:
                print(f"Sync: {e}")
                self.msleep(10)

    def stop(self):
        self.stop_recording()
        self.running = False; self.wait()