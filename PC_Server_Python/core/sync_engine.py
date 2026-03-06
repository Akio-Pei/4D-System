import cv2
import numpy as np
import time
import os
import csv
import json
import datetime
import shutil
import threading
import queue
from collections import deque
from PyQt6.QtCore import QThread, pyqtSignal
from algorithms.vignetting import VignettingCorrector
from algorithms.alignment import ImageAligner
from algorithms.event_sim import PseudoEventGen
from core.detector import SentryDetector
from config import THERMAL_W, THERMAL_H, VIS_W, VIS_H


class SyncEngine(QThread):
    update_signal = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict)
    log_signal = pyqtSignal(str)

    def __init__(self, q_vis, q_therm):
        super().__init__()
        self.q_vis, self.q_therm = q_vis, q_therm
        self.running = True
        self.mode = "LOCKED"
        self.checker_mode = False
        self.trigger_mode = "BOTH"

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

        self.is_recording = False
        self.rec_dir = ""
        self.rec_frame_id = 0
        self.depth_yaw = 0.0
        self.depth_pitch = 0.5
        self.depth_inverse = False

        self.prev_evt_frame = None

        # 🟢 【核心规则】：标准块大小死死锁定为 60 帧
        self.CHUNK_SIZE = 60
        self.lost_frames = 0
        self.TOLERANCE_FRAMES = 10

        self.engine_start_time = time.time()
        self.bg_thermal_map = np.full((VIS_H, VIS_W), -50.0, dtype=np.float32)
        self.current_t_sens = 25.0

        self.last_target_box = None
        self.io_queue = queue.Queue()
        self.io_thread = threading.Thread(target=self._io_writer_loop, daemon=True)
        self.io_thread.start()

    def _io_writer_loop(self):
        while self.running:
            try:
                task = self.io_queue.get(timeout=0.1)
                cmd = task[0]

                if cmd == "INIT_CSV":
                    rec_dir = task[1]
                    with open(os.path.join(rec_dir, "alignment.csv"), "w", newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(["frame_id", "timestamp_vis", "timestamp_therm"])

                elif cmd == "WRITE_FRAME":
                    _, rec_dir, fid, vis, therm, evt, v_ts, t_ts = task
                    cv2.imwrite(os.path.join(rec_dir, "visual", f"{fid:06d}.png"), vis)
                    cv2.imwrite(os.path.join(rec_dir, "thermal", f"{fid:06d}.png"), therm)
                    cv2.imwrite(os.path.join(rec_dir, "event_hq", f"{fid:06d}.png"), evt)

                    with open(os.path.join(rec_dir, "alignment.csv"), "a", newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([fid, v_ts, t_ts])

                elif cmd == "DESTROY_DIR":
                    rec_dir = task[1]
                    try:
                        shutil.rmtree(rec_dir)
                    except:
                        pass
            except queue.Empty:
                pass

    def generate_robust_events(self, curr_img):
        if self.prev_evt_frame is None:
            self.prev_evt_frame = curr_img.copy()
            return np.zeros_like(curr_img)
        blur_curr = cv2.GaussianBlur(curr_img, (3, 3), 0)
        blur_prev = cv2.GaussianBlur(self.prev_evt_frame, (3, 3), 0)
        diff = cv2.absdiff(blur_curr, blur_prev)
        _, mask = cv2.threshold(diff, 8, 255, cv2.THRESH_BINARY)
        mask_clean = cv2.medianBlur(mask, 3)
        self.prev_evt_frame = curr_img.copy()
        return mask_clean

    def update_detection_params(self, t_val, e_val, cooldown, trg_mode):
        self.current_t_sens = t_val
        if hasattr(self, 'detector'): self.detector.update_params(t_val, e_val, cooldown)
        self.trigger_mode = trg_mode

    def set_mode(self, mode):
        self.mode = mode
        self.log_signal.emit(f">>> MODE: {mode}")

    def update_align_params(self, dx=0, dy=0, set_scale=None, set_angle=None, toggle_checker=False):
        try:
            nx = self.algo_align.x + dx;
            ny = self.algo_align.y + dy
            ns = self.algo_align.scale if not set_scale else set_scale
            na = self.algo_align.angle if set_angle is None else set_angle
            if toggle_checker: self.checker_mode = not self.checker_mode
            self.algo_align.update_params(x=nx, y=ny, scale=ns, angle=na, opacity=0.5)
        except:
            pass

    def update_depth_rotation(self, dx, dy):
        self.depth_yaw += dx * 0.02;
        self.depth_pitch += dy * 0.02
        self.depth_pitch = np.clip(self.depth_pitch, -1.5, 1.5)

    def rotate_image(self, image, angle):
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(image, M, (w, h))

    def start_recording(self):
        if self.is_recording: return
        self.is_recording = True
        self.rec_frame_id = 0
        self.lost_frames = 0

        ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.rec_dir = os.path.join("auto_captures", f"target_{ts_str}")

        for sub in ["visual", "thermal", "event_hq"]:
            os.makedirs(os.path.join(self.rec_dir, sub), exist_ok=True)

        try:
            with open(os.path.join(self.rec_dir, "align_params.json"), "w") as f:
                json.dump({"x": self.algo_align.x, "y": self.algo_align.y, "scale": self.algo_align.scale,
                           "angle": self.algo_align.angle}, f)
        except:
            pass

        self.io_queue.put(("INIT_CSV", self.rec_dir))
        self.log_signal.emit(f"[REC] CHUNK START: {ts_str}")

    def stop_recording(self):
        if not self.is_recording: return
        self.is_recording = False

        # 🟢 如果中途丢失导致不足 60 帧，直接毁尸灭迹
        if self.rec_frame_id < self.CHUNK_SIZE:
            self.log_signal.emit(
                f"[REC] DISCARDED: Incomplete chunk ({self.rec_frame_id}/{self.CHUNK_SIZE}). Destroying...")
            self.io_queue.put(("DESTROY_DIR", self.rec_dir))

    def get_smart_crop(self, image, box, out_size=(120, 120)):
        bx, by, bw, bh = box;
        ih, iw = image.shape[:2]
        pad_x, pad_y = int(bw * 0.5), int(bh * 0.5)
        x1, y1 = max(0, bx - pad_x), max(0, by - pad_y)
        x2, y2 = min(iw, bx + bw + pad_x), min(ih, by + bh + pad_y)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0: return np.zeros((out_size[1], out_size[0], 3), dtype=np.uint8)
        ch, cw = crop.shape[:2];
        tw, th = out_size
        scale = min(tw / cw, th / ch)
        new_w, new_h = int(cw * scale), int(ch * scale)
        resized = cv2.resize(crop, (new_w, new_h))
        canvas = np.zeros((th, tw, 3), dtype=np.uint8)
        dx, dy = (tw - new_w) // 2, (th - new_h) // 2
        canvas[dy:dy + new_h, dx:dx + new_w] = resized
        return canvas

    def render_3d_depth_fusion(self, thermal_raw, event_mask, width=640, height=480):
        grid_w, grid_h = 480, 320
        t_small = cv2.resize(thermal_raw, (grid_w, grid_h), interpolation=cv2.INTER_CUBIC)
        z_therm = t_small.astype(np.float32)
        z_therm = (z_therm - z_therm.min()) / (z_therm.max() - z_therm.min() + 1e-5)
        e_small = cv2.resize(event_mask, (grid_w, grid_h), interpolation=cv2.INTER_NEAREST)

        raw_mask = np.zeros_like(z_therm)
        raw_mask[(z_therm > 0.15) | (e_small > 0)] = 1.0
        fill_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        body_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, fill_kernel)
        body_mask = cv2.GaussianBlur(body_mask, (15, 15), 0)

        z_final = (body_mask * 0.5) + (z_therm * body_mask * 0.5)
        if self.depth_inverse:
            z_final = 1.0 - z_final
            z_final[body_mask < 0.1] = 0

        x = np.linspace(-1.6, 1.6, grid_w);
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

        u = (x_proj * width * 0.8 + width / 2).astype(np.int32)
        v = (y_proj * width * 0.8 + height / 2).astype(np.int32)

        img = np.zeros((height, width, 3), dtype=np.uint8)
        colors = cv2.applyColorMap((z_final.reshape(-1) * 255).astype(np.uint8), cv2.COLORMAP_MAGMA).reshape(-1, 3)
        valid = (u >= 0) & (u < width - 1) & (v >= 0) & (v < height - 1)

        if np.any(valid):
            z_vals = points_rot[valid, 2]
            sort_idx = np.argsort(z_vals)[::-1]
            u_val = u[valid][sort_idx];
            v_val = v[valid][sort_idx];
            c_val = colors[valid][sort_idx]
            img[v_val, u_val] = c_val;
            img[v_val + 1, u_val] = c_val
            img[v_val, u_val + 1] = c_val;
            img[v_val + 1, u_val + 1] = c_val

        return img

    def run(self):
        self.log_signal.emit("[CORE] ASYNC ENGINE STARTED")

        while self.running:
            try:
                if len(self.q_vis) == 0: self.msleep(1); continue
                while len(self.q_vis) > 1: self.q_vis.popleft()
                v_ts, v_fid, v_raw = self.q_vis.popleft()

                v_corr = self.algo_vign.process(v_raw)

                e_mask_fast = self.algo_evt.process_fast(v_corr)
                e_mask_clean = self.generate_robust_events(v_corr)

                v_bg = cv2.cvtColor(v_corr, cv2.COLOR_GRAY2BGR)
                e_disp = np.zeros_like(v_bg)
                e_disp[e_mask_fast > 0] = [0, 255, 0]

                if len(self.q_therm) > 0:
                    while len(self.q_therm) > 1: self.q_therm.pop()
                    t_ts, t_fid, t_raw = self.q_therm.pop()
                    self.cache_t_raw = t_raw
                else:
                    t_ts = v_ts

                tx, ty, tw, th, angle, _ = self.algo_align.get_transform_params()
                raw_f = self.cache_t_raw.astype(np.float32)
                temp_c = raw_f / 64.0 - 273.15

                t_temp_scaled = cv2.resize(temp_c, (tw, th))
                t_temp_rot = self.rotate_image(t_temp_scaled, angle)
                aligned_temp = np.full((VIS_H, VIS_W), -50.0, dtype=np.float32)

                x1 = max(0, tx);
                y1 = max(0, ty)
                x2 = min(VIS_W, tx + tw);
                y2 = min(VIS_H, ty + th)
                vw, vh = x2 - x1, y2 - y1

                aligned_color = np.zeros_like(v_bg)
                if vw > 0 and vh > 0:
                    ox, oy = x1 - tx, y1 - ty
                    if oy + vh <= t_temp_rot.shape[0] and ox + vw <= t_temp_rot.shape[1]:
                        aligned_temp[y1:y2, x1:x2] = t_temp_rot[oy:oy + vh, ox:ox + vw]

                elapsed_time = time.time() - self.engine_start_time
                is_calibrating = elapsed_time < 5.0

                if is_calibrating:
                    self.bg_thermal_map = np.maximum(self.bg_thermal_map, aligned_temp)
                    det_state, target_box = "NONE", None
                else:
                    clean_aligned_temp = aligned_temp.copy()
                    bg_hot_mask = self.bg_thermal_map > (self.current_t_sens - 2.0)
                    clean_aligned_temp[bg_hot_mask] = -50.0
                    det_state, target_box = self.detector.detect(clean_aligned_temp, e_mask_fast)

                if target_box is not None:
                    self.last_target_box = target_box

                triggered = False
                if target_box is not None:
                    if self.trigger_mode == "BOTH" and det_state == "BOTH":
                        triggered = True
                    elif self.trigger_mode == "THERMAL" and det_state in ["THERMAL", "BOTH"]:
                        triggered = True
                    elif self.trigger_mode == "EVENT" and det_state in ["EVENT", "BOTH"]:
                        triggered = True
                    elif self.trigger_mode == "ANY" and det_state != "NONE":
                        triggered = True

                if triggered:
                    self.lost_frames = 0
                    self.start_recording()
                else:
                    if self.is_recording:
                        self.lost_frames += 1
                        if self.lost_frames > self.TOLERANCE_FRAMES:
                            self.stop_recording()

                t_min, t_max = np.min(temp_c), np.max(temp_c)
                if t_max - t_min < 2.0: t_max = t_min + 5.0
                t_norm = ((temp_c - t_min) / (t_max - t_min) * 255).astype(np.uint8)
                t_color = cv2.applyColorMap(t_norm, cv2.COLORMAP_JET)
                t_color_scaled = cv2.resize(t_color, (tw, th))
                t_color_rot = self.rotate_image(t_color_scaled, angle)

                final_fusion = v_bg.copy()
                if vw > 0 and vh > 0:
                    ox, oy = x1 - tx, y1 - ty
                    if oy + vh <= t_color_rot.shape[0] and ox + vw <= t_color_rot.shape[1]:
                        t_crop = t_color_rot[oy:oy + vh, ox:ox + vw]
                        aligned_color[y1:y2, x1:x2] = t_crop
                        v_crop = v_bg[y1:y2, x1:x2]
                        if self.checker_mode:
                            mask = ((np.indices((vh, vw))[0] // 32 + np.indices((vh, vw))[1] // 32) % 2 == 0)
                            blend = np.where(np.dstack([mask] * 3), v_crop, t_crop)
                        else:
                            blend = cv2.addWeighted(v_crop, 0.6, t_crop, 0.7, 0)
                        final_fusion[y1:y2, x1:x2] = blend
                        if self.mode != "LOCKED":
                            cv2.rectangle(final_fusion, (x1, y1), (x2, y2), (255, 255, 0), 1)

                if is_calibrating:
                    cv2.putText(final_fusion, f"BG CALIBRATION: {5.0 - elapsed_time:.1f}s", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

                if self.is_recording and self.last_target_box is not None:
                    bx, by, bw, bh = self.last_target_box
                    pad = int(max(bw, bh) * 0.3)
                    side = max(bw, bh) + pad * 2
                    cx, cy = bx + bw // 2, by + bh // 2

                    x1_c = cx - side // 2;
                    y1_c = cy - side // 2
                    x2_c = x1_c + side;
                    y2_c = y1_c + side

                    x1_safe = max(0, x1_c);
                    y1_safe = max(0, y1_c)
                    x2_safe = min(VIS_W, x2_c);
                    y2_safe = min(VIS_H, y2_c)

                    if x2_safe > x1_safe and y2_safe > y1_safe:
                        vis_valid = final_fusion[y1_safe:y2_safe, x1_safe:x2_safe]
                        therm_valid = aligned_color[y1_safe:y2_safe, x1_safe:x2_safe]
                        evt_valid = e_mask_clean[y1_safe:y2_safe, x1_safe:x2_safe]

                        cvs_vis = np.zeros((side, side, 3), dtype=np.uint8)
                        cvs_therm = np.zeros((side, side, 3), dtype=np.uint8)
                        cvs_evt = np.zeros((side, side), dtype=np.uint8)

                        sy = y1_safe - y1_c;
                        sx = x1_safe - x1_c
                        eh = y2_safe - y1_safe;
                        ew = x2_safe - x1_safe

                        cvs_vis[sy:sy + eh, sx:sx + ew] = vis_valid
                        cvs_therm[sy:sy + eh, sx:sx + ew] = therm_valid
                        cvs_evt[sy:sy + eh, sx:sx + ew] = evt_valid

                        TRAIN_SIZE = 200
                        crop_vis = cv2.resize(cvs_vis, (TRAIN_SIZE, TRAIN_SIZE), interpolation=cv2.INTER_AREA)
                        crop_therm = cv2.resize(cvs_therm, (TRAIN_SIZE, TRAIN_SIZE), interpolation=cv2.INTER_AREA)
                        crop_evt = cv2.resize(cvs_evt, (TRAIN_SIZE, TRAIN_SIZE), interpolation=cv2.INTER_NEAREST)

                        self.io_queue.put(
                            ("WRITE_FRAME", self.rec_dir, self.rec_frame_id, crop_vis, crop_therm, crop_evt, v_ts,
                             t_ts))
                        self.rec_frame_id += 1

                        # 🟢 【核心分块逻辑】：满 60 帧，立刻封箱，无缝开启下一段录像！
                        if self.rec_frame_id == self.CHUNK_SIZE:
                            self.log_signal.emit(f"[REC] CHUNK COMPLETE (60 Frames): {os.path.basename(self.rec_dir)}")
                            self.is_recording = False
                            # 下一次循环如果 triggered 仍然为 True，会自动调用 start_recording，实现 0 掉帧连续接力！

                t_color_disp = aligned_color

                if target_box is not None:
                    bx, by, bw, bh = target_box
                    box_color = (0, 255, 0) if det_state == "THERMAL" else (255, 200, 0) if det_state == "EVENT" else (
                        0, 0, 255)
                    tgt_text = f"[{det_state}] LOCKED" + (" REC" if self.is_recording else "")

                    def draw_tactical_box(img, x, y, w, h, color):
                        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
                        l = max(10, min(w, h) // 4);
                        t = 3
                        for start, end in [((x, y), (x + l, y)), ((x, y), (x, y + l)), ((x + w, y), (x + w - l, y)),
                                           ((x + w, y), (x + w, y + l)),
                                           ((x, y + h), (x + l, y + h)), ((x, y + h), (x, y + h - l)),
                                           ((x + w, y + h), (x + w - l, y + h)), ((x + w, y + h), (x + w, y + h - l))]:
                            cv2.line(img, start, end, color, t)
                        cv2.putText(img, tgt_text, (x, max(15, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                        cv2.putText(img, tgt_text, (x, max(15, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    draw_tactical_box(final_fusion, bx, by, bw, bh, box_color)
                    draw_tactical_box(e_disp, bx, by, bw, bh, box_color)
                    draw_tactical_box(t_color_disp, bx, by, bw, bh, box_color)
                    roi_display = self.get_smart_crop(v_bg, target_box, (120, 120))
                else:
                    roi_display = np.zeros((120, 120, 3), dtype=np.uint8)
                    cv2.putText(roi_display, "NO TGT", (30, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

                if self.mode == "LOCKED" and vw > 0 and vh > 0:
                    final_fusion, e_disp, t_color_disp = [img[y1:y2, x1:x2] for img in
                                                          [final_fusion, e_disp, t_color_disp]]

                depth_display = self.render_3d_depth_fusion(self.cache_t_raw, e_mask_fast, 640, 480)

                self.fps_cnt += 1
                if time.time() - self.fps_timer >= 1.0:
                    self.curr_fps = self.fps_cnt;
                    self.fps_cnt = 0;
                    self.fps_timer = time.time()

                info = {"fps": self.curr_fps, "mode": "REC ●" if self.is_recording else "SCAN",
                        "rec": "REC" if self.is_recording else ""}
                self.update_signal.emit(final_fusion, t_color_disp, e_disp, roi_display, depth_display, info)

            except Exception as e:
                print(f"Sync: {e}")
                self.msleep(10)

    def stop(self):
        self.stop_recording()
        self.running = False
        self.wait()