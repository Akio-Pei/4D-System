# =========================================================================
# 文件名: comms_engine.py
# 描述: 战术终端后端引擎 - 负责通信、协议解析、时间对齐、图像融合算法
# 作者: J.H. Pei
# =========================================================================

import socket
import struct
import numpy as np
import cv2
import time
from collections import deque
from PyQt6.QtCore import QThread, pyqtSignal

# --- [配置区] ---
HOST = '0.0.0.0'
PORT_VIDEO = 8888  # OV9281 端口
PORT_THERMAL = 8889  # Tiny1-C 端口

# 图像尺寸配置
THERMAL_W, THERMAL_H = 256, 192  # Tiny1-C 原始分辨率
THERMAL_SIZE = THERMAL_W * THERMAL_H * 2  # 16bit 数据大小

# 队列缓存长度
MAX_QUEUE_SIZE = 100


class DataReceiver(QThread):
    """
    [类功能]: 单一通道的数据接收器
    [逻辑]: 这是一个死循环线程，只负责从网线里把数据读出来，塞进队列，不处理业务。
    """
    log_signal = pyqtSignal(str)  # 日志信号

    def __init__(self, port, queue, mode):
        super().__init__()
        self.port = port
        self.queue = queue
        self.mode = mode  # "video" 或 "thermal"
        self.running = True

    def recv_all(self, sock, count):
        """
        [工具函数]: 解决 TCP 粘包问题
        确保读够 count 个字节才返回，否则图像会花屏。
        """
        buf = b''
        while count:
            try:
                newbuf = sock.recv(count)
                if not newbuf: return None
                buf += newbuf
                count -= len(newbuf)
            except:
                return None
        return buf

    def run(self):
        # 1. 创建套接字
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            sock.bind((HOST, self.port))
            sock.listen(1)
            self.log_signal.emit(f"[{self.mode.upper()}] PORT {self.port} LISTENING...")
        except Exception as e:
            self.log_signal.emit(f"[ERR] BIND FAIL: {e}")
            return

        while self.running:
            try:
                # 2. 等待设备连接 (阻塞式)
                conn, addr = sock.accept()
                self.log_signal.emit(f"[{self.mode.upper()}] LINK ACTIVE: {addr[0]}")

                while self.running:
                    # --- 第一步：读包头 (16字节) ---
                    # 格式: Timestamp(8) + Size(4) + FrameID(4)
                    header = self.recv_all(conn, 16)
                    if not header: break
                    ts, size, fid = struct.unpack("=QII", header)

                    # --- 第二步：读负载数据 ---
                    payload = self.recv_all(conn, size)
                    if not payload: break

                    # --- 第三步：初步解码 ---
                    data_item = None
                    if self.mode == "video":
                        # OV9281: JPEG -> 灰度图
                        # 传输JPEG是为了省带宽，这里解压成矩阵
                        frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                        if frame is not None: data_item = frame

                    elif self.mode == "thermal":
                        # Tiny1-C: Raw Bytes -> 温度矩阵(uint16)
                        if size == THERMAL_SIZE:
                            data_item = np.frombuffer(payload, dtype=np.uint16).reshape((THERMAL_H, THERMAL_W))

                    # --- 第四步：入队 ---
                    if data_item is not None:
                        if len(self.queue) >= MAX_QUEUE_SIZE:
                            self.queue.popleft()  # 挤掉旧数据
                        self.queue.append((ts, fid, data_item))

            except Exception as e:
                self.log_signal.emit(f"[{self.mode}] LINK DROPPED: {e}")
            finally:
                if 'conn' in locals(): conn.close()

    def stop(self):
        self.running = False
        self.wait()


class SyncEngine(QThread):
    """
    [类功能]: 核心大脑 - 负责同步、融合、算法处理
    [逻辑]: 自动判断当前有几个摄像头在线，并决定输出什么画面。
    """
    # 信号: 主画面, 副画面, 信息字典
    update_signal = pyqtSignal(np.ndarray, np.ndarray, dict)
    log_signal = pyqtSignal(str)

    def __init__(self, q_video, q_thermal):
        super().__init__()
        self.q_video = q_video
        self.q_thermal = q_thermal
        self.running = True
        self.fusion_mode = "EDGE"  # 默认融合模式: EDGE (边缘) 或 CHECKER (棋盘)

        # --- [关键] 标定矩阵 (Homography) ---
        # 等你把铜板拍好了，算出单应性矩阵，填在这里！
        # 现在暂时用单位矩阵 (Identity)，假设两个相机是完全平行的 (实际上肯定不是)
        self.H_matrix = np.eye(3, dtype=np.float32)

        # 预生成一个黑屏，没信号时用
        self.blank = np.zeros((THERMAL_H, THERMAL_W, 3), dtype=np.uint8)

    def set_fusion_style(self, style):
        """切换融合风格: 'EDGE' 或 'CHECKER'"""
        self.fusion_mode = style
        self.log_signal.emit(f"[SYS] FUSION MODE SWITCHED TO: {style}")

    def run(self):
        self.log_signal.emit("[CORE] SYNC ENGINE STARTED...")

        while self.running:
            # 检查两个队列是否有数据
            has_t = len(self.q_thermal) > 0
            has_v = len(self.q_video) > 0

            # === 情况 A: 双路都有 (执行对齐+融合) ===
            if has_t and has_v:
                self.process_fusion()

            # === 情况 B: 只有热成像 (单路显示) ===
            elif has_t:
                self.process_single("thermal")

            # === 情况 C: 只有可见光 (单路显示) ===
            elif has_v:
                self.process_single("video")

            # === 情况 D: 没信号 ===
            else:
                self.msleep(10)  # 休息，降低CPU占用

    def process_single(self, mode):
        """处理单路信号"""
        if mode == "thermal":
            ts, fid, data = self.q_thermal.popleft()
            # 归一化并转伪彩
            norm = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
            color = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)

            info = {"mode": "SINGLE_IR", "temp": self.get_center_temp(data), "fid": fid}
            self.update_signal.emit(color, self.blank, info)

        elif mode == "video":
            ts, fid, data = self.q_video.popleft()
            # 转成彩色方便显示
            color = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)

            info = {"mode": "SINGLE_VIS", "fid": fid}
            self.update_signal.emit(color, self.blank, info)

    def process_fusion(self):
        """处理双路融合 (核心算法)"""
        # 1. 取出最新的一帧热成像 (基准)
        t_ts, t_fid, t_data = self.q_thermal.popleft()

        # 2. 寻找匹配的视频帧 (简单版：取最新)
        # 实际应该遍历队列找时间戳差值最小的，这里为了代码清晰简化了
        v_ts, v_fid, v_data = self.q_video[-1]
        self.q_video.clear()  # 清空视频队列，防止堆积

        # 3. 图像预处理
        # 3.1 热成像 -> 彩色底图
        t_norm = cv2.normalize(t_data, None, 0, 255, cv2.NORM_MINMAX)
        t_color = cv2.applyColorMap(t_norm.astype(np.uint8), cv2.COLORMAP_JET)

        # 3.2 可见光 -> 空间对齐 (Warp)
        # 9281分辨率高，Tiny1C分辨率低，且视角不同
        # 必须先把可见光缩放/变形到和热成像一样大 (256x192)
        # TODO: 以后这里要用 cv2.warpPerspective(v_data, self.H_matrix, ...)
        v_aligned = cv2.resize(v_data, (THERMAL_W, THERMAL_H))

        # 4. 根据模式进行融合
        fused_img = None

        if self.fusion_mode == "EDGE":
            # --- 模式 A: 边缘融合 (Edge Fusion) ---
            # 提取可见光边缘
            edges = cv2.Canny(v_aligned, 100, 200)
            # 将边缘染成亮绿色
            edge_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            edge_color[:] = (0, 0, 0)  # 黑色背景
            edge_color[edges > 0] = (0, 255, 0)  # 绿色线条

            # 叠加: 热成像(1.0) + 绿色边缘(0.8)
            fused_img = cv2.addWeighted(t_color, 1.0, edge_color, 0.8, 0)

        elif self.fusion_mode == "CHECKER":
            # --- 模式 B: 棋盘格融合 (Checkerboard) ---
            # 生成棋盘格掩码 (32x32 像素一个格子)
            block_size = 32
            rows, cols = THERMAL_H, THERMAL_W
            # 利用 numpy 广播生成网格
            # (y // 32 + x // 32) % 2 == 0 为黑格，== 1 为白格
            x_idx = np.arange(cols) // block_size
            y_idx = np.arange(rows) // block_size
            mask = (x_idx[None, :] + y_idx[:, None]) % 2  # 0或1
            mask = mask.astype(np.uint8)

            # 扩展到3通道
            mask_3c = cv2.merge([mask, mask, mask])

            # 可见光部分 (转彩)
            v_color = cv2.cvtColor(v_aligned, cv2.COLOR_GRAY2BGR)

            # 融合: mask为1的地方显示可见光，mask为0的地方显示热成像
            fused_img = t_color * (1 - mask_3c) + v_color * mask_3c

        # 5. 打包发送
        sync_diff = (int(t_ts) - int(v_ts)) / 1000.0  # ms
        info = {
            "mode": f"LOCKED [{self.fusion_mode}]",
            "sync_diff": sync_diff,
            "temp": self.get_center_temp(t_data),
            "t_fid": t_fid
        }

        # 副画面显示变换后的可见光，方便对比
        self.update_signal.emit(fused_img, v_aligned, info)

    def get_center_temp(self, raw):
        """计算中心点温度"""
        # Tiny1-C 原始值转摄氏度公式
        return (float(raw[THERMAL_H // 2, THERMAL_W // 2]) / 64.0) - 273.15

    def stop(self):
        self.running = False
        self.wait()