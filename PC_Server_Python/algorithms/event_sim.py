import cv2
import numpy as np


class PseudoEventGen:
    def __init__(self, width, height, threshold=20):
        self.w = width
        self.h = height

        # === 快速预览版状态 (Lite) ===
        self.prev_fast = None

        # === 满血录制版状态 (HQ) ===
        self.prev_hq = None
        # 去噪核：用于过滤孤立的事件噪点
        self.hq_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def process_fast(self, current_frame):
        """
        [预览流] 极速线性差分
        用途：实时屏幕显示、运动检测触发
        特点：计算极快(<1ms)，但噪点较多
        """
        curr = current_frame.astype(np.int16)

        if self.prev_fast is None:
            self.prev_fast = curr
            return np.zeros((self.h, self.w), dtype=np.uint8)

        diff = curr - self.prev_fast
        self.prev_fast = curr

        mask = np.zeros((self.h, self.w), dtype=np.uint8)
        mask[np.abs(diff) > 25] = 255
        return mask

    def process_hq(self, current_frame):
        """
        [生产流] 满血对数域模拟 (Full-Blooded)
        用途：HexPlane 训练数据生成
        特点：
        1. Log 变换：模拟真实事件相机对光照的非线性响应
        2. 物理阈值：模拟 C 参数
        3. 形态学去噪：提供干净的边缘特征
        """
        # 转为浮点并加1防止log(0)
        curr_float = current_frame.astype(np.float32) + 1.0

        if self.prev_hq is None:
            self.prev_hq = curr_float
            return np.zeros((self.h, self.w), dtype=np.uint8)

        # 1. 对数域变换 (Log Domain)
        # 真实事件相机感知的是亮度的对比度变化 (Log差)，而不是绝对差
        curr_log = np.log(curr_float)
        prev_log = np.log(self.prev_hq)

        # 2. 计算变化率
        diff = curr_log - prev_log
        self.prev_hq = curr_float

        # 3. 物理阈值 (Sensitivity)
        # 0.15 大约对应 15% 的亮度变化，是常用的事件相机阈值
        evt_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        evt_mask[np.abs(diff) > 0.15] = 255

        # 4. 强力去噪 (Denoise)
        # 消除传感器热噪声产生的孤立白点，保证HexPlane学到的是物体轮廓
        clean_mask = cv2.morphologyEx(evt_mask, cv2.MORPH_OPEN, self.hq_kernel)

        return clean_mask

    def reset(self):
        """ 每次开始录制时重置状态，防止上一段录制的残影 """
        self.prev_fast = None
        self.prev_hq = None