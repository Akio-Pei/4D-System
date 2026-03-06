import cv2
import numpy as np
from collections import deque


class PseudoEventGen:
    def __init__(self, width, height):
        self.w = width
        self.h = height

        # [核心优化] N 帧延迟队列
        # 我们抛弃所有花里胡哨的背景建模，回归你初版最稳定的“帧差法”。
        # 这里的 6 代表比较当前帧与 6 帧前的画面。
        # 延迟越大，越能捕捉极慢的动作；且完全不会增加噪点！
        self.delay_frames = 6
        self.fast_queue = deque(maxlen=self.delay_frames)
        self.hq_queue = deque(maxlen=self.delay_frames)

        # 强力物理去噪核：专门消灭孤立的雪花噪点
        self.noise_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # 运动余辉图 (Time Surface)
        self.trail_fast = np.zeros((self.h, self.w), dtype=np.float32)
        self.trail_hq = np.zeros((self.h, self.w), dtype=np.float32)

    def process_fast(self, current_frame):
        # 1. 空间降噪：提前抹平传感器的高频底噪
        curr_blur = cv2.GaussianBlur(current_frame, (5, 5), 0).astype(np.int16)

        # 将当前帧塞入队列
        self.fast_queue.append(curr_blur)

        # 如果队列没满（刚启动的前几帧），先不输出
        if len(self.fast_queue) < self.delay_frames:
            return np.zeros((self.h, self.w), dtype=np.uint8)

        # 2. 延迟帧差：当前帧 - 6帧前的老画面
        # 完美兼顾了初版的“零噪点”和“慢动作捕捉”！
        diff = np.abs(curr_blur - self.fast_queue[0])

        # 3. 干净利落的硬阈值截断（像初版一样抗干扰）
        mask = np.zeros((self.h, self.w), dtype=np.uint8)
        mask[diff > 20] = 255  # 20 是过滤环境微光闪烁的绝佳阈值

        # 4. 形态学开运算：把漏网的单像素噪点直接抹除
        clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.noise_kernel)

        # 5. DVS 拖尾余辉：让线框变实心面，利于目标检测器稳定红框
        # 每次衰减 50，大约 5 帧后残影消失，非常干脆
        self.trail_fast = np.maximum(0, self.trail_fast - 50.0)
        self.trail_fast[clean_mask > 0] = 255.0

        final_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        final_mask[self.trail_fast > 50] = 255

        return final_mask

    def process_hq(self, current_frame):
        # 生产环境的 Log 域满血版，同样适用该延迟队列逻辑
        curr_float = current_frame.astype(np.float32) + 1.0
        curr_blur = cv2.GaussianBlur(curr_float, (5, 5), 0)
        curr_log = np.log(curr_blur)

        self.hq_queue.append(curr_log)
        if len(self.hq_queue) < self.delay_frames:
            return np.zeros((self.h, self.w), dtype=np.uint8)

        diff = np.abs(curr_log - self.hq_queue[0])

        mask = np.zeros((self.h, self.w), dtype=np.uint8)
        # 真实事件相机 C 参数阈值
        mask[diff > 0.15] = 255

        clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.noise_kernel)

        self.trail_hq = np.maximum(0, self.trail_hq - 50.0)
        self.trail_hq[clean_mask > 0] = 255.0

        final_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        final_mask[self.trail_hq > 50] = 255

        return final_mask

    def reset(self):
        # 坚决不粗暴清空，防止切换检测状态时出现“闪黑”
        pass