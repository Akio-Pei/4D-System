# =========================================================================
# 文件名: tactical_ui.py
# 描述: 战术指挥终端 - 负责界面绘制、用户交互、双语显示
# 作者: J.H. Pei
# =========================================================================

import sys
import time
from collections import deque
import numpy as np
import cv2
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QTextEdit,
                             QGroupBox, QGridLayout)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap, QFont, QPainter, QColor, QPen, QIcon

# 导入刚才写的后端
from comms_engine import DataReceiver, SyncEngine, PORT_VIDEO, PORT_THERMAL

# --- 战术翻译字典 ---
TRANSLATIONS = {
    "EN": {
        "title": "TE-GS Multimodal Reconstruction System",
        "status": "STATUS:",
        "sync": "SYNC LATENCY:",
        "online": "SYSTEM ONLINE",
        "offline": "SYSTEM OFFLINE",
        "btn_link": "INITIALIZE DATALINK",
        "btn_active": "LINK ACTIVE",
        "btn_mode": "SWITCH FUSION: ",  # 融合切换
        "hud_main": "PRIMARY / FUSION",
        "hud_sub": "SECONDARY / RAW",
        "waiting": "AWAITING SIGNAL..."
    },
    "CN": {
        "title": "TE-GS 多模态夜间重建系统",
        "status": "系统状态:",
        "sync": "同步延迟:",
        "online": "系统在线",
        "offline": "系统离线",
        "btn_link": "初始化数据链路",
        "btn_active": "链路已激活",
        "btn_mode": "切换融合模式: ",
        "hud_main": "主视图 / 融合",
        "hud_sub": "副视图 / 原始",
        "waiting": "等待信号输入..."
    }
}

# --- 军工样式表 ---
MILITARY_STYLE = """
QMainWindow { background-color: #080808; }
QWidget { color: #cccccc; font-family: 'Consolas', monospace; }
QGroupBox { border: 1px solid #333; margin-top: 20px; font-weight: bold; color: #00ff00; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
QPushButton { background-color: #112211; border: 1px solid #224422; color: #00ee00; padding: 8px; font-weight: bold; }
QPushButton:hover { background-color: #225522; border-color: #00ff00; color: #ffffff; }
QTextEdit { background-color: #000; border: 1px solid #333; color: #00ff00; font-size: 11px; }
QLabel#Val { color: #00ff00; font-weight: bold; font-size: 14px; }
QLabel#Warn { color: #ff3333; font-weight: bold; }
"""


class HUDDisplay(QLabel):
    """
    [自定义控件]: 战术显示屏
    重写 paintEvent 来绘制十字准星、角标和 OSD 文字
    """

    def __init__(self, title_key, color_hex="#00ff00", lang="CN"):
        super().__init__()
        self.title_key = title_key
        self.accent_color = QColor(color_hex)
        self.lang = lang
        self.setMinimumSize(480, 360)
        self.setStyleSheet("background-color: #000; border: 1px solid #333;")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_cache = {}

    def update_frame(self, cv_img, info=None):
        if info: self.info_cache = info

        # 格式转换: OpenCV(BGR) -> Qt(RGB)
        if len(cv_img.shape) == 2:
            h, w = cv_img.shape
            fmt = QImage.Format.Format_Grayscale8
        else:
            h, w, ch = cv_img.shape
            fmt = QImage.Format.Format_RGB888
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        q_img = QImage(cv_img.data, w, h, w * (3 if len(cv_img.shape) > 2 else 1), fmt)
        self.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.width(), self.height(), Qt.AspectRatioMode.KeepAspectRatio))
        self.update()  # 触发重绘

    def paintEvent(self, event):
        super().paintEvent(event)

        # 获取当前画笔
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        t = TRANSLATIONS[self.lang]

        # 如果没有图像，显示等待文字
        if not self.pixmap():
            p.setPen(QColor(100, 100, 100))
            p.setFont(QFont("Consolas", 12))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, t["waiting"])
            return

        # 1. 绘制战术角标 (Bracket)
        pen = QPen(self.accent_color)
        pen.setWidth(2)
        p.setPen(pen)
        gap = 20
        # 左上
        p.drawLine(10, 10 + gap, 10, 10);
        p.drawLine(10, 10, 10 + gap, 10)
        # 右下
        p.drawLine(w - 10 - gap, h - 10, w - 10, h - 10);
        p.drawLine(w - 10, h - 10, w - 10, h - 10 - gap)

        # 2. 绘制 OSD 信息
        p.setPen(QColor(255, 255, 255))
        p.setFont(QFont("Consolas", 9))

        # 标题
        p.drawText(20, 25, t[self.title_key])

        # 动态数据
        if self.info_cache:
            # 显示模式
            if "mode" in self.info_cache:
                mode_str = self.info_cache["mode"]
                p.setFont(QFont("Consolas", 10, QFont.Weight.Bold))

                # 根据模式变色
                if "LOCKED" in mode_str:
                    # 融合锁定: 绿底黑字
                    p.fillRect(w // 2 - 60, h - 30, 120, 20, QColor(0, 255, 0))
                    p.setPen(QColor(0, 0, 0))
                else:
                    # 单路: 橙色文字
                    p.setPen(QColor(255, 165, 0))

                p.drawText(w // 2 - 50, h - 15, mode_str)

            # 恢复白色画其他
            p.setPen(QColor(255, 255, 255))

            # 温度
            if "temp" in self.info_cache:
                val = self.info_cache['temp']
                p.drawText(20, h - 20, f"TEMP: {val:.1f}°C")

            # 帧号
            if "fid" in self.info_cache:
                p.drawText(w - 100, 25, f"#{self.info_cache['fid']}")


class TacticalTerminal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.lang = "CN"  # 默认中文
        self.q_video = deque(maxlen=100)
        self.q_thermal = deque(maxlen=100)
        self.current_fusion = "EDGE"  # 当前融合模式

        self.init_ui()
        self.setStyleSheet(MILITARY_STYLE)

    def init_ui(self):
        t = TRANSLATIONS[self.lang]
        self.setWindowTitle(t["title"])
        self.resize(1280, 800)

        self.setWindowIcon(QIcon("images/logo.ico"))

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # --- 顶部状态栏 ---
        top = QHBoxLayout()

        self.btn_lang = QPushButton("[ EN / CN ]")
        self.btn_lang.setFixedWidth(100)
        self.btn_lang.clicked.connect(self.toggle_lang)

        self.lbl_stat = QLabel(t["offline"])
        self.lbl_stat.setObjectName("Warn")  # 红色

        self.lbl_sync = QLabel("-- ms")
        self.lbl_sync.setObjectName("Val")  # 绿色

        top.addWidget(self.btn_lang)
        top.addSpacing(20)
        top.addWidget(QLabel(t["status"]))
        top.addWidget(self.lbl_stat)
        top.addSpacing(40)
        top.addWidget(QLabel(t["sync"]))
        top.addWidget(self.lbl_sync)
        top.addStretch()
        top.addWidget(QLabel("J.H. PEI | HEXPLANE PROTO"))

        # --- 中间屏幕 ---
        screens = QHBoxLayout()
        self.hud_main = HUDDisplay("hud_main", "#00ff00", self.lang)
        self.hud_sub = HUDDisplay("hud_sub", "#008800", self.lang)
        screens.addWidget(self.hud_main, stretch=6)
        screens.addWidget(self.hud_sub, stretch=4)

        # --- 底部控制 ---
        bot = QHBoxLayout()

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(120)

        ctrl_grp = QGroupBox("COMMAND")
        ctrl_grp.setFixedWidth(320)
        grid = QGridLayout()

        self.btn_link = QPushButton(t["btn_link"])
        self.btn_link.clicked.connect(self.start_link)

        # 融合切换按钮
        self.btn_mode = QPushButton(t["btn_mode"] + "EDGE")
        self.btn_mode.clicked.connect(self.toggle_fusion_mode)
        self.btn_mode.setEnabled(False)  # 连接后才可用

        grid.addWidget(self.btn_link, 0, 0)
        grid.addWidget(self.btn_mode, 1, 0)
        ctrl_grp.setLayout(grid)

        bot.addWidget(self.log_box, 7)
        bot.addWidget(ctrl_grp, 3)

        layout.addLayout(top)
        layout.addLayout(screens, 8)
        layout.addLayout(bot, 2)

    def log(self, msg):
        t_str = time.strftime("%H:%M:%S")
        self.log_box.append(f"[{t_str}] {msg}")
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

    def start_link(self):
        t = TRANSLATIONS[self.lang]
        self.btn_link.setEnabled(False)
        self.btn_link.setText(t["btn_active"])
        self.btn_mode.setEnabled(True)

        self.lbl_stat.setText(t["online"])
        self.lbl_stat.setObjectName("Val")  # 变绿
        self.lbl_stat.style().unpolish(self.lbl_stat)
        self.lbl_stat.style().polish(self.lbl_stat)

        # 启动后端线程
        self.th_v = DataReceiver(PORT_VIDEO, self.q_video, "video")
        self.th_t = DataReceiver(PORT_THERMAL, self.q_thermal, "thermal")
        self.th_v.log_signal.connect(self.log)
        self.th_t.log_signal.connect(self.log)
        self.th_v.start()
        self.th_t.start()

        # 启动引擎
        self.engine = SyncEngine(self.q_video, self.q_thermal)
        self.engine.log_signal.connect(self.log)
        self.engine.update_signal.connect(self.update_screens)
        self.engine.start()

    def toggle_fusion_mode(self):
        # 切换 EDGE <-> CHECKER
        if self.current_fusion == "EDGE":
            self.current_fusion = "CHECKER"
        else:
            self.current_fusion = "EDGE"

        # 通知后端
        if hasattr(self, 'engine'):
            self.engine.set_fusion_style(self.current_fusion)

        # 更新按钮文字
        t = TRANSLATIONS[self.lang]
        self.btn_mode.setText(t["btn_mode"] + self.current_fusion)

    def toggle_lang(self):
        self.lang = "EN" if self.lang == "CN" else "CN"
        t = TRANSLATIONS[self.lang]

        # 刷新文字
        self.setWindowTitle(t["title"])
        self.btn_link.setText(t["btn_active"] if not self.btn_link.isEnabled() else t["btn_link"])
        self.btn_mode.setText(t["btn_mode"] + self.current_fusion)
        self.hud_main.lang = self.lang
        self.hud_sub.lang = self.lang
        self.hud_main.update()  # 触发重绘
        self.hud_sub.update()

    @pyqtSlot(np.ndarray, np.ndarray, dict)
    def update_screens(self, main_img, sub_img, info):
        self.hud_main.update_frame(main_img, info)
        self.hud_sub.update_frame(sub_img, info)

        diff = info.get("sync_diff", 0)
        self.lbl_sync.setText(f"{diff * 1000:.1f} ms")

    def closeEvent(self, event):
        if hasattr(self, 'th_v'): self.th_v.stop()
        if hasattr(self, 'th_t'): self.th_t.stop()
        if hasattr(self, 'engine'): self.engine.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = TacticalTerminal()
    win.show()
    sys.exit(app.exec())