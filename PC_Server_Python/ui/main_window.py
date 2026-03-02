import numpy as np
import cv2
import os
import json
import datetime
import subprocess  # <--- [新增] 用于启动脚本
import sys  # <--- [新增] 用于获取 python 环境
from PyQt6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QPushButton,
                             QFrame, QGridLayout, QLabel, QTextEdit, QSizePolicy, QSlider, QMessageBox,
                             QFileDialog)  # <--- [新增] QFileDialog
from PyQt6.QtCore import Qt, pyqtSlot, QRect, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QImage, QPixmap, QFont, QIcon
from core.data_link import DataReceiver
from core.sync_engine import SyncEngine
from config import PORT_VIDEO, PORT_THERMAL

TRANS = {
    "EN": {
        "title": "ELL-4D Reconstruction Terminal",
        "btn_start": "SYSTEM START", "btn_stop": "SYSTEM HALT",
        "mode_locked": "MODE: LOCKED", "mode_adjust": "MODE: ADJUST",
        "check": "CHECKER PATTERN", "rot": "ROTATION", "scale": "SCALE", "fine": "FINE",
        "lang": "LANG: EN", "gen_4d": "RUN HEXPLANE BRIDGE",  # 修改了文案
        "hud_main": "FUSION OPTIC", "hud_sub1": "THERMAL SENSOR", "hud_sub2": "EVENT TRACKER",
        "hud_roi": "TARGET ROI", "hud_depth": "ROUGH 4D DEPTH"
    },
    "CN": {
        "title": "极弱光4D重建终端",
        "btn_start": "系统启动", "btn_stop": "系统终止",
        "mode_locked": "模式: 锁定", "mode_adjust": "模式: 校准",
        "check": "棋盘对比", "rot": "旋转修正", "scale": "缩放调整", "fine": "精细微调",
        "lang": "语言: 中文", "gen_4d": "一键生成4D模型",  # 修改了文案
        "hud_main": "融合主视野", "hud_sub1": "热成像传感器", "hud_sub2": "事件流传感器",
        "hud_roi": "目标特写", "hud_depth": "实时4D预览"
    }
}

SLIDER_STYLE = """
QSlider::groove:horizontal { border: 1px solid #004400; height: 6px; background: #001100; margin: 0px 0; }
QSlider::handle:horizontal { background: #00ff00; border: 1px solid #00ff00; width: 10px; height: 14px; margin: -5px 0; border-radius: 1px; }
"""


class HUDDisplay(QLabel):
    clicked = pyqtSignal(str)
    dragged = pyqtSignal(int, int)

    def __init__(self, name_key, color_hex, is_main=False, offline_text="NO SIGNAL"):
        super().__init__()
        self.name_key = name_key;
        self.display_name = name_key
        self.color = QColor(color_hex);
        self.is_main = is_main
        self.offline_text = offline_text
        self.setMinimumSize(100, 80)
        self.setStyleSheet("background-color: #050505;")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.info = {};
        self.raw_thermal = None;
        self.hover_temp = None
        self.last_mouse_pos_for_paint = None
        self.drag_start_pos = None
        self.content_type = "NONE"
        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def set_display_name(self, name):
        self.display_name = name;
        self.update()

    def update_frame(self, cv_img, content_type, info=None, raw_t=None):
        try:
            self.content_type = content_type
            if info: self.info = info
            if raw_t is not None: self.raw_thermal = raw_t
            if cv_img is None or cv_img.shape[0] < 10: return
            if not cv_img.flags['C_CONTIGUOUS']: cv_img = np.ascontiguousarray(cv_img)
            h, w = cv_img.shape[:2]
            if len(cv_img.shape) == 2:
                fmt = QImage.Format.Format_Grayscale8;
                bpl = w
            else:
                fmt = QImage.Format.Format_RGB888;
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB);
                bpl = \
                    cv_img.strides[0]
            q_img = QImage(cv_img.data, w, h, bpl, fmt)
            self.setPixmap(QPixmap.fromImage(q_img).scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                           Qt.TransformationMode.FastTransformation))
            self.update()
        except:
            pass

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.name_key)
            self.drag_start_pos = e.pos()

    def mouseMoveEvent(self, e):
        self.last_mouse_pos_for_paint = e.pos()
        if e.buttons() & Qt.MouseButton.LeftButton and self.drag_start_pos is not None:
            dx = e.pos().x() - self.drag_start_pos.x()
            dy = e.pos().y() - self.drag_start_pos.y()
            if self.is_main: self.dragged.emit(dx, dy)
            self.drag_start_pos = e.pos()
        self.update()

    def paintEvent(self, event):
        p = QPainter(self);
        p.fillRect(self.rect(), QColor(0, 0, 0))
        if self.pixmap() and not self.pixmap().isNull() and self.pixmap().width() > 10:
            pix = self.pixmap()
            x = (self.width() - pix.width()) // 2;
            y = (self.height() - pix.height()) // 2
            p.drawPixmap(x, y, pix)

            pen = QPen(self.color);
            pen.setWidth(2);
            p.setPen(pen)
            r = QRect(x, y, pix.width(), pix.height()).adjusted(0, 0, -1, -1);
            l = 15
            p.drawLine(r.left(), r.top(), r.left() + l, r.top());
            p.drawLine(r.left(), r.top(), r.left(), r.top() + l)
            p.drawLine(r.right(), r.top(), r.right() - l, r.top());
            p.drawLine(r.right(), r.top(), r.right(), r.top() + l)
            p.drawLine(r.right(), r.bottom(), r.right() - l, r.bottom());
            p.drawLine(r.right(), r.bottom(), r.right(), r.bottom() - l)
            p.drawLine(r.left(), r.bottom(), r.left() + l, r.bottom());
            p.drawLine(r.left(), r.bottom(), r.left(), r.bottom() - l)

            p.setPen(QColor(255, 255, 255));
            p.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
            p.drawText(r.left() + 5, r.top() + 15, self.display_name)

            if self.is_main and self.info.get("rec") == "REC":
                p.setPen(QColor(255, 0, 0));
                p.drawText(r.right() - 40, r.top() + 15, "● REC")
        else:
            p.setPen(QColor(80, 80, 80));
            p.setFont(QFont("Consolas", 10))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.offline_text)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cur_lang = "EN"
        if os.path.exists("images/logo.ico"): self.setWindowIcon(QIcon("images/logo.ico"))
        self.setWindowTitle(TRANS[self.cur_lang]["title"])
        self.resize(1600, 900)
        self.setStyleSheet(
            "QMainWindow { background: #080808; } QLabel { color: #0f0; font-family: Consolas; } QTextEdit { background: #000; border: 1px solid #333; color: #0f0; }")

        self.win_state = {
            "hud_main": "FUSION",
            "hud_sub1": "THERMAL",
            "hud_sub2": "EVENT",
            "hud_roi": "ROI",
            "hud_depth": "DEPTH"
        }

        self.is_recording = False
        self.init_ui();
        self.update_ui_text()

    def init_ui(self):
        base = QWidget();
        self.setCentralWidget(base)
        layout = QHBoxLayout(base);
        layout.setContentsMargins(10, 10, 10, 10);
        layout.setSpacing(10)

        self.hud_main = HUDDisplay("hud_main", "#00ff00", True, "SYSTEM OFFLINE")
        self.hud_main.clicked.connect(self.handle_swap)
        self.hud_main.dragged.connect(self.handle_drag)

        right_panel = QWidget();
        self.r_lay = QGridLayout(right_panel);
        self.r_lay.setContentsMargins(0, 0, 0, 0)
        self.r_lay.setVerticalSpacing(2);
        self.r_lay.setHorizontalSpacing(8)

        self.hud_sub1 = HUDDisplay("hud_sub1", "#ff5500", False, "THERMAL DISC.")
        self.hud_sub1.clicked.connect(self.handle_swap)
        self.hud_sub2 = HUDDisplay("hud_sub2", "#00ffff", False, "EVENT DISC.")
        self.hud_sub2.clicked.connect(self.handle_swap)
        self.hud_roi = HUDDisplay("hud_roi", "#ff00ff", False, "NO TARGET")
        self.hud_roi.clicked.connect(self.handle_swap)
        self.hud_depth = HUDDisplay("hud_depth", "#ffff00", False, "WAITING...")
        self.hud_depth.clicked.connect(self.handle_swap)

        self.r_lay.addWidget(self.hud_sub1, 0, 0);
        self.r_lay.addWidget(self.hud_sub2, 0, 1)
        self.r_lay.addWidget(self.hud_roi, 1, 0);
        self.r_lay.addWidget(self.hud_depth, 1, 1)

        self.debug_container = QWidget()
        dc_layout = QVBoxLayout(self.debug_container);
        dc_layout.setContentsMargins(0, 0, 0, 0);
        dc_layout.setSpacing(0)

        self.align_frame = QFrame();
        self.align_frame.setStyleSheet("border:1px solid #004400; background:#020202;")
        self.align_frame.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        ag = QGridLayout(self.align_frame);
        ag.setContentsMargins(6, 6, 6, 6);
        ag.setVerticalSpacing(6)

        self.btn_check = QPushButton("CHECK");
        self.btn_check.setStyleSheet("background:#222; color:#aaa; border:none; padding:4px;")
        self.btn_check.clicked.connect(self.toggle_checker);
        self.btn_check.setEnabled(False)
        ag.addWidget(self.btn_check, 0, 0, 1, 2)

        self.lbl_rot = QLabel("ROT");
        self.lbl_rot.setFont(QFont("Consolas", 8))
        self.sld_rot = QSlider(Qt.Orientation.Horizontal);
        self.sld_rot.setStyleSheet(SLIDER_STYLE)
        self.sld_rot.setRange(-20, 20);
        self.sld_rot.valueChanged.connect(lambda v: self.eng.update_align_params(set_angle=v))
        ag.addWidget(self.lbl_rot, 1, 0);
        ag.addWidget(self.sld_rot, 1, 1)

        self.lbl_sc = QLabel("SC");
        self.lbl_sc.setFont(QFont("Consolas", 8))
        self.sld_sc = QSlider(Qt.Orientation.Horizontal);
        self.sld_sc.setStyleSheet(SLIDER_STYLE)
        self.sld_sc.setRange(5, 50);
        self.sld_sc.setValue(25);
        self.sld_sc.valueChanged.connect(lambda v: self.eng.update_align_params(set_scale=v / 10.0))
        ag.addWidget(self.lbl_sc, 2, 0);
        ag.addWidget(self.sld_sc, 2, 1)

        dc_layout.addWidget(self.align_frame)
        self.log_v = QTextEdit();
        self.log_v.setReadOnly(True);
        self.log_v.setFixedHeight(45)
        self.log_v.setStyleSheet(
            "border: 1px solid #333; font-size: 10px; background: #000; color: #0f0; margin: 0px; border-top: 0px;")
        dc_layout.addWidget(self.log_v)
        self.r_lay.addWidget(self.debug_container, 2, 0, 1, 2)

        self.core_container = QWidget()
        cc_layout = QVBoxLayout(self.core_container);
        cc_layout.setContentsMargins(0, 0, 0, 0);
        cc_layout.setSpacing(2)

        self.btn_gen = QPushButton("GENERATE 4D MODEL");
        self.btn_gen.setStyleSheet(
            "background:#442200; color:#fa0; font-weight:bold; padding:6px; border:1px solid #fa0;")
        # 🔥 连接到新的调用函数
        self.btn_gen.clicked.connect(self.generate_4d)
        cc_layout.addWidget(self.btn_gen)

        bot_row = QHBoxLayout();
        bot_row.setSpacing(2)
        self.btn_mode = QPushButton("MODE");
        self.btn_mode.setStyleSheet("background:#112211; color:#0f0; padding:6px; border:1px solid #050;")
        self.btn_mode.clicked.connect(self.toggle_mode);
        self.btn_mode.setEnabled(False)
        self.btn_lang = QPushButton("LANG");
        self.btn_lang.setStyleSheet("background:#001133; color:#0ff; padding:6px; border:1px solid #005577;")
        self.btn_lang.clicked.connect(self.toggle_lang)
        self.btn_start = QPushButton("START");
        self.btn_start.setStyleSheet(
            "background:#003300; color:#fff; font-weight:bold; padding:6px; border:1px solid #0f0;")
        self.btn_start.clicked.connect(self.start)

        bot_row.addWidget(self.btn_mode);
        bot_row.addWidget(self.btn_lang);
        bot_row.addWidget(self.btn_start)
        cc_layout.addLayout(bot_row)
        self.r_lay.addWidget(self.core_container, 3, 0, 1, 2)

        layout.addWidget(self.hud_main, 65);
        layout.addWidget(right_panel, 35)
        self.r_lay.setRowStretch(0, 3);
        self.r_lay.setRowStretch(1, 3);
        self.r_lay.setRowStretch(2, 0);
        self.r_lay.setRowStretch(3, 0)

    def generate_4d(self):
        """
        弹出文件选择框，选择文件夹后，自动调用 bridge.py
        """
        # 默认打开 auto_captures 方便选择
        default_dir = os.path.join(os.getcwd(), "auto_captures")
        if not os.path.exists(default_dir): os.makedirs(default_dir, exist_ok=True)

        target_dir = QFileDialog.getExistingDirectory(self, "Select Recorded Data Folder", default_dir)

        if target_dir:
            self.log(f">>> Selected for 4D: {os.path.basename(target_dir)}")
            self.log(f">>> Launching Bridge...")

            script_path = os.path.join("tools", "bridge.py")
            if not os.path.exists(script_path):
                QMessageBox.critical(self, "Error", f"Missing {script_path}!\nPlease create it first.")
                return

            try:
                # 启动 bridge.py，传入选中的文件夹路径
                subprocess.Popen([sys.executable, script_path, "--target", target_dir])
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to launch bridge:\n{e}")

    # ... (其他函数保持不变) ...
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_F11:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()

    def toggle_lang(self):
        self.cur_lang = "CN" if self.cur_lang == "EN" else "EN";
        self.update_ui_text()

    def update_ui_text(self):
        t = TRANS[self.cur_lang];
        self.setWindowTitle(t["title"])
        self.btn_start.setText(t["btn_stop"] if hasattr(self, 'eng') and self.eng.isRunning() else t["btn_start"])
        self.btn_mode.setText(t["mode_adjust"] if "ADJUST" in self.btn_mode.text() else t["mode_locked"])
        self.btn_check.setText(t["check"]);
        self.btn_lang.setText(t["lang"]);
        self.btn_gen.setText(t["gen_4d"])  # 更新文案
        self.lbl_rot.setText(t["rot"]);
        self.lbl_sc.setText(t["scale"])
        for hud in [self.hud_main, self.hud_sub1, self.hud_sub2, self.hud_roi, self.hud_depth]:
            hud.set_display_name(t.get(hud.name_key, hud.name_key))

    def log(self, m):
        self.log_v.append(f"> {m}")

    def start(self):
        try:
            from collections import deque
            self.qv = deque(maxlen=4);
            self.qt = deque(maxlen=4)
            self.th_v = DataReceiver(PORT_VIDEO, self.qv, "video");
            self.th_t = DataReceiver(PORT_THERMAL, self.qt, "thermal");
            self.th_v.log_signal.connect(self.log);
            self.th_t.log_signal.connect(self.log)
            self.th_v.start();
            self.th_t.start()
            self.eng = SyncEngine(self.qv, self.qt)
            self.eng.update_signal.connect(self.update_displays);
            self.eng.log_signal.connect(self.log)
            self.eng.start()
            self.btn_start.setDisabled(True);
            self.btn_mode.setEnabled(True);
            self.btn_check.setEnabled(True)
            self.eng.set_mode("ADJUST");
            self.btn_mode.setText(TRANS[self.cur_lang]["mode_adjust"]);
            self.btn_start.setText(TRANS[self.cur_lang]["btn_stop"])
            self.debug_container.setVisible(True)
        except Exception as e:
            QMessageBox.critical(self, "System Error", f"Startup Failed:\n{e}")

    def toggle_mode(self):
        t = TRANS[self.cur_lang]
        is_adj = "ADJUST" in self.btn_mode.text() or "校准" in self.btn_mode.text()
        if is_adj:
            nm, txt = "LOCKED", t["mode_locked"]
            self.debug_container.setVisible(False)
            self.r_lay.setRowStretch(0, 4);
            self.r_lay.setRowStretch(1, 4)
        else:
            nm, txt = "ADJUST", t["mode_adjust"]
            self.debug_container.setVisible(True)
            self.r_lay.setRowStretch(0, 3);
            self.r_lay.setRowStretch(1, 3);
            self.r_lay.setRowStretch(2, 0)
        self.btn_mode.setText(txt);
        self.eng.set_mode(nm)

    def toggle_checker(self):
        self.eng.update_align_params(toggle_checker=True)

    def handle_drag(self, dx, dy):
        c_type = self.win_state["hud_main"]
        if c_type == "DEPTH": self.eng.update_depth_rotation(dx, dy); return
        if c_type == "FUSION" and "ADJUST" in self.btn_mode.text(): self.eng.update_align_params(dx=dx, dy=dy)

    def handle_swap(self, clicked_key):
        if clicked_key == "hud_main": return
        c_clk = self.win_state.get(clicked_key, "NONE");
        c_main = self.win_state["hud_main"]
        if c_clk == "NONE": return
        self.win_state["hud_main"] = c_clk;
        self.win_state[clicked_key] = c_main
        self.log(f"SWAP: {clicked_key} -> MAIN");
        self.update_ui_text()

    @pyqtSlot(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict)
    def update_displays(self, fus, raw_therm, raw_evt, roi, depth, info):
        try:
            content_map = {"FUSION": fus, "THERMAL": raw_therm, "EVENT": raw_evt, "ROI": roi, "DEPTH": depth}
            for hud_key in ["hud_main", "hud_sub1", "hud_sub2", "hud_roi", "hud_depth"]:
                c_type = self.win_state.get(hud_key, "NONE")
                img = content_map.get(c_type, None)
                raw_t_data = self.eng.cache_t_raw if c_type == "THERMAL" else None
                getattr(self, hud_key).update_frame(img, c_type, info, raw_t_data)
        except:
            pass

    def closeEvent(self, e):
        try:
            self.eng.stop(); self.th_v.stop(); self.th_t.stop()
        except:
            pass