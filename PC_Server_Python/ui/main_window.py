import numpy as np
import cv2
import os
import json
import re
import datetime
import subprocess
import sys
import glob
from PyQt6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QPushButton,
                             QGridLayout, QLabel, QTextEdit, QSizePolicy, QSlider, QMessageBox,
                             QFileDialog, QGroupBox, QSpinBox, QComboBox, QProgressBar)
from PyQt6.QtCore import Qt, pyqtSlot, QRect, pyqtSignal, QProcess, QTimer
from PyQt6.QtGui import QPainter, QColor, QPen, QImage, QPixmap, QFont, QIcon
from core.data_link import DataReceiver
from core.sync_engine import SyncEngine

try:
    from config import PORT_VIDEO, PORT_THERMAL
except ImportError:
    PORT_VIDEO, PORT_THERMAL = 5000, 5001

TRANS = {
    "EN": {
        "title": "ELL-4D Reconstruction Terminal",
        "btn_start": "SYSTEM START", "btn_stop": "SYSTEM HALT",
        "mode_locked": "MODE: LOCKED", "mode_adjust": "MODE: ADJUST",
        "check": "CHECKER PATTERN", "rot": "ROTATION", "scale": "SCALE",
        "x_crs": "X-CRS", "x_fin": "X-FIN", "y_crs": "Y-CRS", "y_fin": "Y-FIN",
        "lang": "LANG: EN", "gen_4d": "RUN HEXPLANE BRIDGE",
        "load_4d": "LOAD 4D MATRIX",
        "exit_4d": "EXIT 4D PREVIEW",
        "time_lbl": "TIME:",
        "hud_main": "FUSION OPTIC", "hud_sub1": "THERMAL SENSOR", "hud_sub2": "EVENT TRACKER",
        "hud_roi": "TARGET ROI", "hud_depth": "ROUGH 4D DEPTH",
        "grp_net": "NETWORK CONFIG", "grp_align": "TACTICAL ALIGNMENT", "grp_sys": "SYSTEM CORE",
        "grp_det": "SENSITIVITY & TRIGGER",
        "t_sens": "T-SENS:", "e_sens": "E-SENS:", "cool": "COOLDOWN:", "trg": "REC TRG:"
    },
    "CN": {
        "title": "极弱光4D重建终端",
        "btn_start": "系统启动", "btn_stop": "系统终止",
        "mode_locked": "模式: 锁定", "mode_adjust": "模式: 校准",
        "check": "棋盘对比", "rot": "旋转修正", "scale": "缩放调整",
        "x_crs": "水平粗调", "x_fin": "水平精调", "y_crs": "垂直粗调", "y_fin": "垂直精调",
        "lang": "语言: 中文", "gen_4d": "一键重建并烘焙 4D 场",
        "load_4d": "加载时空光场矩阵",
        "exit_4d": "退出 4D 预览",
        "time_lbl": "时间轴:",
        "hud_main": "融合主视野", "hud_sub1": "热成像传感器", "hud_sub2": "事件流传感器",
        "hud_roi": "目标特写", "hud_depth": "实时4D预览",
        "grp_net": "多路网口配置", "grp_align": "光轴校准面板", "grp_sys": "系统核心控制",
        "grp_det": "目标检测与触发",
        "t_sens": "热阈值:", "e_sens": "事件域:", "cool": "冷却(帧):", "trg": "触发逻辑:"
    }
}

UI_STYLESHEET = """
QMainWindow { background: #050505; }
QLabel { color: #00ff00; font-family: Consolas; font-size: 11px; }
QGroupBox { border: 1px solid #004400; border-radius: 3px; margin-top: 2ex; font-family: Consolas; color: #00aa00; font-weight: bold; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; left: 10px; }
QSpinBox, QComboBox { background: #001100; color: #00ff00; border: 1px solid #005500; font-family: Consolas; padding: 2px; }
QTextEdit { background: #000; border: 1px solid #003300; color: #00ff00; font-family: Consolas; }
QSlider::groove:horizontal { border: 1px solid #003300; height: 4px; background: #001100; margin: 0px 0; }
QSlider::handle:horizontal { background: #00ff00; border: 1px solid #00aa00; width: 12px; height: 16px; margin: -6px 0; border-radius: 2px; }
QPushButton { font-family: Consolas; font-weight: bold; border-radius: 2px; }
QProgressBar { border: 1px solid #005500; background: #001100; text-align: center; color: #0f0; font-family: Consolas; font-weight: bold;}
QProgressBar::chunk { background-color: #00aa00; width: 10px; margin: 0.5px; }
"""


class HUDDisplay(QLabel):
    clicked = pyqtSignal(str)
    dragged = pyqtSignal(int, int)

    def __init__(self, name_key, color_hex, is_main=False, offline_text="NO SIGNAL"):
        super().__init__()
        self.name_key = name_key
        self.display_name = name_key
        self.color = QColor(color_hex)
        self.is_main = is_main
        self.offline_text = offline_text
        self.setMinimumSize(100, 80)
        self.setStyleSheet("background-color: #020202; border: 1px solid #111;")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.info = {}
        self.drag_start_pos = None
        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def set_display_name(self, name):
        self.display_name = name
        self.update()

    def update_frame(self, cv_img, content_type, info=None):
        try:
            if info is not None: self.info = info
            if cv_img is None or cv_img.shape[0] < 10: return
            if not cv_img.flags['C_CONTIGUOUS']: cv_img = np.ascontiguousarray(cv_img)
            h, w = cv_img.shape[:2]
            fmt = QImage.Format.Format_RGB888 if len(cv_img.shape) == 3 else QImage.Format.Format_Grayscale8
            if len(cv_img.shape) == 3: cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            q_img = QImage(cv_img.data, w, h, cv_img.strides[0], fmt)
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
        if e.buttons() & Qt.MouseButton.LeftButton and self.drag_start_pos is not None:
            if self.is_main:
                self.dragged.emit(e.pos().x() - self.drag_start_pos.x(), e.pos().y() - self.drag_start_pos.y())
            self.drag_start_pos = e.pos()
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(2, 2, 2))
        if self.pixmap() and not self.pixmap().isNull():
            pix = self.pixmap()
            x, y = (self.width() - pix.width()) // 2, (self.height() - pix.height()) // 2
            p.drawPixmap(x, y, pix)
            pen = QPen(self.color)
            pen.setWidth(2)
            p.setPen(pen)
            r = QRect(x, y, pix.width(), pix.height()).adjusted(0, 0, -1, -1)
            l = 15
            p.drawLine(r.left(), r.top(), r.left() + l, r.top())
            p.drawLine(r.left(), r.top(), r.left(), r.top() + l)
            p.drawLine(r.right(), r.bottom(), r.right() - l, r.bottom())
            p.drawLine(r.right(), r.bottom(), r.right(), r.bottom() - l)
            p.setPen(QColor(255, 255, 255))
            p.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
            p.drawText(r.left() + 5, r.top() + 15, self.display_name)
        else:
            p.setPen(QColor(60, 60, 60))
            p.setFont(QFont("Consolas", 10))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.offline_text)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cur_lang = "EN"
        self._last_vals = {"x_crs": 0, "x_fin": 0, "y_crs": 0, "y_fin": 0}
        self.bridge_phase = 0

        self.matrix_frames = []
        self.is_matrix_mode = False
        self.curr_t = 0
        self.curr_v = 0
        self.drag_acc_x = 0
        self.drag_acc_y = 0

        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self.play_next_frame)

        if os.path.exists("images/logo.ico"): self.setWindowIcon(QIcon("images/logo.ico"))
        self.setWindowTitle(TRANS[self.cur_lang]["title"])
        self.resize(1600, 900)
        self.setStyleSheet(UI_STYLESHEET)
        self.win_state = {"hud_main": "FUSION", "hud_sub1": "THERMAL", "hud_sub2": "EVENT", "hud_roi": "ROI",
                          "hud_depth": "DEPTH"}
        self.init_ui()
        self.update_ui_text()

    def init_ui(self):
        base = QWidget();
        self.setCentralWidget(base)
        main_layout = QHBoxLayout(base);
        main_layout.setContentsMargins(10, 10, 10, 10);
        main_layout.setSpacing(10)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(5)

        self.hud_main = HUDDisplay("hud_main", "#00ff00", True, "SYSTEM OFFLINE")
        self.hud_main.clicked.connect(self.handle_swap);
        self.hud_main.dragged.connect(self.handle_drag)
        left_layout.addWidget(self.hud_main, 1)

        self.player_ctrl_widget = QWidget()
        ctrl_layout = QHBoxLayout(self.player_ctrl_widget)
        ctrl_layout.setContentsMargins(5, 5, 5, 5)

        self.lbl_time = QLabel("TIME:")
        self.lbl_time.setStyleSheet("color: #00ffcc; font-weight: bold;")
        self.sld_time = QSlider(Qt.Orientation.Horizontal)
        self.sld_time.setRange(0, 29)
        self.sld_time.valueChanged.connect(self.on_time_slider_changed)

        self.btn_exit_preview = QPushButton("EXIT 4D PREVIEW")
        self.btn_exit_preview.setStyleSheet(
            "background:#440000; color:#ff3333; border:1px solid #ff3333; padding: 6px 12px;")
        self.btn_exit_preview.clicked.connect(self.exit_4d_preview)

        ctrl_layout.addWidget(self.lbl_time)
        ctrl_layout.addWidget(self.sld_time, 1)
        ctrl_layout.addWidget(self.btn_exit_preview)

        self.player_ctrl_widget.setVisible(False)
        left_layout.addWidget(self.player_ctrl_widget, 0)

        main_layout.addWidget(left_panel, 65)

        right_panel = QWidget();
        r_layout = QVBoxLayout(right_panel);
        r_layout.setContentsMargins(0, 0, 0, 0);
        r_layout.setSpacing(6)

        hud_grid = QGridLayout();
        hud_grid.setContentsMargins(0, 0, 0, 0);
        hud_grid.setSpacing(6)
        self.hud_sub1 = HUDDisplay("hud_sub1", "#ff5500", False);
        self.hud_sub1.clicked.connect(self.handle_swap)
        self.hud_sub2 = HUDDisplay("hud_sub2", "#00ffff", False);
        self.hud_sub2.clicked.connect(self.handle_swap)
        self.hud_roi = HUDDisplay("hud_roi", "#ff00ff", False);
        self.hud_roi.clicked.connect(self.handle_swap)
        self.hud_depth = HUDDisplay("hud_depth", "#ffff00", False);
        self.hud_depth.clicked.connect(self.handle_swap)
        hud_grid.addWidget(self.hud_sub1, 0, 0);
        hud_grid.addWidget(self.hud_sub2, 0, 1)
        hud_grid.addWidget(self.hud_roi, 1, 0);
        hud_grid.addWidget(self.hud_depth, 1, 1)
        r_layout.addLayout(hud_grid, 4)

        self.grp_net = QGroupBox("NETWORK CONFIG");
        net_lay = QHBoxLayout(self.grp_net);
        net_lay.setContentsMargins(10, 15, 10, 10)
        self.lbl_port_v = QLabel("VIS:");
        self.spin_v = QSpinBox();
        self.spin_v.setRange(1000, 65535);
        self.spin_v.setValue(PORT_VIDEO)
        self.lbl_port_t = QLabel("THR:");
        self.spin_t = QSpinBox();
        self.spin_t.setRange(1000, 65535);
        self.spin_t.setValue(PORT_THERMAL)
        net_lay.addWidget(self.lbl_port_v);
        net_lay.addWidget(self.spin_v);
        net_lay.addWidget(self.lbl_port_t);
        net_lay.addWidget(self.spin_t)
        r_layout.addWidget(self.grp_net, 0)

        self.grp_det = QGroupBox("SENSITIVITY & TRIGGER");
        det_lay = QGridLayout(self.grp_det);
        det_lay.setContentsMargins(10, 15, 10, 10)
        self.lbl_t_sens = QLabel("T-SENS:");
        self.sld_t_sens = QSlider(Qt.Orientation.Horizontal);
        self.sld_t_sens.setRange(20, 35);
        self.sld_t_sens.setValue(25)
        self.lbl_e_sens = QLabel("E-SENS:");
        self.sld_e_sens = QSlider(Qt.Orientation.Horizontal);
        self.sld_e_sens.setRange(10, 60);
        self.sld_e_sens.setValue(30)
        self.lbl_cool = QLabel("COOLDOWN:");
        self.sld_cool = QSlider(Qt.Orientation.Horizontal);
        self.sld_cool.setRange(1, 60);
        self.sld_cool.setValue(5)
        self.lbl_trg = QLabel("REC TRG:");
        self.cmb_trg = QComboBox();
        self.cmb_trg.addItems(["BOTH", "THERMAL", "EVENT", "ANY"])
        for w in [self.sld_t_sens, self.sld_e_sens, self.sld_cool]: w.valueChanged.connect(self.update_det_params)
        self.cmb_trg.currentTextChanged.connect(self.update_det_params)
        det_lay.addWidget(self.lbl_t_sens, 0, 0);
        det_lay.addWidget(self.sld_t_sens, 0, 1)
        det_lay.addWidget(self.lbl_e_sens, 1, 0);
        det_lay.addWidget(self.sld_e_sens, 1, 1)
        det_lay.addWidget(self.lbl_cool, 2, 0);
        det_lay.addWidget(self.sld_cool, 2, 1)
        det_lay.addWidget(self.lbl_trg, 3, 0);
        det_lay.addWidget(self.cmb_trg, 3, 1)
        r_layout.addWidget(self.grp_det, 0)

        self.grp_align = QGroupBox("TACTICAL ALIGNMENT");
        ag = QGridLayout(self.grp_align);
        ag.setContentsMargins(10, 15, 10, 10);
        ag.setVerticalSpacing(4)
        self.btn_check = QPushButton("CHECKER PATTERN");
        self.btn_check.setStyleSheet("background:#111; color:#0f0; padding:4px; border:1px solid #050;");
        self.btn_check.clicked.connect(self.toggle_checker);
        self.btn_check.setEnabled(False)
        ag.addWidget(self.btn_check, 0, 0, 1, 2)
        self.lbl_x_crs = QLabel("X-CRS");
        self.sld_x_crs = QSlider(Qt.Orientation.Horizontal)
        self.lbl_x_fin = QLabel("X-FIN");
        self.sld_x_fin = QSlider(Qt.Orientation.Horizontal)
        self.lbl_y_crs = QLabel("Y-CRS");
        self.sld_y_crs = QSlider(Qt.Orientation.Horizontal)
        self.lbl_y_fin = QLabel("Y-FIN");
        self.sld_y_fin = QSlider(Qt.Orientation.Horizontal)
        for i, (lbl, sld, mi, ma, key) in enumerate([(self.lbl_x_crs, self.sld_x_crs, -200, 200, "x_crs"),
                                                     (self.lbl_x_fin, self.sld_x_fin, -20, 20, "x_fin"),
                                                     (self.lbl_y_crs, self.sld_y_crs, -200, 200, "y_crs"),
                                                     (self.lbl_y_fin, self.sld_y_fin, -20, 20, "y_fin")]):
            lbl.setFixedWidth(40);
            sld.setRange(mi, ma);
            sld.setValue(0);
            sld.valueChanged.connect(lambda v, k=key: self.on_align_slider_change(v, k));
            sld.sliderReleased.connect(self.reset_slider_origins)
            ag.addWidget(lbl, i + 1, 0);
            ag.addWidget(sld, i + 1, 1)
        self.lbl_rot = QLabel("ROT");
        self.sld_rot = QSlider(Qt.Orientation.Horizontal);
        self.sld_rot.setRange(-20, 20);
        self.sld_rot.valueChanged.connect(lambda v: self.eng.update_align_params(set_angle=v))
        self.lbl_sc = QLabel("SCALE");
        self.sld_sc = QSlider(Qt.Orientation.Horizontal);
        self.sld_sc.setRange(5, 50);
        self.sld_sc.setValue(25);
        self.sld_sc.valueChanged.connect(lambda v: self.eng.update_align_params(set_scale=v / 10.0))
        ag.addWidget(self.lbl_rot, 5, 0);
        ag.addWidget(self.sld_rot, 5, 1);
        ag.addWidget(self.lbl_sc, 6, 0);
        ag.addWidget(self.sld_sc, 6, 1)
        r_layout.addWidget(self.grp_align, 0);
        self.grp_align.setVisible(False)

        self.log_v = QTextEdit();
        self.log_v.setReadOnly(True);
        self.log_v.setFixedHeight(50)
        r_layout.addWidget(self.log_v, 0)

        self.grp_sys = QGroupBox("SYSTEM CORE");
        sys_lay = QVBoxLayout(self.grp_sys);
        sys_lay.setContentsMargins(10, 10, 10, 10);
        sys_lay.setSpacing(6)
        self.prog_gen = QProgressBar();
        self.prog_gen.setRange(0, 100);
        self.prog_gen.setValue(0);
        self.prog_gen.hide()
        sys_lay.addWidget(self.prog_gen)

        btn_row = QHBoxLayout()
        self.btn_gen = QPushButton("RUN HEXPLANE BRIDGE");
        self.btn_gen.setStyleSheet("background:#442200; color:#fa0; padding:8px; border:1px solid #fa0;")
        self.btn_gen.clicked.connect(self.generate_4d)

        self.btn_load_4d = QPushButton("LOAD 4D MATRIX");
        self.btn_load_4d.setStyleSheet("background:#003366; color:#0af; padding:8px; border:1px solid #0af;")
        self.btn_load_4d.clicked.connect(self.load_local_4d)

        btn_row.addWidget(self.btn_gen)
        btn_row.addWidget(self.btn_load_4d)
        sys_lay.addLayout(btn_row)

        bot_row = QHBoxLayout();
        bot_row.setSpacing(4)
        self.btn_mode = QPushButton("MODE: LOCKED");
        self.btn_mode.setStyleSheet("background:#001122; color:#0af; padding:6px; border:1px solid #005588;");
        self.btn_mode.clicked.connect(self.toggle_mode);
        self.btn_mode.setEnabled(False)
        self.btn_lang = QPushButton("LANG: EN");
        self.btn_lang.setStyleSheet("background:#222; color:#aaa; padding:6px; border:1px solid #555;");
        self.btn_lang.clicked.connect(self.toggle_lang)
        self.btn_start = QPushButton("SYSTEM START");
        self.btn_start.setStyleSheet("background:#004400; color:#0f0; padding:6px; border:1px solid #0f0;");
        self.btn_start.clicked.connect(self.start)
        bot_row.addWidget(self.btn_mode);
        bot_row.addWidget(self.btn_lang);
        bot_row.addWidget(self.btn_start)
        sys_lay.addLayout(bot_row);
        r_layout.addWidget(self.grp_sys, 0)
        main_layout.addWidget(right_panel, 35)

    def update_det_params(self):
        if not hasattr(self, 'eng'): return
        self.eng.update_detection_params(self.sld_t_sens.value(), self.sld_e_sens.value() * 100, self.sld_cool.value(),
                                         self.cmb_trg.currentText())

    def on_align_slider_change(self, val, key):
        if not hasattr(self, 'eng'): return
        diff = val - self._last_vals[key]
        if "x_" in key:
            self.eng.update_align_params(dx=diff)
        else:
            self.eng.update_align_params(dy=diff)
        self._last_vals[key] = val

    def reset_slider_origins(self):
        for sld in [self.sld_x_crs, self.sld_x_fin, self.sld_y_crs, self.sld_y_fin]:
            sld.blockSignals(True);
            sld.setValue(0);
            sld.blockSignals(False)
        for k in self._last_vals: self._last_vals[k] = 0

    def generate_4d(self):
        d_dir = os.path.join(os.getcwd(), "auto_captures")
        if not os.path.exists(d_dir): os.makedirs(d_dir, exist_ok=True)
        t_dir = QFileDialog.getExistingDirectory(self, "Select Recorded Data Folder", d_dir)
        if t_dir:
            if not os.path.exists("tools/bridge.py"): return QMessageBox.critical(self, "Error",
                                                                                  "Missing tools/bridge.py!")
            self.btn_gen.setDisabled(True);
            self.prog_gen.show();
            self.prog_gen.setValue(0)
            self.bridge_phase = 0;
            self.log_msg(">>> ESTABLISHING HEXPLANE PIPELINE...")
            self.bridge_process = QProcess(self)
            self.bridge_process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
            self.bridge_process.readyReadStandardOutput.connect(self.handle_bridge_output)
            self.bridge_process.finished.connect(self.handle_bridge_finished)
            self.bridge_process.start(sys.executable, ["tools/bridge.py", "--target", t_dir])

    def handle_bridge_output(self):
        raw_bytes = self.bridge_process.readAllStandardOutput().data()
        data_str = raw_bytes.decode('utf-8', errors='replace')
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

        for line in data_str.replace('\r', '\n').split('\n'):
            line = line.strip()
            if not line: continue
            clean_line = ansi_escape.sub('', line)

            if "[RESULT_PATH]:" in clean_line:
                self.rendered_path = clean_line.split("[RESULT_PATH]:")[-1].strip()

            if "[1/3]" in clean_line:
                self.bridge_phase = 1; self.prog_gen.setValue(5)
            elif "[2/3]" in clean_line:
                self.bridge_phase = 2; self.prog_gen.setValue(25)
            elif "核心训练" in clean_line:
                self.bridge_phase = 3; self.prog_gen.setValue(30)
            elif "烘焙" in clean_line:
                self.bridge_phase = 4; self.prog_gen.setValue(85)

            if self.bridge_phase == 3:
                match = re.search(r'(\d+)/3000', clean_line)
                if match:
                    self.prog_gen.setValue(30 + int((int(match.group(1)) / 3000.0) * 55))
                    self.btn_gen.setText(f"TRAINING... {match.group(1)}/3000")
            elif self.bridge_phase == 4:
                # 🔥 实时解析烘焙 1080 帧的进度，不再“假死”
                match = re.search(r'(\d+)/1080', clean_line)
                if match:
                    self.prog_gen.setValue(85 + int((int(match.group(1)) / 1080.0) * 15))
                    self.btn_gen.setText(f"BAKING MATRIX... {match.group(1)}/1080")

            if "it/s" not in clean_line and "s/it" not in clean_line and "%|" not in clean_line:
                self.log_msg(clean_line)

    def handle_bridge_finished(self):
        self.btn_gen.setEnabled(True);
        self.btn_gen.setText(TRANS[self.cur_lang]["gen_4d"])
        self.prog_gen.setValue(100)
        self.log_msg(">>> HEXPLANE PIPELINE COMPLETED.")

        if hasattr(self, 'rendered_path') and os.path.exists(self.rendered_path):
            self.load_rendered_4d(self.rendered_path)

    def load_local_4d(self):
        base_log_dir = r"D:\CPP\HexPlane\log"
        if not os.path.exists(base_log_dir): base_log_dir = os.getcwd()
        folder = QFileDialog.getExistingDirectory(self, "Select 4D Result Folder", base_log_dir)
        if not folder: return

        target_sub = None
        for sub in ["imgs_test_all", "imgs_path_all", "imgs_train_all"]:
            if os.path.exists(os.path.join(folder, sub)):
                if glob.glob(os.path.join(folder, sub, "*.png")):
                    target_sub = os.path.join(folder, sub)
                    break

        if not target_sub:
            QMessageBox.warning(self, "Error", "No valid rendered images found in this folder!")
            return

        self.load_rendered_4d(target_sub)

    def load_rendered_4d(self, path):
        self.matrix_frames = []
        files = [f for f in glob.glob(os.path.join(path, "test*.png")) if "_gt" not in f]

        # 🔥 致命排序修复：提取数字进行比较，绝对防止 test1000 排在 test101 前面！
        def get_idx(f):
            m = re.search(r'test(\d+)\.png', os.path.basename(f))
            return int(m.group(1)) if m else 0

        files.sort(key=get_idx)

        if len(files) == 30 * 36:
            self.is_matrix_mode = True
            for t in range(30):
                row = []
                for v in range(36):
                    idx = t * 36 + v
                    img = cv2.imread(files[idx])
                    row.append(img)
                self.matrix_frames.append(row)

            self.curr_t, self.curr_v = 0, 0
            self.log_msg(f"✅ 成功解包时空光场矩阵！({len(files)}帧无损导入)")
            self.win_state["hud_main"] = "4D_PLAYER"
            self.hud_main.set_display_name("4D LIGHT FIELD [X-DRAG: 360° | Y-DRAG: TIME]")
            self.hud_main.color = QColor("#00ffcc")

            self.player_ctrl_widget.setVisible(True)
            self.sld_time.blockSignals(True)
            self.sld_time.setValue(0)
            self.sld_time.blockSignals(False)

            self.play_timer.start(33)
        else:
            self.log_msg(f"⚠️ 矩阵数据不完整 (仅找到 {len(files)} 帧)。请点击 RUN 重新训练与烘焙！")

    def on_time_slider_changed(self, val):
        if not self.is_matrix_mode: return
        self.play_timer.stop()
        self.curr_t = val
        self.hud_main.update_frame(self.matrix_frames[self.curr_t][self.curr_v], "4D_PLAYER", {"rec": ""})

    def exit_4d_preview(self):
        self.play_timer.stop()
        self.is_matrix_mode = False
        self.matrix_frames = []
        self.player_ctrl_widget.setVisible(False)

        self.win_state["hud_main"] = "FUSION"
        self.hud_main.color = QColor("#00ff00")
        self.hud_main.setPixmap(QPixmap())
        self.update_ui_text()
        self.log_msg("⏹️ 已退出 4D 预览，返回实时监控模式。")

    @pyqtSlot()
    def play_next_frame(self):
        if not self.matrix_frames: return
        self.curr_t = (self.curr_t + 1) % 30

        self.sld_time.blockSignals(True)
        self.sld_time.setValue(self.curr_t)
        self.sld_time.blockSignals(False)

        self.hud_main.update_frame(self.matrix_frames[self.curr_t][self.curr_v], "4D_PLAYER", {"rec": ""})

    def handle_drag(self, dx, dy):
        c_type = self.win_state["hud_main"]
        if c_type == "DEPTH":
            self.eng.update_depth_rotation(dx, dy)
        elif c_type == "FUSION" and self.grp_align.isVisible():
            self.eng.update_align_params(dx=dx, dy=dy)
        elif c_type == "4D_PLAYER":
            self.play_timer.stop()
            if self.is_matrix_mode:
                self.drag_acc_x += dx
                self.drag_acc_y += dy

                if abs(self.drag_acc_x) > 10:
                    step = 1 if self.drag_acc_x > 0 else -1
                    self.curr_v = (self.curr_v + step) % 36
                    self.drag_acc_x = 0

                if abs(self.drag_acc_y) > 10:
                    step = 1 if self.drag_acc_y > 0 else -1
                    self.curr_t = (self.curr_t + step) % 30
                    self.drag_acc_y = 0

                    self.sld_time.blockSignals(True)
                    self.sld_time.setValue(self.curr_t)
                    self.sld_time.blockSignals(False)

                self.hud_main.update_frame(self.matrix_frames[self.curr_t][self.curr_v], "4D_PLAYER", {"rec": ""})

    def handle_swap(self, clicked_key):
        if clicked_key == "hud_main":
            if self.win_state.get("hud_main") == "4D_PLAYER":
                if self.play_timer.isActive():
                    self.play_timer.stop()
                    self.log_msg("⏸️ 4D 空间已冻结 (鼠标X轴旋转，Y轴回溯时间)")
                else:
                    self.play_timer.start(33)
                    self.log_msg("▶️ 4D 空间恢复演化")
            return

        self.win_state["hud_main"], self.win_state[clicked_key] = self.win_state.get(clicked_key), self.win_state[
            "hud_main"]
        self.update_ui_text()

    def toggle_lang(self):
        self.cur_lang = "CN" if self.cur_lang == "EN" else "EN";
        self.update_ui_text()

    def update_ui_text(self):
        t = TRANS[self.cur_lang]
        self.setWindowTitle(t["title"]);
        self.grp_net.setTitle(t["grp_net"]);
        self.grp_align.setTitle(t["grp_align"]);
        self.grp_sys.setTitle(t["grp_sys"]);
        self.grp_det.setTitle(t["grp_det"])
        self.lbl_t_sens.setText(t["t_sens"]);
        self.lbl_e_sens.setText(t["e_sens"]);
        self.lbl_cool.setText(t["cool"]);
        self.lbl_trg.setText(t["trg"])
        self.btn_start.setText(t["btn_stop"] if hasattr(self, 'eng') and self.eng.isRunning() else t["btn_start"])
        self.btn_mode.setText(
            t["mode_adjust"] if self.btn_mode and "ADJUST" in self.btn_mode.text() else t["mode_locked"])
        self.btn_check.setText(t["check"]);
        self.btn_lang.setText(t["lang"]);
        self.btn_gen.setText(t["gen_4d"]);
        self.btn_load_4d.setText(t["load_4d"])

        self.lbl_time.setText(t["time_lbl"])
        self.btn_exit_preview.setText(t["exit_4d"])

        self.lbl_rot.setText(t["rot"].split()[0]);
        self.lbl_sc.setText(t["scale"].split()[0]);
        self.lbl_x_crs.setText(t["x_crs"].split()[0]);
        self.lbl_x_fin.setText(t["x_fin"].split()[0]);
        self.lbl_y_crs.setText(t["y_crs"].split()[0]);
        self.lbl_y_fin.setText(t["y_fin"].split()[0])
        for hud in [self.hud_main, self.hud_sub1, self.hud_sub2, self.hud_roi, self.hud_depth]:
            if hud.name_key == "hud_main" and self.win_state.get("hud_main") == "4D_PLAYER": continue
            hud.set_display_name(t.get(hud.name_key, hud.name_key))

    def start(self):
        if hasattr(self, 'eng') and self.eng.isRunning():
            self.eng.stop();
            self.th_v.stop();
            self.th_t.stop()
            self.spin_v.setEnabled(True);
            self.spin_t.setEnabled(True)
            self.btn_start.setText(TRANS[self.cur_lang]["btn_start"]);
            self.btn_start.setStyleSheet("background:#004400; color:#0f0;")
            return
        try:
            self.spin_v.setEnabled(False);
            self.spin_t.setEnabled(False)
            from collections import deque
            self.qv, self.qt = deque(maxlen=4), deque(maxlen=4)
            self.th_v = DataReceiver(self.spin_v.value(), self.qv, "video")
            self.th_t = DataReceiver(self.spin_t.value(), self.qt, "thermal")
            self.th_v.log_signal.connect(self.log_msg);
            self.th_t.log_signal.connect(self.log_msg)
            self.th_v.start();
            self.th_t.start()
            self.eng = SyncEngine(self.qv, self.qt)
            self.eng.update_signal.connect(self.update_displays);
            self.eng.log_signal.connect(self.log_msg)
            self.update_det_params();
            self.eng.start()
            self.btn_start.setText(TRANS[self.cur_lang]["btn_stop"]);
            self.btn_start.setStyleSheet("background:#660000; color:#f00;")
            self.btn_mode.setEnabled(True);
            self.btn_check.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "System Error", f"Startup Failed:\n{e}");
            self.spin_v.setEnabled(True);
            self.spin_t.setEnabled(True)

    def log_msg(self, msg):
        self.log_v.append(f"> {msg}");
        self.log_v.verticalScrollBar().setValue(self.log_v.verticalScrollBar().maximum())

    def toggle_mode(self):
        t = TRANS[self.cur_lang]
        is_adj = "ADJUST" in self.btn_mode.text() or "校准" in self.btn_mode.text()
        if is_adj:
            nm, txt = "LOCKED", t["mode_locked"]; self.grp_align.setVisible(False); self.btn_mode.setStyleSheet(
                "background:#001122; color:#0af; padding:6px; border:1px solid #005588;")
        else:
            nm, txt = "ADJUST", t["mode_adjust"]; self.grp_align.setVisible(True); self.btn_mode.setStyleSheet(
                "background:#332200; color:#fd0; padding:6px; border:1px solid #aa6600;")
        self.btn_mode.setText(txt);
        self.eng.set_mode(nm)

    def toggle_checker(self):
        self.eng.update_align_params(toggle_checker=True)

    @pyqtSlot(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict)
    def update_displays(self, fus, raw_therm, raw_evt, roi, depth, info):
        try:
            cmap = {"FUSION": fus, "THERMAL": raw_therm, "EVENT": raw_evt, "ROI": roi, "DEPTH": depth}
            for hud_key in ["hud_main", "hud_sub1", "hud_sub2", "hud_roi", "hud_depth"]:
                if hud_key == "hud_main" and self.win_state.get("hud_main") == "4D_PLAYER": continue
                getattr(self, hud_key).update_frame(cmap.get(self.win_state.get(hud_key)), self.win_state.get(hud_key),
                                                    info)
        except:
            pass

    def closeEvent(self, e):
        try:
            self.eng.stop(); self.th_v.stop(); self.th_t.stop()
        except:
            pass


if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv);
    window = MainWindow();
    window.show();
    sys.exit(app.exec())