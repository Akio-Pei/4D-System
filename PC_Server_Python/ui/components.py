from PyQt6.QtWidgets import QLabel
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QFont
from PyQt6.QtCore import Qt, QRect
import numpy as np
import cv2


class HUDDisplay(QLabel):
    # 兼容两种初始化方式：(name, color, is_main) 和 (title_key, color, lang...)
    def __init__(self, *args, **kwargs):
        super().__init__()
        # 简单参数解析
        self.name = args[0] if len(args) > 0 else "HUD"
        self.color = QColor(args[1] if len(args) > 1 else "#00ff00")

        self.setMinimumSize(320, 240)
        self.setStyleSheet("background-color: #050505;")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)

        self.raw_thermal = None
        self.hover_temp = None
        self.content_type = "NONE"

    def update_frame(self, cv_img, content_type="NONE", info=None, raw_t=None):
        self.content_type = content_type
        if raw_t is not None: self.raw_thermal = raw_t

        if cv_img is None: return

        if not cv_img.flags['C_CONTIGUOUS']: cv_img = np.ascontiguousarray(cv_img)
        h, w = cv_img.shape[:2]
        if len(cv_img.shape) == 2:
            fmt = QImage.Format.Format_Grayscale8; bpl = w
        else:
            fmt = QImage.Format.Format_RGB888
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            if not cv_img.flags['C_CONTIGUOUS']: cv_img = np.ascontiguousarray(cv_img)
            bpl = cv_img.strides[0]

        q_img = QImage(cv_img.data, w, h, bpl, fmt)
        self.setPixmap(QPixmap.fromImage(q_img).scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                       Qt.TransformationMode.FastTransformation))
        self.update()

    def paintEvent(self, event):
        p = QPainter(self);
        p.fillRect(self.rect(), QColor(0, 0, 0))

        if self.pixmap():
            pix = self.pixmap()
            x = (self.width() - pix.width()) // 2
            y = (self.height() - pix.height()) // 2
            p.drawPixmap(x, y, pix)

            # === 只画四个角 (Corner Brackets) ===
            pen = QPen(self.color);
            pen.setWidth(2);
            p.setPen(pen)
            rect = QRect(x, y, pix.width(), pix.height())
            r = rect.adjusted(0, 0, -1, -1);
            l = 15

            # 左上
            p.drawLine(r.left(), r.top(), r.left() + l, r.top())
            p.drawLine(r.left(), r.top(), r.left(), r.top() + l)
            # 右上
            p.drawLine(r.right(), r.top(), r.right() - l, r.top())
            p.drawLine(r.right(), r.top(), r.right(), r.top() + l)
            # 右下
            p.drawLine(r.right(), r.bottom(), r.right() - l, r.bottom())
            p.drawLine(r.right(), r.bottom(), r.right(), r.bottom() - l)
            # 左下
            p.drawLine(r.left(), r.bottom(), r.left() + l, r.bottom())
            p.drawLine(r.left(), r.bottom(), r.left(), r.bottom() - l)

            # 标题 (左上角)
            p.setPen(QColor(255, 255, 255));
            p.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
            p.drawText(r.left() + 5, r.top() + 18, str(self.content_type))

        else:
            # === 无信号时：只显示小字 ===
            p.setPen(QColor(80, 80, 80))
            p.setFont(QFont("Consolas", 10))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "NO SIGNAL")