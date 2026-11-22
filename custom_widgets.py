import math

import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QRect, QRectF, QPointF
from PyQt6.QtGui import (QPainter, QPen, QColor, QBrush, QFont, QPolygonF, QPixmap, QLinearGradient)

try:                                                   # numpy -> QPolygonF without a Python loop
    from pyqtgraph.functions import create_qpolygonf, ndarray_from_qpolygonf
except Exception:                                      # very old pyqtgraph: fall back to the slow path
    create_qpolygonf = ndarray_from_qpolygonf = None


def make_polygon(px, py):
    n = len(px)
    if create_qpolygonf is not None:
        try:
            poly = create_qpolygonf(n)
            arr = ndarray_from_qpolygonf(poly)
            arr[:, 0] = px
            arr[:, 1] = py
            return poly
        except Exception:
            pass
    return QPolygonF([QPointF(a, b) for a, b in zip(px.tolist(), py.tolist())])


# ==========================================================================
# Tires
# ==========================================================================
class TireTelemetryWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(180)
        self.temps = [0, 0, 0, 0]
        self._label_font = QFont("Consolas", 11, QFont.Weight.Bold)
        self._value_font = QFont("Consolas", 13, QFont.Weight.Bold)

    def update_temps(self, temps):
        temps = list(temps)
        if temps == self.temps:                 # tyre temps change slowly: skip identical frames
            return
        self.temps = temps
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        tire_w = max(45, int(w * 0.15))
        tire_h = max(75, int(h * 0.3))
        gap_x = max(25, int(tire_w * 0.5))
        gap_y = max(15, int(tire_h * 0.2))
        cx, cy = w / 2, h / 2

        rects = [
            QRect(int(cx - tire_w - gap_x), int(cy - tire_h - gap_y), tire_w, tire_h),
            QRect(int(cx + gap_x),          int(cy - tire_h - gap_y), tire_w, tire_h),
            QRect(int(cx - tire_w - gap_x), int(cy + gap_y),          tire_w, tire_h),
            QRect(int(cx + gap_x),          int(cy + gap_y),          tire_w, tire_h),
        ]
        labels = ["FL", "FR", "RL", "RR"]

        for i, rect in enumerate(rects):
            temp = self.temps[i]
            color = QColor(64, 196, 255) if temp < 70 else QColor(255, 61, 0) if temp > 105 else QColor(0, 230, 118)
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor(255, 255, 255, 50), 2))
            painter.drawRoundedRect(rect, 6, 6)

            painter.setPen(QColor(200, 200, 200))
            painter.setFont(self._label_font)
            painter.drawText(QRect(rect.x(), rect.y() - 20, rect.width(), 20), Qt.AlignmentFlag.AlignCenter, labels[i])

            painter.setPen(QColor(255, 255, 255))
            painter.setFont(self._value_font)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, f"{temp}°")
        painter.end()


# ==========================================================================
# RPM bar (plain QPainter - a stylesheet-driven QProgressBar re-parses its
# gradient stylesheet on every repaint, which is slow at 60 Hz)
# ==========================================================================
class RpmBarWidget(QWidget):
    def __init__(self):
        super().__init__()
        self._max = 8500
        self._value = 0
        self.setFixedHeight(24)
        self._pen = QPen(QColor("#333333"), 1)

    def setRange(self, lo, hi):
        if hi != self._max:
            self._max = max(1, hi)
            self.update()

    def setValue(self, v):
        v = max(0, min(int(v), self._max))
        if v != self._value:
            self._value = v
            self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(self._pen)
        p.setBrush(QColor("#0A0A0A"))
        p.drawRoundedRect(r, 4, 4)
        frac = self._value / self._max
        if frac > 0:
            inner = r.adjusted(1.5, 1.5, -1.5, -1.5)
            bar = QRectF(inner.x(), inner.y(), inner.width() * frac, inner.height())
            grad = QLinearGradient(inner.left(), 0, inner.right(), 0)   # gradient spans the full bar, like before
            grad.setColorAt(0.0, QColor("#FFFFFF"))
            grad.setColorAt(0.7, QColor("#D4AF37"))
            grad.setColorAt(1.0, QColor("#FF3D00"))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(grad))
            p.drawRoundedRect(bar, 2, 2)
        p.end()


# ==========================================================================
# Pedal trace (plain QPainter instead of a pyqtgraph PlotWidget: no scene
# graph, no autorange, no per-frame list copies)
# ==========================================================================
class ScrollingTraceWidget(QWidget):
    def __init__(self, length=150):
        super().__init__()
        self.trace_length = length
        self._thr = np.zeros(length, dtype=np.float64)
        self._brk = np.zeros(length, dtype=np.float64)
        self._xs = np.linspace(0.0, 1.0, length)
        self._pen_t = QPen(QColor(0, 230, 118), 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        self._pen_b = QPen(QColor(255, 61, 0), 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        self.setMinimumHeight(60)

    def update_trace(self, throttle, brake):
        self._thr[:-1] = self._thr[1:]
        self._thr[-1] = throttle
        self._brk[:-1] = self._brk[1:]
        self._brk[-1] = brake
        self.update()

    def paintEvent(self, event):
        w, h = self.width(), self.height()
        if w < 10 or h < 10:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pad = 4.0
        px = pad + self._xs * (w - 2 * pad)
        for data, pen in ((self._thr, self._pen_t), (self._brk, self._pen_b)):
            py = (h - pad) - (data / 100.0) * (h - 2 * pad)
            p.setPen(pen)
            p.drawPolyline(make_polygon(px, py))
        p.end()


# ==========================================================================
# 3D track map
# ==========================================================================
def project_3d(x, y, z, cx, cz, scale, angle_yaw, angle_pitch):
    """Scalar projection (single markers)."""
    tx, ty, tz = x - cx, y * 15.0, z - cz
    rx = tx * math.cos(angle_yaw) - tz * math.sin(angle_yaw)
    rz = tx * math.sin(angle_yaw) + tz * math.cos(angle_yaw)
    ry = ty * math.cos(angle_pitch) - rz * math.sin(angle_pitch)
    return rx * scale, -ry * scale


def project_3d_np(x, y, z, cx, cz, scale, yaw, pitch):
    """Same maths as project_3d, for whole arrays at once."""
    tx, ty, tz = x - cx, y * 15.0, z - cz
    cyaw, syaw = math.cos(yaw), math.sin(yaw)
    rx = tx * cyaw - tz * syaw
    rz = tx * syaw + tz * cyaw
    ry = ty * math.cos(pitch) - rz * math.sin(pitch)
    return rx * scale, -ry * scale


class Interactive3DMapWidget(QWidget):
    """
    The track is drawn into an off-screen pixmap that is only re-rendered when the view really changes
    (resize, rotate, bounds, new lap). Every normal frame is: blit the pixmap + draw the car dot.
    In live mode new path segments are appended to the pixmap incrementally, so cost per frame is
    constant no matter how long the lap is (Nordschleife included).
    """
    LIVE_MIN_STEP_M = 2.0          # don't store path points closer than this
    MAX_DRAW_POINTS = 2500         # full redraws are decimated to about this many vertices
    BOUNDS_PAD = 0.06              # live view grows in 6 % steps so it doesn't re-render constantly
    _LIVE_COLORS = (QColor(100, 100, 100), QColor(0, 230, 118), QColor(255, 61, 0))   # coast, throttle, brake

    def __init__(self, is_live=False):
        super().__init__()
        self.is_live = is_live
        self.lap_a = None
        self.lap_b = None
        self.bounds = None                           # [min_x, max_x, min_z, max_z]
        self.yaw, self.pitch = 0.5, 0.6
        self.last_mouse_pos = None
        self.playback_time_ms = 0

        self._pts = np.empty((4096, 4), dtype=np.float64)    # x, y, z, colour code
        self._n = 0
        self._live_gen = 0
        self._static_gen = 0
        self.car_pos = None

        self._layer = None
        self._layer_key = None
        self._drawn = 0
        self.setMinimumSize(200, 200)

    # ---- interaction ----
    def mousePressEvent(self, event):
        self.last_mouse_pos = event.position()

    def mouseReleaseEvent(self, event):
        self.last_mouse_pos = None

    def mouseMoveEvent(self, event):
        if self.last_mouse_pos is not None:
            pos = event.position()
            self.yaw += (pos.x() - self.last_mouse_pos.x()) * 0.01
            self.pitch = max(-1.5, min(1.5, self.pitch + (pos.y() - self.last_mouse_pos.y()) * 0.01))
            self.last_mouse_pos = pos
            self.update()

    # ---- bounds ----
    def _expand_bounds(self, x, z, pad=0.0):
        b = self.bounds
        if b is None:
            self.bounds = [x, x, z, z]
            return True
        changed = False
        rx, rz = max(b[1] - b[0], 50.0), max(b[3] - b[2], 50.0)
        if x < b[0]: b[0], changed = x - pad * rx, True
        if x > b[1]: b[1], changed = x + pad * rx, True
        if z < b[2]: b[2], changed = z - pad * rz, True
        if z > b[3]: b[3], changed = z + pad * rz, True
        return changed

    # ---- live mode ----
    def reset_live(self):
        """New session / track: forget the path and rescale to the new circuit."""
        self._n = 0
        self._live_gen += 1
        self.car_pos = None
        self.bounds = None
        self.update()

    def update_position(self, x, y, z, throttle, brake, is_new_lap):
        if is_new_lap:
            self._n = 0
            self._live_gen += 1
        self.car_pos = (x, y, z)
        self._expand_bounds(x, z, self.BOUNDS_PAD)
        code = 2 if brake > 0 else (1 if throttle > 0 else 0)
        n = self._n
        if n:
            lx, ly, lz = self._pts[n - 1, 0], self._pts[n - 1, 1], self._pts[n - 1, 2]
            if (x - lx) ** 2 + (y - ly) ** 2 + (z - lz) ** 2 < self.LIVE_MIN_STEP_M ** 2:
                self.update()
                return
        if n == len(self._pts):
            self._pts = np.concatenate([self._pts, np.empty_like(self._pts)])
        self._pts[n] = (x, y, z, code)
        self._n = n + 1
        self.update()

    # ---- analysis mode ----
    def set_laps(self, lap_a, lap_b=None):
        self.lap_a = lap_a if (lap_a is not None and len(lap_a)) else None
        self.lap_b = lap_b if (lap_b is not None and len(lap_b)) else None
        self.bounds = None
        for lap in (self.lap_a, self.lap_b):
            if lap is not None:
                self._expand_bounds(float(lap['x'].min()), float(lap['z'].min()))
                self._expand_bounds(float(lap['x'].max()), float(lap['z'].max()))
        self._static_gen += 1
        self.update()

    # ---- rendering ----
    def _view(self):
        w, h = self.width(), self.height()
        b = self.bounds
        cx, cz = (b[0] + b[1]) / 2, (b[2] + b[3]) / 2
        scale = min(w, h) / max(b[1] - b[0], b[3] - b[2], 1.0) * 0.8
        return w, h, cx, cz, scale

    @staticmethod
    def _stride(n, limit):
        return max(1, int(math.ceil(n / float(limit))))

    def _new_layer(self, w, h, dpr):
        layer = QPixmap(max(1, int(w * dpr)), max(1, int(h * dpr)))
        layer.setDevicePixelRatio(dpr)
        layer.fill(Qt.GlobalColor.transparent)
        return layer

    def _render_static(self, painter, w, h, cx, cz, scale):
        for lap, color in ((self.lap_a, QColor(212, 175, 55, 200)), (self.lap_b, QColor(255, 255, 255, 120))):
            if lap is None:
                continue
            s = self._stride(len(lap['x']), self.MAX_DRAW_POINTS)
            x = lap['x'][::s].astype(np.float64)
            y = lap['y'][::s].astype(np.float64)
            z = lap['z'][::s].astype(np.float64)
            px, py = project_3d_np(x, y, z, cx, cz, scale, self.yaw, self.pitch)
            painter.setPen(QPen(color, 4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            painter.drawPolyline(make_polygon(px + w / 2, py + h / 2))

    def _render_live_full(self, painter, w, h, cx, cz, scale):
        n = self._n
        if n < 2:
            return
        arr = self._pts[:n]
        s = self._stride(n, self.MAX_DRAW_POINTS)
        idx = np.arange(0, n, s)
        if idx[-1] != n - 1:
            idx = np.append(idx, n - 1)
        a = arr[idx]
        px, py = project_3d_np(a[:, 0], a[:, 1], a[:, 2], cx, cz, scale, self.yaw, self.pitch)
        px, py = px + w / 2, py + h / 2
        codes = a[:, 3].astype(np.int8)
        cuts = (np.flatnonzero(np.diff(codes)) + 1).tolist()
        edges = sorted(set([1] + [c for c in cuts if c >= 1] + [len(a)]))
        for st, en in zip(edges[:-1], edges[1:]):          # segment i joins point i-1 -> i, coloured by point i
            painter.setPen(QPen(self._LIVE_COLORS[int(codes[st])], 3, Qt.PenStyle.SolidLine,
                                Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            painter.drawPolyline(make_polygon(px[st - 1:en], py[st - 1:en]))
        self._drawn = n

    def _render_live_incremental(self, painter, w, h, cx, cz, scale):
        n, k = self._n, self._drawn
        if n - k < 1 or n < 2:
            return
        arr = self._pts[max(k - 1, 0):n]
        px, py = project_3d_np(arr[:, 0], arr[:, 1], arr[:, 2], cx, cz, scale, self.yaw, self.pitch)
        px, py = px + w / 2, py + h / 2
        codes = arr[:, 3].astype(np.int8)
        last_code = None
        for i in range(1, len(arr)):
            c = int(codes[i])
            if c != last_code:
                painter.setPen(QPen(self._LIVE_COLORS[c], 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                last_code = c
            painter.drawLine(QPointF(px[i - 1], py[i - 1]), QPointF(px[i], py[i]))
        self._drawn = n

    def paintEvent(self, event):
        painter = QPainter(self)
        if self.bounds is None:
            painter.end()
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h, cx, cz, scale = self._view()
        dpr = self.devicePixelRatioF()
        key = (w, h, dpr, round(self.yaw, 4), round(self.pitch, 4), tuple(self.bounds),
               self._live_gen if self.is_live else self._static_gen)

        if key != self._layer_key or self._layer is None:                 # full re-render (rare)
            self._layer = self._new_layer(w, h, dpr)
            lp = QPainter(self._layer)
            lp.setRenderHint(QPainter.RenderHint.Antialiasing)
            if self.is_live:
                self._drawn = 0
                self._render_live_full(lp, w, h, cx, cz, scale)
            else:
                self._render_static(lp, w, h, cx, cz, scale)
            lp.end()
            self._layer_key = key
        elif self.is_live and self._n > self._drawn:                       # append only the new segments
            lp = QPainter(self._layer)
            lp.setRenderHint(QPainter.RenderHint.Antialiasing)
            self._render_live_incremental(lp, w, h, cx, cz, scale)
            lp.end()

        painter.drawPixmap(0, 0, self._layer)
        x_off, y_off = w / 2, h / 2

        if self.is_live:
            if self.car_pos is not None:
                car_x, car_y = project_3d(*self.car_pos, cx, cz, scale, self.yaw, self.pitch)
                painter.setBrush(QBrush(QColor(212, 175, 55)))
                painter.setPen(QPen(QColor(0, 0, 0), 2))
                painter.drawEllipse(QPointF(car_x + x_off, car_y + y_off), 7, 7)
        else:
            for lap, color in ((self.lap_a, QColor(212, 175, 55)), (self.lap_b, QColor(255, 255, 255))):
                if lap is None:
                    continue
                idx = int(min(np.searchsorted(lap['t'], self.playback_time_ms), len(lap['t']) - 1))
                mx, my = project_3d(float(lap['x'][idx]), float(lap['y'][idx]), float(lap['z'][idx]),
                                    cx, cz, scale, self.yaw, self.pitch)
                painter.setBrush(QBrush(color))
                painter.setPen(QPen(QColor(0, 0, 0), 2))
                painter.drawEllipse(QPointF(mx + x_off, my + y_off), 8, 8)
        painter.end()
