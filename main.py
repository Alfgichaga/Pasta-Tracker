import sys
import os
import gc
import json
import time
import uuid
import math
import ipaddress

import numpy as np
import pyqtgraph as pg

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QProgressBar, QLineEdit,
                             QPushButton, QFrame, QTabWidget, QTreeWidget, QTreeWidgetItem,
                             QAbstractItemView, QInputDialog, QSizePolicy, QSlider, QComboBox)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPixmap, QIcon

from car_database import CAR_DB, TRACKS, data_path, read_json, atomic_write_json
from backend import LiveGT7Thread, LapCache, LapTracker, CatalogUpdater
from custom_widgets import Interactive3DMapWidget, ScrollingTraceWidget, TireTelemetryWidget, RpmBarWidget

SETTINGS_FILE = data_path("gt7_settings.json")
MAX_UNSAVED_LAPS = 40          # un-kept laps beyond this are dropped oldest-first (bounds memory use)

STATUS_STYLE = "color: {c}; font-weight: bold; padding-left: 10px; font-size: {s}px;"


class RaceModeTab(QWidget):
    def __init__(self):
        super().__init__()
        main_layout = QHBoxLayout(self)

        def create_header(text):
            lbl = QLabel(text)
            lbl.setObjectName("panel_header")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            return lbl

        left_col = QVBoxLayout()
        map_frame = QFrame()
        map_layout = QVBoxLayout(map_frame)
        self.track_map = Interactive3DMapWidget(is_live=True)
        map_layout.addWidget(create_header("3D ELEVATION MAP"))
        map_layout.addWidget(self.track_map, stretch=1)
        left_col.addWidget(map_frame)

        center_col = QVBoxLayout()
        pt_frame = QFrame()
        pt_layout = QVBoxLayout(pt_frame)
        pt_layout.addWidget(create_header("POWERTRAIN"))

        speed_gear_layout = QHBoxLayout()
        self.speed_lbl = QLabel("0")
        self.speed_lbl.setFont(QFont("Consolas", 60, QFont.Weight.Bold))
        self.speed_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gear_lbl = QLabel("N")
        self.gear_lbl.setFont(QFont("Consolas", 45, QFont.Weight.Bold))
        self.gear_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gear_lbl.setStyleSheet("color: #D4AF37; font-weight: bold;")

        speed_gear_layout.addWidget(self.speed_lbl)
        speed_gear_layout.addWidget(self.gear_lbl)
        pt_layout.addLayout(speed_gear_layout)

        self.rpm_cap = 8500
        self.rpm_bar = RpmBarWidget()
        self.rpm_bar.setRange(0, self.rpm_cap)
        pt_layout.addWidget(QLabel("RPM LINE", alignment=Qt.AlignmentFlag.AlignCenter))
        pt_layout.addWidget(self.rpm_bar)

        in_frame = QFrame()
        in_layout = QVBoxLayout(in_frame)
        self.trace_widget = ScrollingTraceWidget()
        in_layout.addWidget(create_header("PEDAL INPUTS"))
        in_layout.addWidget(self.trace_widget)

        center_col.addWidget(pt_frame, stretch=1)
        center_col.addWidget(in_frame, stretch=1)

        right_col = QVBoxLayout()
        sess_frame = QFrame()
        sess_layout = QVBoxLayout(sess_frame)

        self.delta_lbl = QLabel("+0.000")
        self.delta_lbl.setFont(QFont("Consolas", 32, QFont.Weight.Bold))
        self.delta_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.cur_lap_lbl = QLabel("00:00.000")
        self.cur_lap_lbl.setFont(QFont("Consolas", 26, QFont.Weight.Bold))
        self.cur_lap_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.best_lap_lbl = QLabel("00:00.000")
        self.best_lap_lbl.setFont(QFont("Consolas", 26, QFont.Weight.Bold))
        self.best_lap_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.best_lap_lbl.setStyleSheet("color: #D4AF37;")

        self.tire_widget = TireTelemetryWidget()

        sess_layout.addWidget(create_header("LIVE DELTA"))
        sess_layout.addWidget(self.delta_lbl)

        sess_layout.addWidget(create_header("SESSION LAPS"))
        laps_row = QHBoxLayout()

        cur_col = QVBoxLayout()
        cur_col.addWidget(QLabel("CURRENT", alignment=Qt.AlignmentFlag.AlignCenter, styleSheet="color: #AAAAAA; font-size: 11px;"))
        cur_col.addWidget(self.cur_lap_lbl)

        best_col = QVBoxLayout()
        best_col.addWidget(QLabel("SESSION BEST", alignment=Qt.AlignmentFlag.AlignCenter, styleSheet="color: #AAAAAA; font-size: 11px;"))
        best_col.addWidget(self.best_lap_lbl)

        laps_row.addLayout(cur_col)
        laps_row.addLayout(best_col)
        sess_layout.addLayout(laps_row)

        sess_layout.addWidget(create_header("TIRE TEMPS"))
        sess_layout.addWidget(self.tire_widget)
        sess_layout.addStretch()
        right_col.addWidget(sess_frame)

        main_layout.addLayout(left_col, stretch=4)
        main_layout.addLayout(center_col, stretch=3)
        main_layout.addLayout(right_col, stretch=3)

    def set_rpm_cap(self, cap):
        self.rpm_cap = cap
        self.rpm_bar.setRange(0, cap)


class AnalyzerTab(QWidget):
    track_changed = pyqtSignal(str)       # name of the track of the most recent lap

    def __init__(self):
        super().__init__()
        self.lap_cache = LapCache()
        self.lap_database = self.lap_cache.laps.copy()

        self.lap_a_frames = None
        self.lap_b_frames = None
        self.max_time_ms = 0

        layout = QHBoxLayout(self)

        sidebar = QFrame()
        sidebar.setMinimumWidth(260)
        sidebar.setMaximumWidth(350)
        sidebar_layout = QVBoxLayout(sidebar)

        lbl = QLabel("TELEMETRY VAULTS")
        lbl.setObjectName("panel_header")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sidebar_layout.addWidget(lbl)

        self.lap_tree = QTreeWidget()
        self.lap_tree.setHeaderHidden(True)
        self.lap_tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.lap_tree.itemSelectionChanged.connect(self.plot_selected_laps)
        sidebar_layout.addWidget(self.lap_tree)

        btn_layout = QHBoxLayout()
        self.save_btn = QPushButton("KEEP LAP")
        self.ren_btn = QPushButton("RENAME")
        self.del_btn = QPushButton("DELETE")
        for btn in [self.save_btn, self.ren_btn, self.del_btn]:
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.setStyleSheet("font-size: 11px; padding: 6px;")
        self.save_btn.clicked.connect(self.keep_lap)
        self.ren_btn.clicked.connect(self.rename_lap)
        self.del_btn.clicked.connect(self.delete_lap)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.ren_btn)
        btn_layout.addWidget(self.del_btn)
        sidebar_layout.addLayout(btn_layout)

        self.track_btn = QPushButton("SET TRACK")
        self.track_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.track_btn.setStyleSheet("font-size: 11px; padding: 6px;")
        self.track_btn.setToolTip("Name the track for the selected lap. Every lap recorded on the same "
                                  "track (past and future) is then recognised automatically.")
        self.track_btn.clicked.connect(self.set_track)
        sidebar_layout.addWidget(self.track_btn)

        sidebar_layout.addWidget(QLabel("Ctrl+Click to compare 2 laps", alignment=Qt.AlignmentFlag.AlignCenter, styleSheet="color:#777; font-size: 11px;"))
        layout.addWidget(sidebar)

        main_pane = QVBoxLayout()
        graph_map_split = QHBoxLayout()

        pg.setConfigOptions(antialias=True, useOpenGL=False)
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setBackground('transparent')

        # SPEED PLOT
        self.speed_plot = self.plot_widget.addPlot(row=0, col=0)
        self.speed_plot.showGrid(x=True, y=True, alpha=0.15)
        self.speed_plot.setLabel('left', 'Speed (km/h)', color='#AAAAAA')
        self.speed_plot.setMouseEnabled(x=True, y=False)
        self.speed_plot.getViewBox().setDefaultPadding(0.0)
        self.speed_plot.setLimits(xMin=0, xMax=100)

        # INPUT PLOT
        self.input_plot = self.plot_widget.addPlot(row=1, col=0)
        self.input_plot.showGrid(x=True, y=True, alpha=0.15)
        self.input_plot.setLabel('left', 'Inputs (%)', color='#AAAAAA')
        self.input_plot.setLabel('bottom', 'Lap Distance (%)', color='#AAAAAA')
        self.input_plot.setYRange(0, 105)
        self.input_plot.setXLink(self.speed_plot)
        self.input_plot.setMouseEnabled(x=True, y=False)
        self.input_plot.getViewBox().setDefaultPadding(0.0)
        self.input_plot.setLimits(xMin=0, xMax=100)

        # VERTICAL PLAYBACK LINE
        self.cursor_line_speed = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('w', width=2))
        self.cursor_line_input = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('w', width=2))
        self.speed_plot.addItem(self.cursor_line_speed)
        self.input_plot.addItem(self.cursor_line_input)

        graph_map_split.addWidget(self.plot_widget, stretch=5)

        map_frame = QFrame()
        map_layout = QVBoxLayout(map_frame)
        map_layout.setContentsMargins(0, 5, 0, 0)
        self.analyzer_map = Interactive3DMapWidget(is_live=False)
        lbl = QLabel("(Click & Drag to Rotate 3D Track)", alignment=Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("color:#AAAAAA; font-size: 11px; font-weight: bold;")
        map_layout.addWidget(lbl)
        map_layout.addWidget(self.analyzer_map, stretch=1)

        graph_map_split.addWidget(map_frame, stretch=4)
        main_pane.addLayout(graph_map_split, stretch=1)

        playback_layout = QHBoxLayout()
        self.play_btn = QPushButton("▶ PLAY")
        self.play_btn.setFixedWidth(100)
        self.play_btn.clicked.connect(self.toggle_playback)

        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.5x Speed", "1.0x Speed", "1.5x Speed", "2.0x Speed"])
        self.speed_combo.setCurrentIndex(1)
        self.speed_combo.setFixedWidth(120)

        self.playback_slider = QSlider(Qt.Orientation.Horizontal)
        self.playback_slider.setRange(0, 0)
        self.playback_slider.valueChanged.connect(self.slider_moved)

        playback_layout.addWidget(self.play_btn)
        playback_layout.addWidget(self.speed_combo)
        playback_layout.addWidget(self.playback_slider)
        main_pane.addLayout(playback_layout)

        layout.addLayout(main_pane, stretch=1)

        self._groups = {}
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.playback_tick)
        self.is_playing = False
        self.last_playback_realtime = 0
        self.refresh_list()

    # ---------------- helpers ----------------
    def _selected_lap_items(self):
        return [i for i in self.lap_tree.selectedItems() if i.data(0, Qt.ItemDataRole.UserRole) is not None]

    def _find_lap(self, lap_id):
        return next((l for l in self.lap_database if l['id'] == lap_id), None)

    def _reset_plots(self):
        self.speed_plot.clear()
        self.input_plot.clear()
        self.speed_plot.addItem(self.cursor_line_speed)
        self.input_plot.addItem(self.cursor_line_input)
        self.analyzer_map.set_laps(None, None)
        self.lap_a_frames = self.lap_b_frames = None
        self.max_time_ms = 0
        self.playback_slider.setRange(0, 0)

    # ---------------- playback ----------------
    def toggle_playback(self):
        if self.lap_a_frames is None:
            return
        self.is_playing = not self.is_playing
        if self.is_playing:
            if self.playback_slider.value() >= self.max_time_ms:
                self.playback_slider.setValue(0)
            self.play_btn.setText("⏸ PAUSE")
            self.play_btn.setStyleSheet("background-color: #D4AF37; color: #000000;")
            self.last_playback_realtime = time.perf_counter()
            self.playback_timer.start(16)
        else:
            self.play_btn.setText("▶ PLAY")
            self.play_btn.setStyleSheet("")
            self.playback_timer.stop()

    def playback_tick(self):
        now = time.perf_counter()
        dt_seconds = now - self.last_playback_realtime
        self.last_playback_realtime = now

        speed_multiplier = float(self.speed_combo.currentText().split('x')[0])
        new_val = self.playback_slider.value() + dt_seconds * 1000.0 * speed_multiplier

        if new_val >= self.max_time_ms:
            new_val = self.max_time_ms
            self.toggle_playback()
        self.playback_slider.setValue(int(new_val))

    def slider_moved(self, value):
        if self.lap_a_frames is None:
            return
        self.analyzer_map.playback_time_ms = value
        self.analyzer_map.update()

        t = self.lap_a_frames['t']
        idx = int(min(np.searchsorted(t, value), len(t) - 1))
        percent_a = float(self.lap_a_frames['pct'][idx])
        self.cursor_line_speed.setValue(percent_a)
        self.cursor_line_input.setValue(percent_a)

    # ---------------- lap database ----------------
    def add_lap(self, lap_num, lap_time_ms, lap_data, lap_type, car_id):
        name = f"Lap {lap_num} - {LapTracker.format_time(lap_time_ms)}"
        if lap_type != "HOT":
            name = f"[{lap_type}] {name}"

        # Names are NOT baked into the lap any more: only car_id and the finish-line fingerprint are
        # stored, and names are resolved when the list is drawn - so catalogue updates and newly named
        # tracks fix every lap, including ones saved long ago.
        fp = TRACKS.make_fingerprint(lap_data, lap_type != "OUT")

        self.lap_database.append({
            'id': uuid.uuid4().hex, 'name': name, 'perm': False,
            'time_ms': lap_time_ms, 'car_id': car_id, 'type': lap_type,
            'fp': fp, 'data': lap_data,
        })

        unsaved = [l for l in self.lap_database if not l.get('perm', False)]
        if len(unsaved) > MAX_UNSAVED_LAPS:
            drop = {id(l) for l in unsaved[:len(unsaved) - MAX_UNSAVED_LAPS]}
            dropped_ids = {l['id'] for l in self.lap_database if id(l) in drop}
            self.lap_database = [l for l in self.lap_database if id(l) not in drop]
            self._remove_lap_items(dropped_ids)    # remove just the oldest row(s), no rebuild
        self._add_lap_item(self.lap_database[-1])  # normal case: add ONE row, rebuild nothing
        self.track_changed.emit(TRACKS.display_name(fp))

    def _new_track_item(self, track):
        item = QTreeWidgetItem([track])
        item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        item.setForeground(0, QColor("#D4AF37"))
        # named tracks sort before "Unidentified ..." groups
        idx = self.lap_tree.topLevelItemCount()
        if not track.startswith("Unidentified"):
            for i in range(self.lap_tree.topLevelItemCount()):
                if self.lap_tree.topLevelItem(i).text(0).startswith("Unidentified"):
                    idx = i
                    break
        self.lap_tree.insertTopLevelItem(idx, item)
        item.setExpanded(True)
        return item

    def _new_car_item(self, track_item, car):
        item = QTreeWidgetItem(track_item, [f"Vehicle: {car}"])
        item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        item.setForeground(0, QColor("#FFFFFF"))
        item.setExpanded(True)
        return item

    def _add_lap_item(self, lap, selected_ids=()):
        track = TRACKS.display_name(lap.get('fp'))
        car = CAR_DB.name(lap.get('car_id'))
        track_item = self._groups.get(track)
        if track_item is None:
            track_item = self._groups[track] = self._new_track_item(track)
        car_item = self._groups.get((track, car))
        if car_item is None:
            car_item = self._groups[(track, car)] = self._new_car_item(track_item, car)
        prefix = "★ " if lap.get('perm', False) else "  "
        lap_item = QTreeWidgetItem(car_item, [prefix + lap['name']])
        lap_item.setData(0, Qt.ItemDataRole.UserRole, lap['id'])
        if lap['id'] in selected_ids:
            lap_item.setSelected(True)

    def _remove_lap_items(self, lap_ids):
        """Remove specific lap rows (and any group left empty) without rebuilding the tree."""
        selection_changed = False
        for ti in range(self.lap_tree.topLevelItemCount() - 1, -1, -1):
            track_item = self.lap_tree.topLevelItem(ti)
            for ci in range(track_item.childCount() - 1, -1, -1):
                car_item = track_item.child(ci)
                for li in range(car_item.childCount() - 1, -1, -1):
                    lap_item = car_item.child(li)
                    if lap_item.data(0, Qt.ItemDataRole.UserRole) in lap_ids:
                        selection_changed |= lap_item.isSelected()
                        car_item.removeChild(lap_item)
                if car_item.childCount() == 0:
                    track_item.removeChild(car_item)
                    self._groups = {k: v for k, v in self._groups.items() if v is not car_item}
            if track_item.childCount() == 0:
                self.lap_tree.takeTopLevelItem(ti)
                self._groups = {k: v for k, v in self._groups.items() if v is not track_item}
        if selection_changed:
            self.plot_selected_laps()

    def refresh_list(self):
        """Full rebuild - only for rare operations (keep / rename / delete / set track / catalogue update)."""
        selected_ids = {i.data(0, Qt.ItemDataRole.UserRole) for i in self._selected_lap_items()}
        self.lap_tree.setUpdatesEnabled(False)     # one repaint at the end instead of one per item
        self.lap_tree.blockSignals(True)           # rebuilding must not wipe the user's selection/plots
        self.lap_tree.clear()
        self._groups = {}
        for lap in self.lap_database:
            self._add_lap_item(lap, selected_ids)
        self.lap_tree.blockSignals(False)
        self.lap_tree.setUpdatesEnabled(True)

        now_ids = {i.data(0, Qt.ItemDataRole.UserRole) for i in self._selected_lap_items()}
        if now_ids != selected_ids:                # a selected lap disappeared (deleted / trimmed)
            self.plot_selected_laps()

    def keep_lap(self):
        for item in self._selected_lap_items():
            lap = self._find_lap(item.data(0, Qt.ItemDataRole.UserRole))
            if lap:
                lap['perm'] = True
                self.lap_cache.save_lap(lap)       # queued to a background thread - returns instantly
        self.refresh_list()

    def rename_lap(self):
        selected = self._selected_lap_items()
        if not selected:
            return
        lap_obj = self._find_lap(selected[0].data(0, Qt.ItemDataRole.UserRole))
        if not lap_obj:
            return
        text, ok = QInputDialog.getText(self, "Rename Lap", "New Name:", text=lap_obj['name'])
        if ok and text.strip():
            lap_obj['name'] = text.strip()
            if lap_obj.get('perm'):
                self.lap_cache.save_lap(lap_obj)
            self.refresh_list()

    def delete_lap(self):
        selected = self._selected_lap_items()
        if not selected:
            return
        ids = {i.data(0, Qt.ItemDataRole.UserRole) for i in selected}
        for l in self.lap_database:
            if l['id'] in ids and l.get('perm'):
                self.lap_cache.delete_lap(l['id'])
        self.lap_database = [l for l in self.lap_database if l['id'] not in ids]
        self._reset_plots()
        self.refresh_list()

    def set_track(self):
        """Name the track of the selected lap once; every matching lap is recognised from then on."""
        selected = self._selected_lap_items()
        lap = self._find_lap(selected[0].data(0, Qt.ItemDataRole.UserRole)) if selected else None
        if not lap or not lap.get('fp'):
            self.track_changed.emit("Select a lap first, then press SET TRACK")
            return
        fp = lap['fp']
        options = TRACKS.suggestions(fp)
        labels = [lbl for lbl, _ in options]
        hint = ("Closest lap length first (* = likely). You can also type your own name.\n"
                f"Measured lap length: {fp['len']:.0f} m" if fp.get('full') else
                "Pick the track, or type your own name.")
        choice, ok = QInputDialog.getItem(self, "Set Track", hint, labels or [""], 0, True)
        if not ok or not choice.strip():
            return
        label_to_name = {lbl: nm for lbl, nm in options}
        name = label_to_name.get(choice, choice.strip())
        if TRACKS.learn(fp, name):
            self.refresh_list()
            self.track_changed.emit(name)

    def plot_selected_laps(self):
        selected = self._selected_lap_items()
        if len(selected) > 2:                   # keep only the two most recently clicked
            self.lap_tree.blockSignals(True)
            selected[0].setSelected(False)
            self.lap_tree.blockSignals(False)
            selected = self._selected_lap_items()

        self._reset_plots()
        if not selected:
            return

        colors = [(212, 175, 55), (255, 255, 255)]
        slots = [None, None]
        for idx, item in enumerate(selected[:2]):
            lap_obj = self._find_lap(item.data(0, Qt.ItemDataRole.UserRole))
            if not lap_obj or lap_obj.get('data') is None:
                continue
            frames = lap_obj['data']
            pen_c = pg.mkPen(color=colors[idx], width=2)
            curves = [
                self.speed_plot.plot(frames['pct'], frames['s'], pen=pen_c),
                self.input_plot.plot(frames['pct'], frames['th'], pen=pen_c),
                self.input_plot.plot(frames['pct'], frames['br'], pen=pg.mkPen(
                    color=(255, 61, 0), width=2,
                    style=Qt.PenStyle.DotLine if idx == 1 else Qt.PenStyle.SolidLine)),
            ]
            for curve in curves:               # set AFTER the curve is inside its plot (setting at creation errors)
                curve.setDownsampling(auto=True, method='peak')
                curve.setClipToView(True)
            slots[idx] = frames
            self.max_time_ms = max(self.max_time_ms, float(frames['t'][-1]))

        self.lap_a_frames, self.lap_b_frames = slots
        if self.lap_a_frames is None:           # only the 2nd slot loaded: promote it
            self.lap_a_frames, self.lap_b_frames = self.lap_b_frames, None

        self.playback_slider.setRange(0, int(self.max_time_ms))
        self.analyzer_map.set_laps(self.lap_a_frames, self.lap_b_frames)
        self.speed_plot.setXRange(0, 100.0, padding=0.0)
        self.input_plot.setXRange(0, 100.0, padding=0.0)


class GT7Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GT7 Telemetry Suite - #41 (GTSA)")

        script_dir = os.path.dirname(os.path.abspath(__file__))
        pasta_path = os.path.join(script_dir, "pasta_logo_transparent.png")
        sagt_path = os.path.join(script_dir, "SAGT logo.jpeg")

        if os.path.exists(pasta_path):
            self.setWindowIcon(QIcon(pasta_path))

        self.resize(1350, 800)

        self.setStyleSheet("""
            QMainWindow { background-color: #000000; }
            QTabWidget::pane { border: none; top: -1px; }
            QTabBar::tab { background: #111111; color: #888888; padding: 14px 40px; font-family: Consolas; font-weight: bold; font-size: 15px; border-top-left-radius: 8px; border-top-right-radius: 8px; margin-right: 4px;}
            QTabBar::tab:selected { background: #1A1A1A; color: #D4AF37; border-bottom: 3px solid #D4AF37; }
            QTabBar::tab:hover { color: #FFFFFF; }

            QFrame { background-color: #0A0A0A; border-radius: 8px; border: 1px solid #222222; }
            QLabel { color: #FFFFFF; font-family: Consolas, monospace; border: none; }

            QLabel#panel_header { color: #000000; font-size: 15px; font-weight: bold; background-color: #D4AF37; padding: 6px; border-radius: 4px; }

            QPushButton { background-color: #111111; border: 1px solid #D4AF37; border-radius: 4px; padding: 10px; color: #D4AF37; font-weight: bold; font-size: 14px; }
            QPushButton:hover { background-color: #D4AF37; color: #000000; }

            QLineEdit { background-color: #111111; border: 1px solid #333333; border-radius: 4px; padding: 8px; color: white; font-family: Consolas; }
            QComboBox { background-color: #111111; color: #D4AF37; border: 1px solid #D4AF37; border-radius: 4px; padding: 5px; }

            QTreeWidget { background-color: #0A0A0A; border: none; outline: 0; font-size: 14px; color: white;}
            QTreeWidget::item { padding: 8px; border-bottom: 1px solid #222222; }
            QTreeWidget::item:selected { background-color: #111111; color: #D4AF37; font-weight: bold; border-left: 4px solid #D4AF37; }

            QSlider::groove:horizontal { border: 1px solid #333333; height: 8px; background: #000000; margin: 2px 0; border-radius: 4px; }
            QSlider::handle:horizontal { background: #D4AF37; border: 1px solid #FFFFFF; width: 14px; margin: -4px 0; border-radius: 7px; }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        header_layout = QHBoxLayout()

        self.pasta_logo_lbl = QLabel()
        if os.path.exists(pasta_path):
            self.pasta_logo_lbl.setPixmap(QPixmap(pasta_path).scaled(
                80, 80, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            self.pasta_logo_lbl.setText("[ PASTA LOGO MISSING ]")
            self.pasta_logo_lbl.setStyleSheet("color: #FF3D00; font-weight: bold; border: 1px dashed #FF3D00; padding: 5px;")
        header_layout.addWidget(self.pasta_logo_lbl)

        self.sagt_logo_lbl = QLabel()
        if os.path.exists(sagt_path):
            self.sagt_logo_lbl.setPixmap(QPixmap(sagt_path).scaled(
                200, 80, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            self.sagt_logo_lbl.setText("[ SAGT LOGO MISSING ]")
            self.sagt_logo_lbl.setStyleSheet("color: #FF3D00; font-weight: bold; border: 1px dashed #FF3D00; padding: 5px;")
        header_layout.addWidget(self.sagt_logo_lbl)
        header_layout.addStretch()

        saved_ip = str(read_json(SETTINGS_FILE, {}).get("ps_ip", ""))
        self.ip_input = QLineEdit(saved_ip)
        self.ip_input.setPlaceholderText("192.168.x.x")
        self.ip_input.setFixedWidth(160)
        self.ip_input.returnPressed.connect(self.update_ip)

        self.update_btn = QPushButton("CONNECT")
        self.update_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.update_btn.clicked.connect(self.update_ip)

        self.status_lbl = QLabel("● DISCONNECTED")
        self._status_state = None
        self._set_status("disconnected", "● DISCONNECTED", "#FF3D00", 16)

        header_layout.addWidget(QLabel("PS5 IP:", styleSheet="font-weight: bold; font-size: 14px;"))
        header_layout.addWidget(self.ip_input)
        header_layout.addWidget(self.update_btn)
        header_layout.addWidget(self.status_lbl)
        main_layout.addLayout(header_layout)

        # Live "what am I driving, where" strip
        self.info_lbl = QLabel("CAR: —     TRACK: —")
        self.info_lbl.setStyleSheet("color: #D4AF37; font-weight: bold; font-size: 14px; padding: 2px 6px;")
        main_layout.addWidget(self.info_lbl)
        self._car_text, self._track_text, self._car_id = "—", "—", None
        self._net_text = ""

        self.tabs = QTabWidget()
        self.race_tab = RaceModeTab()
        self.analyzer_tab = AnalyzerTab()
        self.analyzer_tab.track_changed.connect(self.on_track_changed)

        self.tabs.addTab(self.race_tab, "RACE DASHBOARD")
        self.tabs.addTab(self.analyzer_tab, "TELEMETRY ANALYZER")
        main_layout.addWidget(self.tabs)

        self._delta_style = None
        self.telemetry_thread = LiveGT7Thread(self.ip_input.text())
        self.telemetry_thread.lap_completed.connect(self.analyzer_tab.add_lap)
        self.telemetry_thread.error_occurred.connect(self.display_error)
        # Updates are driven by the packets themselves ("data ready" fires at most once until collected), so the
        # display is phase-locked to the PS5's 60 Hz stream instead of beating against a separate UI timer.
        self.telemetry_thread.data_ready.connect(self.poll_telemetry)
        self.telemetry_thread.start(QThread.Priority.HighPriority)

        # Safety net only (e.g. a missed notification); normal updates never wait for this
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self.poll_telemetry)
        self.poll_timer.start(100)

        # Refresh car/track catalogue in the background (at most once a week, silent when offline)
        self.catalog_updater = CatalogUpdater()
        self.catalog_updater.finished_ok.connect(self.on_catalog_updated)
        self.catalog_updater.start()

    # ---------------- status helpers ----------------
    def _set_status(self, state, text, color, size):
        if state == self._status_state and self.status_lbl.text() == text:
            return
        self._status_state = state
        self.status_lbl.setText(text)
        self.status_lbl.setStyleSheet(STATUS_STYLE.format(c=color, s=size))

    def _refresh_info(self):
        net = f"     |     LINK: {self._net_text}" if self._net_text else ""
        self.info_lbl.setText(f"CAR: {self._car_text}     TRACK: {self._track_text}{net}")

    def on_track_changed(self, name):
        self._track_text = name
        self._refresh_info()

    def on_catalog_updated(self, updated):
        if updated:
            CAR_DB.reload()
            TRACKS.reload_courses()
            self.analyzer_tab.refresh_list()
            if self._car_id is not None and self._car_id > 0:
                self._car_text = CAR_DB.name(self._car_id)
                self._refresh_info()

    # ---------------- connection ----------------
    def update_ip(self):
        new_ip = self.ip_input.text().strip()
        try:
            ipaddress.ip_address(new_ip)
        except ValueError:
            self._set_status("error", "● INVALID IP ADDRESS", "#FF3D00", 14)
            return
        atomic_write_json(SETTINGS_FILE, {"ps_ip": new_ip})
        self._set_status("connecting", "● CONNECTING...", "#D4AF37", 16)
        self.telemetry_thread.set_ip(new_ip)

    def poll_telemetry(self):
        snap = self.telemetry_thread.take_snapshot()
        if snap is not None:
            self.update_ui(snap)

    def display_error(self, error_msg):
        self._set_status("error", f"● {error_msg}", "#FF3D00", 14)

    def update_ui(self, data):
        if not data.get('connected', False):
            self._set_status("disconnected", "● DISCONNECTED", "#FF3D00", 16)
            if self._net_text:
                self._net_text = ""
                self._refresh_info()
            return

        if data.get('paused'):
            self._set_status("paused", "● PAUSED / LOADING", "#D4AF37", 16)
        else:
            self._set_status("active", "● DATA STREAM ACTIVE", "#00E676", 16)

        rt = self.race_tab

        car_id = data.get('car_id')
        if car_id != self._car_id:
            self._car_id = car_id
            self._car_text = CAR_DB.name(car_id) if car_id and car_id > 0 else "—"
            rt.set_rpm_cap(8500)               # new car: re-learn its rev range
            self._refresh_info()

        if data['rpm'] > rt.rpm_cap:           # bar never saturates on high-revving cars
            rt.set_rpm_cap(int(math.ceil(data['rpm'] / 500.0) * 500))

        if data.get('session_reset'):
            rt.track_map.reset_live()

        # Every packet since the last frame is applied in order (not just the newest), so pedal traces and the
        # map path are complete and evenly spaced even when Wi-Fi delivers packets in bursts.
        samples = data.get('samples')
        if samples:
            for x, y, z, th, br, new_lap in samples:
                rt.track_map.update_position(x, y, z, th, br, new_lap)
                rt.trace_widget.update_trace(th, br)
        else:
            rt.track_map.update_position(data['pos_x'], data['pos_y'], data['pos_z'],
                                         data['throttle'], data['brake'], data['is_new_lap'])

        net = data.get('net')
        if net and net['hz']:
            text = f"{net['hz']} Hz · loss {net['lost_pct']:.1f}% · max gap {net['gap_ms']} ms"
            if text != self._net_text:
                self._net_text = text
                self._refresh_info()

        if not rt.isVisible():                 # analyzer tab open / window minimised: skip all label work
            return

        def put(label, text):                  # touching a QLabel re-lays-out the tab, so only when changed
            if label.text() != text:
                label.setText(text)

        put(rt.speed_lbl, str(data['speed']))
        put(rt.gear_lbl, str(data['gear']))
        rt.rpm_bar.setValue(min(data['rpm'], rt.rpm_cap))
        rt.tire_widget.update_temps(data['temps'])
        put(rt.cur_lap_lbl, data['cur_lap'])
        put(rt.best_lap_lbl, data['best_lap'])
        put(rt.delta_lbl, f"{data['delta']:+.3f}")

        if not data['has_best'] or data['delta'] == 0.0:
            style = "color: #AAAAAA;"
        elif data['delta'] < 0:
            style = "color: #00E676;"
        else:
            style = "color: #FF3D00;"
        if style != self._delta_style:         # only touch the stylesheet when it actually changes
            self._delta_style = style
            rt.delta_lbl.setStyleSheet(style)

    def closeEvent(self, event):
        self.poll_timer.stop()
        self.telemetry_thread.stop()
        self.telemetry_thread.wait(2000)
        self.catalog_updater.wait(2000)
        self.analyzer_tab.lap_cache.flush()        # let queued lap writes finish
        super().closeEvent(event)


def tune_runtime():
    """Lower input-to-screen latency of the telemetry path."""
    sys.setswitchinterval(0.001)               # let the network thread grab the GIL within 1 ms (default 5 ms)
    if os.name == "nt":                        # Windows: default timer tick is ~15.6 ms; ask for 1 ms
        try:
            import ctypes
            ctypes.windll.winmm.timeBeginPeriod(1)
        except Exception:
            pass


if __name__ == "__main__":
    tune_runtime()
    app = QApplication(sys.argv)
    window = GT7Dashboard()
    window.show()
    # Everything built so far is long-lived: park it so the garbage collector never rescans it mid-race,
    # and make the remaining collections rare (per-frame code allocates very few container objects now).
    gc.collect()
    gc.freeze()
    gc.set_threshold(50000, 20, 100)
    sys.exit(app.exec())
