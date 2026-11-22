import os
import time
import json
import uuid
import queue
import select
import socket
import struct
import threading

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from Crypto.Cipher import Salsa20

from car_database import data_path, read_json, atomic_write_json, refresh_catalog, TRACKS

# ==========================================================================
# Lap data: stored as columns (numpy arrays) instead of one dict per frame.
# ~15x less memory, much faster plotting, and far smaller save files.
# ==========================================================================
FRAME_KEYS = ('t', 'd', 'pct', 'x', 'y', 'z', 's', 'th', 'br')
_F64_KEYS = ('t', 'd')
_ROUND = {'t': 1, 'd': 2, 'pct': 3, 'x': 2, 'y': 2, 'z': 2, 's': 2, 'th': 0, 'br': 0}


def build_lap_data(rows):
    """rows: (N, 8) array of t_ms, dist_m, x, y, z, speed_kmh, throttle, brake."""
    a = np.asarray(rows, dtype=np.float64)
    d = a[:, 1].copy()
    total = max(float(d[-1]), 1e-4)
    return {
        't': a[:, 0].copy(), 'd': d,
        'pct': ((d / total) * 100.0).astype(np.float32),
        'x': a[:, 2].astype(np.float32), 'y': a[:, 3].astype(np.float32), 'z': a[:, 4].astype(np.float32),
        's': a[:, 5].astype(np.float32), 'th': a[:, 6].astype(np.float32), 'br': a[:, 7].astype(np.float32),
    }


def lap_data_from_json(raw):
    """Accepts the new column format and the legacy list-of-dicts format."""
    try:
        if isinstance(raw, dict):
            cols = {k: np.asarray(raw[k], dtype=np.float64 if k in _F64_KEYS else np.float32) for k in FRAME_KEYS}
        elif isinstance(raw, list) and raw:
            cols = {k: np.asarray([f[k] for f in raw], dtype=np.float64 if k in _F64_KEYS else np.float32)
                    for k in FRAME_KEYS}
        else:
            return None
    except (KeyError, TypeError, ValueError):
        return None
    n = len(cols['t'])
    if n < 2 or any(len(v) != n for v in cols.values()):
        return None
    return cols


def lap_data_to_json(data):
    out = {}
    for k in FRAME_KEYS:
        arr = np.round(data[k].astype(np.float64), _ROUND[k])
        out[k] = arr.astype(np.int64).tolist() if _ROUND[k] == 0 else arr.tolist()
    return out


class LapCache:
    """
    One small .npz file per kept lap (folder gt7_laps/).
      * keeping / renaming / deleting a lap touches ONE file, not the whole database
      * all disk IO happens on a background thread, so the UI never freezes while saving
      * loading is binary (no JSON parsing): ~20x faster start-up
      * a damaged file can only ever lose that one lap
    The old gt7_saved_laps.json is imported automatically the first time.
    """
    def __init__(self):
        self.dir = data_path("gt7_laps")
        os.makedirs(self.dir, exist_ok=True)
        self._q = queue.Queue()
        self._worker = threading.Thread(target=self._run, name="LapCacheWriter", daemon=True)
        self._worker.start()
        self._migrate_legacy()
        self.laps = self.load()

    # ---- paths / (de)serialisation ----
    def _path(self, lap_id):
        return os.path.join(self.dir, f"{lap_id}.npz")

    @staticmethod
    def _write_file(path, meta, data):
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            np.savez(f, meta=np.array(json.dumps(meta)), **{k: data[k] for k in FRAME_KEYS})
        os.replace(tmp, path)

    @staticmethod
    def _meta_of(lap):
        return {'id': lap['id'], 'name': lap['name'], 'time_ms': int(lap['time_ms']),
                'car_id': lap.get('car_id'), 'type': lap.get('type', 'HOT'), 'fp': lap.get('fp')}

    def _read_file(self, path):
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z['meta']))
            data = {k: z[k] for k in FRAME_KEYS}
        if len(data['t']) < 2:
            return None
        meta['perm'] = True
        meta['data'] = data
        if not meta.get('fp'):
            meta['fp'] = TRACKS.make_fingerprint(data, meta.get('type') != 'OUT')
        return meta

    # ---- loading ----
    def load(self):
        entries = []
        try:
            names = [n for n in os.listdir(self.dir) if n.endswith(".npz")]
        except OSError:
            return []
        for n in names:
            path = os.path.join(self.dir, n)
            try:
                lap = self._read_file(path)
                if lap:
                    entries.append((os.path.getmtime(path), lap))
            except Exception:
                continue                                # a damaged file never blocks the others
        entries.sort(key=lambda e: e[0])
        return [lap for _, lap in entries]

    def _migrate_legacy(self):
        for legacy in (data_path("gt7_saved_laps.json"), "gt7_saved_laps.json"):
            if not os.path.exists(legacy):
                continue
            raw = read_json(legacy, [])
            ok = True
            for item in raw if isinstance(raw, list) else []:
                try:
                    data = lap_data_from_json(item.get('data'))
                    if data is None:
                        continue
                    lap_type = item.get('type', 'HOT')
                    lap = {'id': str(item.get('id') or uuid.uuid4().hex), 'name': item.get('name') or 'Lap',
                           'time_ms': int(item.get('time_ms', 0)), 'car_id': item.get('car_id'), 'type': lap_type,
                           'fp': item.get('fp') or TRACKS.make_fingerprint(data, lap_type != 'OUT')}
                    lap['id'] = lap['id'].replace('.', '_')
                    self._write_file(self._path(lap['id']), self._meta_of(lap), data)
                except Exception:
                    ok = False
            if ok:
                try:
                    os.replace(legacy, legacy + ".migrated")      # keep a backup, never re-import
                except OSError:
                    pass

    # ---- background writer ----
    def _run(self):
        while True:
            op, payload = self._q.get()
            try:
                if op == "save":
                    path, meta, data = payload
                    self._write_file(path, meta, data)
                elif op == "delete":
                    try:
                        os.remove(payload)
                    except FileNotFoundError:
                        pass
            except Exception:
                pass
            finally:
                self._q.task_done()

    def save_lap(self, lap):
        """Queue one lap for writing. Returns immediately."""
        lap_id = str(lap['id']).replace('.', '_')
        self._q.put(("save", (self._path(lap_id), self._meta_of(lap), lap['data'])))

    def delete_lap(self, lap_id):
        self._q.put(("delete", self._path(str(lap_id).replace('.', '_'))))

    def flush(self, timeout=5.0):
        """Wait (bounded) for queued writes - call on shutdown."""
        t0 = time.time()
        while self._q.unfinished_tasks and time.time() - t0 < timeout:
            time.sleep(0.01)


# ==========================================================================
# Lap tracking
# ==========================================================================
class LapTracker:
    MIN_FRAMES = 240          # ~4 s of data; shorter "laps" are menu/reset noise
    MIN_DIST_M = 150.0
    MAX_STEP_M = 60.0         # position jump bigger than this = teleport/reset, not driving
    TICK_GAP_MAX = 120        # ignore gaps > 2 s when accumulating lap time
    SLOW_FACTOR = 1.05        # laps slower than best * this are tagged "IN"

    def __init__(self):
        self.best_lap_time = float('inf')
        self.best_t = None
        self.best_d = None
        self.best_len = 0.0
        self.last_tick = None
        self.last_pos = None
        self._clear_lap()

    @property
    def has_best(self):
        return self.best_d is not None

    @staticmethod
    def format_time(milliseconds):
        if milliseconds <= 0 or milliseconds == float('inf'):
            return "00:00.000"
        total_ms = int(milliseconds)
        m, rem = divmod(total_ms, 60000)
        s, ms = divmod(rem, 1000)
        return f"{m:02d}:{s:02d}.{ms:03d}"

    def _clear_lap(self):
        if not hasattr(self, 'buf'):
            self.buf = np.empty((16384, 8), dtype=np.float64)   # t, d, x, y, z, speed, throttle, brake
        self.n = 0
        self.distance = 0.0
        self.elapsed_ticks = 0

    def suspend(self):
        """Call while the game is paused/loading so time and distance don't jump on resume."""
        self.last_tick = None
        self.last_pos = None

    def reset_session(self):
        self.best_lap_time = float('inf')
        self.best_t = self.best_d = None
        self.best_len = 0.0
        self._clear_lap()
        self.suspend()

    def _finish_lap(self, official_last_ms):
        if self.n < self.MIN_FRAMES or self.distance < self.MIN_DIST_M:
            return None, 0, "HOT"
        data = build_lap_data(self.buf[:self.n])      # no per-lap list -> array conversion hitch
        lap_len = float(data['d'][-1])

        if official_last_ms > 0:
            lap_time, lap_type = int(official_last_ms), "HOT"
            last_t = float(data['t'][-1])
            if last_t > 0:                       # align our stopwatch with the game's official time
                ratio = official_last_ms / last_t
                if 0.95 < ratio < 1.05:
                    data['t'] = data['t'] * ratio
            plausible = self.best_len == 0 or lap_len >= 0.9 * self.best_len
            if plausible and (self.best_lap_time == float('inf') or lap_time <= self.best_lap_time):
                self.best_lap_time = lap_time
                self.best_t, self.best_d, self.best_len = data['t'].copy(), data['d'].copy(), lap_len
            elif lap_time > self.best_lap_time * self.SLOW_FACTOR:
                lap_type = "IN"
        else:
            lap_time, lap_type = int(data['t'][-1]), "OUT"
        return data, lap_time, lap_type

    def process(self, speed_ms, tick, is_new_lap, throttle, brake, x, y, z, official_last_ms):
        completed, completed_time, lap_type = None, 0, "HOT"
        if is_new_lap:
            completed, completed_time, lap_type = self._finish_lap(official_last_ms)
            self._clear_lap()
            self.suspend()                        # first frame of the new lap is t=0, d=0

        if self.last_tick is not None:
            dt = tick - self.last_tick
            if 0 < dt <= self.TICK_GAP_MAX:
                self.elapsed_ticks += dt
        if self.last_pos is not None:
            lx, ly, lz = self.last_pos
            step = ((x - lx) ** 2 + (y - ly) ** 2 + (z - lz) ** 2) ** 0.5
            if step <= self.MAX_STEP_M:           # distance from real positions, not integrated speed
                self.distance += step
        self.last_tick, self.last_pos = tick, (x, y, z)

        t_ms = self.elapsed_ticks * (1000.0 / 60.0)
        n = self.n
        if n == len(self.buf):
            self.buf = np.concatenate([self.buf, np.empty_like(self.buf)])
        self.buf[n] = (t_ms, self.distance, x, y, z, speed_ms * 3.6, throttle, brake)
        self.n = n + 1

        delta = 0.0
        if self.best_d is not None and self.distance > 5.0:
            delta = (t_ms - float(np.interp(self.distance, self.best_d, self.best_t))) / 1000.0
        return delta, self.format_time(t_ms), completed, completed_time, lap_type


# ==========================================================================
# Packet decoding
# ==========================================================================
GT7_KEY = b'Simulator Interface Packet GT7 v'
GT7_MAGIC = 0x47375330
PACKET_MIN_LEN = 0x128

_S_POS = struct.Struct('<3f')        # 0x04
_S_RPM = struct.Struct('<f')         # 0x3C
_S_SPEED = struct.Struct('<f')       # 0x4C
_S_TEMPS = struct.Struct('<4f')      # 0x60
_S_LAP = struct.Struct('<iHHii')     # 0x70: tick, lap, laps_in_race, best_ms, last_ms
_S_FLAGS = struct.Struct('<H')       # 0x8E
_S_PEDALS = struct.Struct('<BBB')    # 0x90: gear, throttle, brake
_S_CAR = struct.Struct('<i')         # 0x124


def decrypt_packet(raw):
    if len(raw) < PACKET_MIN_LEN:
        return None
    iv1 = struct.unpack_from('<I', raw, 0x40)[0]
    nonce = struct.pack('<II', iv1 ^ 0xDEADBEAF, iv1)
    dec = Salsa20.new(key=GT7_KEY, nonce=nonce).decrypt(raw)
    if struct.unpack_from('<I', dec, 0)[0] != GT7_MAGIC:      # not a GT7 packet / wrong key
        return None
    return dec


def parse_packet(d):
    x, y, z = _S_POS.unpack_from(d, 0x04)
    tick, lap, _, best_ms, last_ms = _S_LAP.unpack_from(d, 0x70)
    gear, throttle, brake = _S_PEDALS.unpack_from(d, 0x90)
    return {
        'x': x, 'y': y, 'z': z,
        'rpm': _S_RPM.unpack_from(d, 0x3C)[0],
        'speed_ms': _S_SPEED.unpack_from(d, 0x4C)[0],
        'temps': _S_TEMPS.unpack_from(d, 0x60),
        'tick': tick, 'lap': lap, 'best_ms': best_ms, 'last_ms': last_ms,
        'flags': _S_FLAGS.unpack_from(d, 0x8E)[0],
        'gear': gear & 0x0F,
        'throttle': int(throttle / 255 * 100),
        'brake': int(brake / 255 * 100),
        'car_id': _S_CAR.unpack_from(d, 0x124)[0],
    }


# ==========================================================================
# Network thread
# ==========================================================================
class LiveGT7Thread(QThread):
    lap_completed = pyqtSignal(int, int, object, str, int)    # lap#, time_ms, lap data dict, type, car_id
    error_occurred = pyqtSignal(str)
    data_ready = pyqtSignal()                                 # "new telemetry waiting" - coalesced, never queues up

    RECV_PORT = 33740
    HEARTBEAT_PORT = 33739
    HEARTBEAT_EVERY = 1.0        # s. PS5 stops streaming if the keep-alive is late; reference tools use ~2-3 s
    DISCONNECT_AFTER = 3.0       # s without packets before we call it a disconnect (rides out Wi-Fi hiccups)
    MAX_BATCH = 120              # max samples handed to the UI in one go (~2 s at 60 Hz)

    def __init__(self, playstation_ip=""):
        super().__init__()
        self.sock = None
        self.tracker = LapTracker()
        self.current_lap_num = -1
        self._running = True
        self._lock = threading.Lock()
        self._ip = playstation_ip.strip()
        self._reconnect = True
        self._last_error = {}
        self._last_delta, self._last_cur = 0.0, "00:00.000"

        # Mailbox shared with the UI thread. The UI is told "data ready" at most once until it has collected it,
        # so a slow UI frame can never build a backlog of events, and nothing is ever lost: every packet's
        # position / pedal values are kept in `_samples` and delivered in order with the next snapshot.
        self._snap_lock = threading.Lock()
        self._snapshot = None
        self._notified = False
        self._samples = []
        self._pending_new_lap = False
        self._pending_reset = False

        # Link-quality statistics (updated once per second)
        self._net = {'hz': 0, 'lost_pct': 0.0, 'gap_ms': 0}
        self._win_start = time.time()
        self._win_pkts = self._win_lost = 0
        self._win_maxgap = 0.0
        self._last_pkt_time = 0.0
        self._last_pkt_id = None
        self._history = []

    # ---- thread-safe controls (called from the UI thread) ----
    def set_ip(self, new_ip):
        with self._lock:
            self._ip = new_ip.strip()
            self._reconnect = True

    def stop(self):
        self._running = False

    def take_snapshot(self):
        """UI thread: newest telemetry dict (with every sample since the last call), or None."""
        with self._snap_lock:
            snap, self._snapshot = self._snapshot, None
            self._notified = False
            if snap is not None:
                snap['samples'] = self._samples
                self._samples = []
                self._pending_new_lap = self._pending_reset = False
            return snap

    def _publish(self, snap):
        with self._snap_lock:
            self._snapshot = snap
            notify = not self._notified
            self._notified = True
        if notify:
            self.data_ready.emit()

    # ---- internals ----
    def _emit_error(self, msg):
        now = time.time()
        if now - self._last_error.get(msg, 0) > 3.0:          # never flood the UI at 60 Hz
            self._last_error[msg] = now
            self.error_occurred.emit(msg)

    def _close_socket(self):
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    def _open_socket(self):
        self._close_socket()
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            if os.name != 'nt':
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:                                              # big kernel buffer: bursts queue there instead of dropping
                s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
            except OSError:
                pass
            try:
                s.bind(('0.0.0.0', self.RECV_PORT))
            except OSError:
                s.bind(('0.0.0.0', 0))                         # PS5 replies to the heartbeat's source port
                self._emit_error("Port 33740 busy - using a fallback port")
            if hasattr(socket, 'SIO_UDP_CONNRESET'):          # Windows: ignore ICMP "port unreachable"
                try:
                    s.ioctl(socket.SIO_UDP_CONNRESET, False)
                except (OSError, ValueError):
                    pass
            s.settimeout(0.02)
            self.sock = s
        except OSError as e:
            self._emit_error(f"Socket Error: {e}")

    def run(self):
        while self._running:
            try:
                self._loop()
            except Exception as e:
                self._emit_error(f"Thread error (restarting): {e}")
                time.sleep(1.0)
        self._close_socket()

    def _recv_all(self):
        """Block briefly for the first packet, then drain everything already waiting. Returns [(bytes, time)]."""
        out = []
        try:
            raw, _ = self.sock.recvfrom(2048)
            out.append((raw, time.time()))
        except (socket.timeout, ConnectionResetError):
            return out
        while len(out) < 256:
            try:
                ready = select.select([self.sock], [], [], 0)[0]
                if not ready:
                    break
                raw, _ = self.sock.recvfrom(2048)
                out.append((raw, time.time()))
            except (OSError, ValueError):
                break
        return out

    def _track_link(self, now, pkt_id):
        """Packet-rate / loss / jitter statistics, shown in the UI so a bad link is visible, not guessed."""
        if self._last_pkt_time:
            self._win_maxgap = max(self._win_maxgap, now - self._last_pkt_time)
        self._last_pkt_time = now
        if self._last_pkt_id is not None:
            missing = pkt_id - self._last_pkt_id - 1
            if 0 < missing < 600:
                self._win_lost += missing
        self._last_pkt_id = pkt_id
        self._win_pkts += 1
        if now - self._win_start >= 1.0:
            span = now - self._win_start
            self._history.append((self._win_pkts, self._win_lost, self._win_maxgap, span))
            del self._history[:-10]                            # rolling 10 s, so a brief loss stays visible
            pkts = sum(h[0] for h in self._history)
            lost = sum(h[1] for h in self._history)
            secs = sum(h[3] for h in self._history)
            total = pkts + lost
            self._net = {'hz': int(round(pkts / secs)) if secs else 0,
                         'lost_pct': (100.0 * lost / total) if total else 0.0,
                         'gap_ms': int(max(h[2] for h in self._history) * 1000)}
            self._win_start, self._win_pkts, self._win_lost, self._win_maxgap = now, 0, 0, 0.0

    def _loop(self):
        last_heartbeat = last_packet = 0.0
        connected = False

        while self._running:
            with self._lock:
                ip, reconnect = self._ip, self._reconnect
                self._reconnect = False
            if reconnect or self.sock is None:
                self._open_socket()
                connected, last_heartbeat = False, 0.0
                self._last_pkt_id, self._last_pkt_time = None, 0.0
                if self.sock is None:
                    time.sleep(1.0)
                    continue

            now = time.time()
            if ip and now - last_heartbeat >= self.HEARTBEAT_EVERY:
                try:
                    self.sock.sendto(b'A', (ip, self.HEARTBEAT_PORT))
                except OSError:
                    pass
                last_heartbeat = now

            try:
                packets = self._recv_all()
            except OSError as e:
                self._emit_error(f"Receive error: {e}")
                self._close_socket()
                time.sleep(0.5)
                continue

            last_p = None
            for raw, t_arrival in packets:
                try:
                    dec = decrypt_packet(raw)
                    if dec is None:
                        continue
                    connected, last_packet = True, t_arrival
                    p = parse_packet(dec)
                    self._track_link(t_arrival, p['tick'])
                    is_new_lap, session_reset = self._process(p)
                    with self._snap_lock:                      # sticky until the UI has consumed them
                        self._pending_new_lap |= is_new_lap
                        self._pending_reset |= session_reset
                        if session_reset:
                            self._samples = []                 # samples from the old session are meaningless
                        if not (p['flags'] & 0b110):
                            self._samples.append((p['x'], p['y'], p['z'], p['throttle'], p['brake'], is_new_lap))
                            if len(self._samples) > self.MAX_BATCH:
                                del self._samples[:len(self._samples) - self.MAX_BATCH]
                    last_p = p
                except Exception as e:
                    self._emit_error(f"Parsing Error: {e}")

            if last_p is not None:                             # ONE snapshot per burst, built from the newest packet
                with self._snap_lock:
                    nl, rs = self._pending_new_lap, self._pending_reset
                self._publish(self._telemetry(last_p, nl, rs))

            if connected and time.time() - last_packet > self.DISCONNECT_AFTER:
                connected = False
                self._last_pkt_id = None
                self._publish({'connected': False})

    def _process(self, p):
        """Advance lap logic for one packet. Returns (is_new_lap, session_reset)."""
        if p['flags'] & 0b110:                   # bit1 = paused, bit2 = loading/processing
            self.tracker.suspend()
            return False, False

        lap = p['lap']
        is_new_lap = session_reset = False
        if self.current_lap_num == -1 or lap < self.current_lap_num:
            self.current_lap_num = lap
            self.tracker.reset_session()
            session_reset = True
        elif lap != self.current_lap_num:
            self.current_lap_num = lap
            is_new_lap = True

        delta, cur_str, comp, comp_time, l_type = self.tracker.process(
            p['speed_ms'], p['tick'], is_new_lap, p['throttle'], p['brake'],
            p['x'], p['y'], p['z'], p['last_ms'])
        self._last_delta, self._last_cur = delta, cur_str

        if comp is not None:
            self.lap_completed.emit(lap - 1, comp_time, comp, l_type, p['car_id'])
        return is_new_lap, session_reset

    def _telemetry(self, p, new_lap, session_reset):
        if p['best_ms'] > 0:
            best_str = LapTracker.format_time(p['best_ms'])
        elif self.tracker.best_lap_time != float('inf'):
            best_str = LapTracker.format_time(self.tracker.best_lap_time)
        else:
            best_str = "00:00.000"
        return {
            'connected': True,
            'car_id': p['car_id'],
            'speed': int(p['speed_ms'] * 3.6),
            'rpm': int(p['rpm']),
            'gear': p['gear'] if p['gear'] > 0 else 'R',
            'throttle': p['throttle'], 'brake': p['brake'],
            'temps': [int(t) for t in p['temps']],
            'delta': self._last_delta,
            'has_best': self.tracker.has_best,
            'cur_lap': self._last_cur, 'best_lap': best_str,
            'pos_x': p['x'], 'pos_y': p['y'], 'pos_z': p['z'],
            'is_new_lap': new_lap, 'session_reset': session_reset,
            'paused': bool(p['flags'] & 0b110),
            'net': dict(self._net),
        }


class CatalogUpdater(QThread):
    """Refreshes the car/track catalogue in the background (at most once a week)."""
    finished_ok = pyqtSignal(bool)

    def run(self):
        try:
            self.finished_ok.emit(bool(refresh_catalog()))
        except Exception:
            self.finished_ok.emit(False)
