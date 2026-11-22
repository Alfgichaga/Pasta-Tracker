"""
Car + track resolution for the GT7 Telemetry Suite.

WHY NAMES WERE SHOWING AS "UNKNOWN":
  * Cars: the built-in sample IDs were wrong (e.g. 3362 is the Ferrari F50 '95, not a
    Porsche 919) and gt7_cars.json was read from the *current working directory*, so
    launching from another folder silently loaded nothing. This module ships a full,
    verified catalogue (gt7_cars.json, ~580 cars), always loads it relative to THIS file,
    and refreshes it weekly from the community database in the background.
  * Tracks: GT7 telemetry has no track ID. The old code guessed the track from
    int(x // 1000) / int(z // 1000) of the start position. In metres that yields tiny
    numbers like "0_0" or "-1_0", so keys such as "131_-22" could never match anything.
    Tracks are now identified by their finish-line fingerprint (position + heading + lap
    length). You name a track ONCE (SET TRACK) and every lap on it - past and future -
    is recognised automatically.
"""
import os
import io
import csv
import json
import math
import time
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def data_path(name):
    """Absolute path of a data file that lives next to the scripts (never the CWD)."""
    return os.path.join(BASE_DIR, name)


CARS_FILE = data_path("gt7_cars.json")                 # bundled / auto-refreshed catalogue
CARS_CUSTOM_FILE = data_path("gt7_cars_custom.json")   # your own additions, never overwritten
COURSES_FILE = data_path("gt7_courses.json")           # bundled / auto-refreshed layouts
TRACKS_FILE = data_path("gt7_tracks.json")             # finish-line fingerprints you named
META_FILE = data_path("gt7_catalog_meta.json")

CATALOG_URL = "https://raw.githubusercontent.com/ddm999/gt7info/web-new/_data/db/{}.csv"
REFRESH_INTERVAL_S = 7 * 24 * 3600


# ----------------------------------------------------------------------------
# Small safe-IO helpers (atomic writes so a crash can never corrupt a data file)
# ----------------------------------------------------------------------------
def read_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def atomic_write_json(path, obj):
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
        os.replace(tmp, path)
        return True
    except (OSError, TypeError, ValueError):
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False


# ----------------------------------------------------------------------------
# Cars
# ----------------------------------------------------------------------------
class CarDatabase:
    def __init__(self):
        self.cars = {}
        self.reload()

    def reload(self):
        cars = {}
        for path in (CARS_FILE, CARS_CUSTOM_FILE):      # custom file wins
            raw = read_json(path, {})
            if isinstance(raw, dict):
                for k, v in raw.items():
                    try:
                        cars[int(k)] = str(v)
                    except (TypeError, ValueError):
                        continue
        self.cars = cars

    def name(self, car_id):
        try:
            car_id = int(car_id)
        except (TypeError, ValueError):
            return "Unknown Car"
        return self.cars.get(car_id, f"Unknown Car (#{car_id})")

    def is_known(self, car_id):
        try:
            return int(car_id) in self.cars
        except (TypeError, ValueError):
            return False


# ----------------------------------------------------------------------------
# Tracks
# ----------------------------------------------------------------------------
def _angle_diff(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


class TrackRegistry:
    MATCH_RADIUS_M = 40.0                 # finish-line position tolerance
    HEADING_TOL = math.radians(70)        # separates forward / reverse layouts
    LEARNED_LEN_TOL = 0.08                # lap-length tolerance vs the learned lap
    LEN_SUGGEST_TOL = 0.03                # "likely match" window for the picker

    def __init__(self):
        self.courses = []
        self.learned = []
        self.reload_courses()
        self.reload_learned()

    # -- loading -------------------------------------------------------------
    def reload_courses(self):
        raw = read_json(COURSES_FILE, [])
        self.courses = [c for c in raw if isinstance(c, dict) and c.get("name")] if isinstance(raw, list) else []

    def reload_learned(self):
        raw = read_json(TRACKS_FILE, [])
        # The old dict format ("131_-22": name) could never match anything, so it is ignored.
        self.learned = [e for e in raw if isinstance(e, dict) and {"name", "x", "z", "h"} <= e.keys()] \
            if isinstance(raw, list) else []

    def _save_learned(self):
        return atomic_write_json(TRACKS_FILE, self.learned)

    # -- fingerprint -----------------------------------------------------------
    @staticmethod
    def make_fingerprint(frames, full_lap):
        """Finish-line fingerprint of a completed lap (frames = dict of numpy arrays)."""
        try:
            x, z, d = frames["x"], frames["z"], frames["d"]
            n = len(x)
            if n < 20:
                return None
            k = min(30, n - 1)
            heading = math.atan2(float(z[-1] - z[-1 - k]), float(x[-1] - x[-1 - k]))
            return {"x": round(float(x[-1]), 1), "z": round(float(z[-1]), 1),
                    "h": round(heading, 3), "len": round(float(d[-1]), 1), "full": bool(full_lap)}
        except (KeyError, TypeError, ValueError, IndexError):
            return None

    # -- matching ------------------------------------------------------------------
    def _matches(self, fp, e):
        dist = math.hypot(fp["x"] - e["x"], fp["z"] - e["z"])
        if dist > self.MATCH_RADIUS_M:
            return None
        if _angle_diff(fp["h"], e["h"]) > self.HEADING_TOL:
            return None
        if fp.get("full") and e.get("len"):
            if abs(fp["len"] - e["len"]) / e["len"] > self.LEARNED_LEN_TOL:
                return None
        return dist

    def identify(self, fp):
        """Learned track name for this fingerprint, or None."""
        if not fp:
            return None
        best, best_d = None, float("inf")
        for e in self.learned:
            d = self._matches(fp, e)
            if d is not None and d < best_d:
                best, best_d = e["name"], d
        return best

    def display_name(self, fp):
        name = self.identify(fp)
        if name:
            return name
        if fp and fp.get("full") and fp.get("len"):
            return f"Unidentified Track (~{fp['len'] / 1000:.1f} km) - use SET TRACK"
        return "Unidentified Track - use SET TRACK"

    def suggestions(self, fp):
        """[(label, name)] - catalogue layouts, closest lap length first."""
        out = []
        measured = fp["len"] if (fp and fp.get("full") and fp.get("len")) else None
        for c in self.courses:
            length = c.get("length") or 0
            if measured and length:
                diff = (measured - length) / length * 100.0
                star = "* " if abs(diff) <= self.LEN_SUGGEST_TOL * 100 else ""
                out.append((abs(diff), f"{star}{c['name']} - {length} m ({diff:+.1f}%)", c["name"]))
            else:
                out.append((0, f"{c['name']} - {length} m", c["name"]))
        out.sort(key=lambda t: (t[0], t[2]) if measured else (0, t[2]))
        return [(label, name) for _, label, name in out]

    # -- learning ---------------------------------------------------------------------
    def learn(self, fp, name):
        if not fp or not name:
            return False
        self.learned = [e for e in self.learned if self._matches(fp, e) is None]
        self.learned.append({"name": name, "x": fp["x"], "z": fp["z"], "h": fp["h"],
                             "len": fp["len"] if fp.get("full") else None})
        return self._save_learned()


# ----------------------------------------------------------------------------
# Catalogue refresh (runs in a background thread, silent on any failure)
# ----------------------------------------------------------------------------
def _fetch_csv(name, timeout=10):
    req = urllib.request.Request(CATALOG_URL.format(name), headers={"User-Agent": "GT7TelemetrySuite/2.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        text = r.read().decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def build_car_map(car_rows, maker_rows):
    makers = {int(r["ID"]): r["Name"].strip() for r in maker_rows}
    cars = {}
    for r in car_rows:
        short = r["ShortName"].strip()
        maker = makers.get(int(r["Maker"]), "")
        name = short if (not maker or short.lower().startswith(maker.lower())) else f"{maker} {short}"
        cars[str(int(r["ID"]))] = name
    return cars


def build_course_list(rows):
    return [{"id": int(r["ID"]), "name": r["Name"].strip(), "length": int(float(r["Length"])),
             "reverse": r["IsReverse"] == "1", "base": int(r["Base"])} for r in rows]


def refresh_catalog(force=False):
    """Download the latest car/course lists. Returns True only if files were updated."""
    meta = read_json(META_FILE, {})
    if not force and time.time() - meta.get("last_check", 0) < REFRESH_INTERVAL_S:
        return False
    try:
        cars = build_car_map(_fetch_csv("cars"), _fetch_csv("maker"))
        courses = build_course_list(_fetch_csv("course"))
    except Exception:
        return False                       # offline / blocked / format changed: keep what we have
    if len(cars) < 300 or len(courses) < 50:
        return False                       # sanity check: never overwrite good data with a bad download
    ok = atomic_write_json(CARS_FILE, cars) and atomic_write_json(COURSES_FILE, courses)
    if ok:
        atomic_write_json(META_FILE, {"last_check": time.time()})
    return ok


CAR_DB = CarDatabase()
TRACKS = TrackRegistry()
