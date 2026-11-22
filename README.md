# Pasta Tracker (GT7 Telemetry Suite)

A modular real-time telemetry dashboard and lap analysis suite for **Gran Turismo 7**, styled with a black, white, and gold **SAGT Esports** theme.

## Features

- **Live Race Dashboard (60Hz):**
  - Real-time speed, gear indicator, and RPM bar.
  - Live delta against your locked **Session Best** lap.
  - Current lap and Session Best lap timers synced to GT7's internal physics tick counter.
  - Live scrolling throttle and brake input traces.
  - Tire temperature visualization (FL, FR, RL, RR).
  - Interactive 3D elevation track map (click and drag to rotate).
- **Telemetry Analyzer & Vaults:**
  - Automatically logs completed laps and groups them by circuit and vehicle.
  - Compare up to two laps side-by-side on speed and pedal input graphs.
  - Cached 3D track playback with adjustable playback speeds (`0.5x`, `1.0x`, `1.5x`, `2.0x`).
  - Keep, rename, and delete laps with persistent JSON storage (`gt7_saved_laps.json`).
- **Custom Car & Track Mapping:**
  - Extend car IDs via `gt7_cars.json` and track coordinate zones via `gt7_tracks.json`.

## Setup & Usage

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Gran Turismo 7:**
   - Ensure your PC and PlayStation (PS4/PS5) are on the same local network.
   - Note your PlayStation's local IP address.
   - Launch **Pasta Tracker**, enter the PlayStation IP in the top bar, and click **CONNECT**.

3. **Run from Source:**
   ```bash
   python main.py
   ```

## Building a Standalone `.exe`

```bash
python -m PyInstaller --noconsole --onefile --name "Pasta Tracker" --add-data "pasta_logo_transparent.png;." --add-data "SAGT logo.jpeg;." main.py
```
