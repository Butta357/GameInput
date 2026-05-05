import tkinter as tk
from tkinter import messagebox, ttk
import threading
import pygame
import time
import csv
import json
import re
import sys
import subprocess
from pathlib import Path

from hit_detector import HitDetector, AVAILABLE as HIT_DETECTION_AVAILABLE

DATA_DIR = Path(__file__).parent / "data"

# Marvel Rivals roster (Season 7.5, May 2026). Combobox stays editable so
# heroes added in future patches can still be typed in.
MARVEL_RIVALS_HEROES = [
    "Adam Warlock", "Angela", "Black Cat", "Black Panther", "Black Widow",
    "Blade", "Captain America", "Cloak and Dagger", "Daredevil", "Deadpool",
    "Doctor Strange", "Elsa Bloodstone", "Emma Frost", "Gambit", "Groot",
    "Hawkeye", "Hela", "Hulk", "Human Torch", "Invisible Woman",
    "Iron Fist", "Iron Man", "Jeff the Land Shark", "Loki", "Luna Snow",
    "Magik", "Magneto", "Mantis", "Mister Fantastic", "Moon Knight",
    "Namor", "Peni Parker", "Phoenix", "Psylocke", "Rocket Raccoon",
    "Rogue", "Scarlet Witch", "Spider-Man", "Squirrel Girl", "Star-Lord",
    "Storm", "The Punisher", "The Thing", "Thor", "Ultron",
    "Venom", "White Fox", "Winter Soldier", "Wolverine",
]

# The hit marker appears on screen slightly after the shot registers on the server.
# We mark the 50 ms window after detection as hit=1 in the CSV.
# The analyser looks back 50–300 ms from each hit event to recover stick state
# at actual shot time.
_HIT_MARK_WINDOW = 0.05  # seconds


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Xbox Controller Monitor")
        self._stop_event = threading.Event()
        self.filename = None
        self._last_hit_time = 0.0
        self._hit_count = 0

        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            messagebox.showerror("Error", "No joystick found")
            self.destroy()
            return
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self.num_axes    = self.joystick.get_numaxes()
        self.num_buttons = self.joystick.get_numbuttons()
        self.num_hats    = self.joystick.get_numhats()

        screen_cx = self.winfo_screenwidth()  // 2
        screen_cy = self.winfo_screenheight() // 2
        self.hit_detector = HitDetector(screen_cx, screen_cy)

        char_frame = tk.Frame(self)
        tk.Label(char_frame, text="Character:").pack(side=tk.LEFT, padx=4)
        self.character_var = tk.StringVar()
        self.character_combo = ttk.Combobox(
            char_frame,
            textvariable=self.character_var,
            values=MARVEL_RIVALS_HEROES,
            width=22,
        )
        self.character_combo.pack(side=tk.LEFT)
        char_frame.pack(pady=10)

        self.start_btn = tk.Button(self, text="Start Recording", command=self.start_recording)
        self.start_btn.pack(pady=10)
        self.stop_btn = tk.Button(self, text="Stop Recording", command=self.stop_recording, state=tk.DISABLED)
        self.stop_btn.pack(pady=10)
        self.analyze_btn = tk.Button(
            self, text="Analyze Latest", command=self.analyze,
            state=tk.NORMAL if self._has_sessions() else tk.DISABLED,
        )
        self.analyze_btn.pack(pady=10)
        self.status_label = tk.Label(self, text=f"Controller: {self.joystick.get_name()}")
        self.status_label.pack(pady=10)

        hit_status = "ready" if HIT_DETECTION_AVAILABLE else "unavailable — install mss and opencv-python"
        self.hit_label = tk.Label(self, text=f"Hit detection: {hit_status}", fg="gray")
        self.hit_label.pack(pady=4)

    def _has_sessions(self):
        return DATA_DIR.is_dir() and any(DATA_DIR.glob("session_*.csv"))

    @staticmethod
    def _sanitize_character(name):
        cleaned = re.sub(r'[^A-Za-z0-9_]+', '_', name.strip()).strip('_').lower()
        return cleaned or None

    def start_recording(self):
        character = self._sanitize_character(self.character_var.get())
        if not character:
            messagebox.showerror("Error", "Please enter a character name before recording.")
            return

        self._stop_event.clear()
        self._last_hit_time = 0.0
        self._hit_count = 0

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.filename = DATA_DIR / f"session_{character}_{timestamp}.csv"
        DATA_DIR.mkdir(exist_ok=True)

        with open(self.filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ['timestamp']
            for i in range(self.num_axes):
                header.append(f'axis_{i}')
            for i in range(self.num_buttons):
                header.append(f'button_{i}')
            for i in range(self.num_hats):
                header.extend([f'hat_{i}_x', f'hat_{i}_y'])
            header.append('hit')
            writer.writerow(header)

        self.status_label.config(text="Recording...")
        self.hit_label.config(
            text="Hits detected: 0" if HIT_DETECTION_AVAILABLE else "Hit detection: unavailable"
        )
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.analyze_btn.config(state=tk.DISABLED)
        self.character_combo.config(state="disabled")

        self.hit_detector.start()
        threading.Thread(target=self.record_loop, daemon=True).start()

    def record_loop(self):
        with open(self.filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            while not self._stop_event.is_set():
                pygame.event.pump()
                now = time.time()

                new_hits = self.hit_detector.get_hits()
                if new_hits:
                    self._last_hit_time = new_hits[-1]
                    self._hit_count += len(new_hits)
                    count = self._hit_count
                    self.after(0, lambda c=count: self.hit_label.config(text=f"Hits detected: {c}"))

                hit = 1 if 0.0 <= now - self._last_hit_time <= _HIT_MARK_WINDOW else 0

                row = [now]
                for i in range(self.num_axes):
                    row.append(self.joystick.get_axis(i))
                for i in range(self.num_buttons):
                    row.append(self.joystick.get_button(i))
                for i in range(self.num_hats):
                    hat = self.joystick.get_hat(i)
                    row.extend([hat[0], hat[1]])
                row.append(hit)
                writer.writerow(row)
                time.sleep(0.01)

    def stop_recording(self):
        self._stop_event.set()
        self.hit_detector.stop()
        self.status_label.config(text="Stopped")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.analyze_btn.config(state=tk.NORMAL)
        self.character_combo.config(state="normal")

    def analyze(self):
        analyzer = Path(__file__).parent / "analyzer.py"
        try:
            subprocess.run([sys.executable, str(analyzer)], check=True)
        except subprocess.CalledProcessError:
            messagebox.showerror("Error", "Analysis failed.")
            return

        lines = ["Analysis complete.\n"]
        for settings_file in sorted(DATA_DIR.glob("*/settings.json")):
            try:
                with open(settings_file) as fp:
                    s = json.load(fp)
                r = s.get("recommended")
                if r:
                    lines.append(f"{s['character']}  (from {', '.join(r['source_axes'])})")
                    diff = s.get("diff")
                    if diff:
                        for key, label in (("custom_minimum_range",           "Custom Minimum Range          "),
                                           ("custom_maximum_range",           "Custom Maximum Range          "),
                                           ("minimum_curve_statics",          "Minimum Curve Statics         "),
                                           ("custom_maximum_dual_zone_curve", "Custom Maximum Dual-zone Curve")):
                            d = diff[key]
                            arrow = "->" if d["delta"] != 0 else "=="
                            lines.append(f"  {label}: {d['current']:>3} {arrow} {d['recommended']:<3}  ({d['delta']:+d})")
                        if diff.get("low_confidence"):
                            lines.append("  !! LOW CONFIDENCE: aim stick barely moved, recommendation unreliable")
                        elif diff.get("material_changes"):
                            lines.append(f"  Material changes (>20%): {', '.join(diff['material_changes'])}")
                    else:
                        lines.append(f"  Custom Minimum Range          : {r['custom_minimum_range']}")
                        lines.append(f"  Custom Maximum Range          : {r['custom_maximum_range']}")
                        lines.append(f"  Minimum Curve Statics         : {r['minimum_curve_statics']}")
                        lines.append(f"  Custom Maximum Dual-zone Curve: {r['custom_maximum_dual_zone_curve']}")
                    cam = s.get("camera_motion")
                    if cam:
                        lines.append(f"  Camera motion p50={cam['p50']} p95={cam['p95']} saturated={cam['saturated_pct']}%")
                    hint = r.get("sensitivity_hint")
                    if hint:
                        lines.append(f"  Sensitivity hint: {hint['suggestion']}")
                    lines.append("")
            except (OSError, json.JSONDecodeError):
                pass
        lines.append(f"Plots and settings.json saved under {DATA_DIR}\\<character>\\")
        messagebox.showinfo("Analysis", "\n".join(lines))


if __name__ == "__main__":
    app = App()
    app.mainloop()
    pygame.quit()
