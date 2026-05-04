import tkinter as tk
from tkinter import messagebox
import threading
import pygame
import time
import csv
import sys
import subprocess
from pathlib import Path

from hit_detector import HitDetector, AVAILABLE as HIT_DETECTION_AVAILABLE

DATA_DIR = Path(__file__).parent / "data"

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

    def start_recording(self):
        self._stop_event.clear()
        self._last_hit_time = 0.0
        self._hit_count = 0

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.filename = DATA_DIR / f"session_{timestamp}.csv"
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

    def analyze(self):
        analyzer = Path(__file__).parent / "analyzer.py"
        try:
            subprocess.run([sys.executable, str(analyzer)], check=True)
            messagebox.showinfo("Analysis", "Analysis complete. Check data folder for plots.")
        except subprocess.CalledProcessError:
            messagebox.showerror("Error", "Analysis failed.")


if __name__ == "__main__":
    app = App()
    app.mainloop()
    pygame.quit()
