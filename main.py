import sys
import tkinter as tk
import tkinter.messagebox
import threading
import pygame
import time
import csv
import os
import subprocess

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Xbox Controller Monitor")
        self.recording = False
        self.filename = None

        # Check controller
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            tkinter.messagebox.showerror("Error", "No joystick found")
            self.quit()
            return
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self.num_axes = self.joystick.get_numaxes()
        self.num_buttons = self.joystick.get_numbuttons()
        self.num_hats = self.joystick.get_numhats()

        # GUI
        self.start_btn = tk.Button(self, text="Start Recording", command=self.start_recording)
        self.start_btn.pack(pady=10)
        self.stop_btn = tk.Button(self, text="Stop Recording", command=self.stop_recording, state=tk.DISABLED)
        self.stop_btn.pack(pady=10)
        self.analyze_btn = tk.Button(self, text="Analyze Latest", command=self.analyze)
        self.analyze_btn.pack(pady=10)
        self.status_label = tk.Label(self, text=f"Controller: {self.joystick.get_name()}")
        self.status_label.pack(pady=10)

    def start_recording(self):
        if not self.recording:
            self.recording = True
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.filename = f"data/session_{timestamp}.csv"
            os.makedirs("data", exist_ok=True)
            with open(self.filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                header = ['timestamp']
                for i in range(self.num_axes):
                    header.append(f'axis_{i}')
                for i in range(self.num_buttons):
                    header.append(f'button_{i}')
                for i in range(self.num_hats):
                    header.extend([f'hat_{i}_x', f'hat_{i}_y'])
                writer.writerow(header)
            self.status_label.config(text="Recording...")
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            threading.Thread(target=self.record_loop, daemon=True).start()

    def record_loop(self):
        while self.recording:
            pygame.event.pump()
            row = [time.time()]
            for i in range(self.num_axes):
                row.append(self.joystick.get_axis(i))
            for i in range(self.num_buttons):
                row.append(self.joystick.get_button(i))
            for i in range(self.num_hats):
                hat = self.joystick.get_hat(i)
                row.extend([hat[0], hat[1]])
            with open(self.filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row)
            time.sleep(0.01)

    def stop_recording(self):
        self.recording = False
        self.status_label.config(text="Stopped")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def analyze(self):
        try:
            subprocess.run([sys.executable, "analyzer.py"], check=True)
            tkinter.messagebox.showinfo("Analysis", "Analysis complete. Check data folder for plots.")
        except subprocess.CalledProcessError:
            tkinter.messagebox.showerror("Error", "Analysis failed.")

if __name__ == "__main__":
    app = App()
    app.mainloop()
    pygame.quit()