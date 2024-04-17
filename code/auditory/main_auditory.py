import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import os
import sounddevice as sd
import soundfile as sf
import csv
import threading
import time

class ROIActivitySoundApp:
    def __init__(self, root, csvfile, sound_folder):
        self.root = root
        self.root.title("ROI Activity Sound App")
        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack()
        
        self.rois = self.read_rois(csvfile)
        self.sound_paths = self.load_sounds(sound_folder)
        self.setup_camera()
        
    def read_rois(self, csvfile):
        """Read ROI definitions from a CSV file."""
        rois = {}
        with open(csvfile, newline='') as file:
            reader = csv.reader(file)
            for idx, row in enumerate(reader, start=1):
                if len(row) != 4:
                    print(f"Skipping malformed row: {row}")  # Skip any incorrectly formatted lines
                    continue
                x1, y1, x2, y2 = map(int, row)  # Convert string to integers
                rois[str(idx)] = [(x1, y1), (x2, y2)]  # Use the row number as the index
        return rois

    def load_sounds(self, folder_path):
        """Load sound files based on ROI indices."""
        sounds = {}
        for roi_index in self.rois.keys():
            sound_path = os.path.join(folder_path, f"{roi_index}.wav")
            if os.path.isfile(sound_path):
                sounds[roi_index] = sound_path
            else:
                messagebox.showwarning("Warning", f"Sound file for ROI {roi_index} not found.")
        return sounds

    def setup_camera(self):
        """Initialize camera and start video capture."""
        self.vid = cv2.VideoCapture(0)
        self.prev_frame = None
        self.update_video_frame()

    def update_video_frame(self):
        """Capture video frames and detect activity in ROIs."""
        ret, frame = self.vid.read()
        if ret:
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.root.image = imgtk  # Store a reference to prevent garbage collection
            self.root.after(10, self.update_video_frame)  # Reschedule itself
        else:
            self.root.after(10, self.update_video_frame)

    def detect_activity(self, current_frame):
        """Detect significant activity in ROIs based on frame differencing. ---THRESHOLD IS SET HERE"""
        if self.prev_frame is not None:
            diff = cv2.absdiff(current_frame, self.prev_frame)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            for roi_index, ((x1, y1), (x2, y2)) in self.rois.items():
                roi_thresh = thresh[y1:y2, x1:x2]
                if cv2.countNonZero(roi_thresh) > 1000:  # Threshold for "significant activity"
                    self.play_sound(roi_index)

    def play_sound(self, roi_index):
        """Play sound corresponding to the ROI index if significant activity is detected."""
        sound_path = self.sound_paths.get(roi_index)
        if sound_path:
            data, fs = sf.read(sound_path)
            sd.play(data, fs)
            sd.wait()

    # def start_video_stream(self):
    #     thread = threading.Thread(target=self.video_loop)
    #     thread.start()

    # def video_loop(self):
    #     while self.running:
    #         self.update_video_frame()
    #         time.sleep(0.1)  # Control the frame rate

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # Optionally hide the main window
    sound_folder = filedialog.askdirectory(title="Select Folder Containing Sound Files")
    if sound_folder:
        root.deiconify()  # Show the main window
        app = ROIActivitySoundApp(root, "auditory_rois.csv", sound_folder)
        root.mainloop()
    else:
        root.destroy()
