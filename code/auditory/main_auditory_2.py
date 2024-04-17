import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
import csv
import os
import threading

class ROIActivityMonitor:
    def __init__(self, csvfile, sound_folder):
        self.rois = self.read_rois(csvfile)
        self.sound_paths = self.load_sounds(sound_folder)
        self.frame_queue = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.setup_camera)
        self.thread.start()

    def read_rois(self, csvfile):
        """Read ROI definitions from a CSV file."""
        rois = {}
        with open(csvfile, newline='') as file:
            reader = csv.reader(file)
            for idx, row in enumerate(reader, start=1):
                if len(row) != 4:
                    continue
                x1, y1, x2, y2 = map(int, row)
                rois[str(idx)] = [(x1, y1), (x2, y2)]
        return rois

    def load_sounds(self, folder_path):
        """Load sound files based on ROI indices."""
        sounds = {}
        for roi_index in self.rois.keys():
            sound_path = os.path.join(folder_path, f"{roi_index}.wav")
            if os.path.isfile(sound_path):
                sounds[roi_index] = sound_path
        return sounds

    def setup_camera(self):
        """Set up camera and display video with ROIs asynchronously."""
        vid = cv2.VideoCapture(0)
        prev_frame = None
        while not self.stop_event.is_set():
            ret, frame = vid.read()
            if not ret:
                break
            if prev_frame is not None:
                self.process_frame(frame, prev_frame)
            prev_frame = frame
            cv2.imshow('ROI Activity Monitor', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        vid.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame, prev_frame):
        """Process each frame to detect activity and draw ROIs."""
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_frame, cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY))
        for roi_index, ((x1, y1), (x2, y2)) in self.rois.items():
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            roi_diff = diff[y1:y2, x1:x2]
            if np.sum(roi_diff) > 5000:  # Adjust this threshold as needed
                threading.Thread(target=self.play_sound, args=(roi_index,)).start()

    def play_sound(self, roi_index):
        """Play sound for active ROI asynchronously."""
        sound_path = self.sound_paths.get(roi_index)
        if sound_path:
            try:
                data, fs = sf.read(sound_path)
                sd.play(data, fs)
                sd.wait()
            except Exception as e:
                print(f"Error playing sound: {e}")

if __name__ == "__main__":
    csv_file = 'auditory_rois.csv'
    sound_folder = filedialog.askdirectory(title="Select Folder Containing Sound Files")
    if sound_folder:
        app = ROIActivityMonitor(csv_file, sound_folder)
    else:
        print("No sound folder selected; exiting application.")
