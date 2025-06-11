import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
import csv
import os
import threading
import time


selected_quadrant = [(90, 128), (597, 388)]
scaling_factor = 2

class ROIActivityMonitor:
    def __init__(self, csvfile, sound_folder):
        #read rois from csv file
        self.rois = self.read_rois(csvfile)
        #initialising variable that shows the indexes
        self.selected_roi_index = None

        self.sound_paths = self.load_sounds(sound_folder)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()
        self.sound_data = self.preload_sounds()
        self.setup_camera()
        self.active_sound_thread = None #tracks the active sound thread
        self.most_active_roi = None # tracks the most active roi
        self.lock = threading.Lock() #prevents two threads from reading/modifying the dictionary at the same time
        self.runnung = True

    def read_rois(self, csvfile):
        """ Adjust ROI coordinates based on cropping and scaling """
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
        sounds = {}
        for roi_index in self.rois.keys():
            sound_path = os.path.join(folder_path, f"{roi_index}.wav")
            if os.path.isfile(sound_path):
                sounds[roi_index] = sound_path
        return sounds

    def preload_sounds(self):
        sound_data = {}
        for roi_index, sound_path in self.sound_paths.items():
            data, _ = sf.read(sound_path)
            sound_data[roi_index] = data
        return sound_data

    def setup_camera(self):
        vid = cv2.VideoCapture(0)
        prev_frame = None
        while True:
            ret, frame = vid.read()
            if not ret:
                break
            cropped_frame = frame[selected_quadrant[0][1]:selected_quadrant[1][1], selected_quadrant[0][0]:selected_quadrant[1][0]]
            frame = cv2.resize(cropped_frame, (0, 0), fx=scaling_factor, fy=scaling_factor)
            self.process_frame(frame, prev_frame)
            prev_frame = frame
            cv2.imshow('Original ROI Activity Monitor', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        vid.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame, prev_frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg_mask = self.bg_subtractor.apply(gray_frame)
        _, thresh = cv2.threshold(fg_mask, 20, 255, cv2.THRESH_BINARY)
        thresh_colored = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        # Initialize variables to track the most active ROI
        max_activity = 0
        most_active_roi = None

        # Loop through each ROI to calculate its activity level
        for roi_index, ((x1, y1), (x2, y2)) in self.rois.items():

            #show rois
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(thresh_colored, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Extract the part of the threshold image that corresponds to the current ROI
            roi_diff = thresh[y1:y2, x1:x2]
            roi_diff = thresh[y1:y2, x1:x2]

            # Calculate the activity level as the sum of values in the ROI's diff area
            activity_level = np.sum(roi_diff)
            if activity_level > max_activity:
                max_activity = activity_level
                most_active_roi = roi_index

        # Check if there is a significantly active ROI
        if most_active_roi and max_activity > 900000:  ##adhust the threshold if too low
            #print to check
            print(f"Highest Activity in ROI {most_active_roi}: {max_activity} - Triggering sound!")

            #play sound of only the most active roi
            threading.Thread(target=self.play_sound, args=(most_active_roi,)).start()

        cv2.imshow('Processed ROI Activity (BG Subtraction & Thresholding)', thresh_colored)

    def play_sound(self, roi_index):
        sound_path = self.sound_paths.get(roi_index)
        if sound_path:
            data = self.sound_data.get(roi_index)
            if data is not None:
                
                sd.play(data, blocking=False)
                time.sleep(1)

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    csv_file = 'auditory_rois.csv'
    sound_folder = filedialog.askdirectory(title="Select Folder Containing Sound Files")
    if sound_folder:
        app = ROIActivityMonitor(csv_file, sound_folder)
        root.mainloop()
    else:
        root.destroy()
