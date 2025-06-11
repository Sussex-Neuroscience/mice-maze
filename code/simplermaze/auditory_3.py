import cv2
import pandas as pd
#from playsound import playsound
import pygame
import threading
import tkinter as tk
from tkinter import filedialog

#selected quadrant from crop_zoom.py
selected_quadrant = [(90, 128), (597, 388)]
scaling_factor = 2

# Initialize Pygame Mixer
pygame.mixer.init()

# Function to play sound
def play_sound(sound_path):
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()
    pygame.mixer.music.load(sound_path)
    pygame.mixer.music.play()

# Initialize Tkinter root, hide the main window, and prompt for folder selection
root = tk.Tk()
root.withdraw() 
sound_folder = filedialog.askdirectory(title='Select Folder Containing Sound Files')  # Prompt user to select directory
root.destroy()  # Destroy the root window after selection

# Load ROI coordinates from CSV without headers
roi_df = pd.read_csv('auditory_rois.csv', header=None, names=['x', 'y', 'w', 'h'])

# Initialize video capture
cap = cv2.VideoCapture(0)  # Adjust the device index based on your camera setup

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Extract and scale the selected quadrant
    x1, y1 = selected_quadrant[0]
    x2, y2 = selected_quadrant[1]
    quadrant = frame[y1:y2, x1:x2]
    quadrant = cv2.resize(quadrant, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_LINEAR)

    # Process each ROI
    for index, row in roi_df.iterrows():
        x, y, w, h = int(row['x']), int(row['y']), int(row['w']), int(row['h'])
        roi = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Draw rectangle around ROI

        # Dummy condition for significant activity (this should be replaced with your actual condition)
        if cv2.meanStdDev(roi)[1][0][0] > 50:  # Random threshold for demonstration
            sound_path = f'{sound_folder}/{index+1}.wav'  # Sound files are named 1.wav, 2.wav, etc.
            threading.Thread(target=play_sound, args=(sound_path,)).start()

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
