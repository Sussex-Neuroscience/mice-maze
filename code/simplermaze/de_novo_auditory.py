import csv
import tkinter as tk
from tkinter import filedialog



#get scaled parameters from crop_zoom.py
selected_quadrant=[(90, 128), (597, 388)]
scaling_factor=2

#retrieve rois coordinates from auditory_rois.csv, counting from 1 - because the sounds are indexed from 1
##these rois are already adapted to the selected quadrant, so there is no need to modify them 
def get_rois():
    rois = {}
    with open("auditory_rois.csv", newline='') as file:
        reader = csv.reader(file)
        for idx, row in enumerate(reader, start=1):
            if len(row) != 4:
                continue
            x1, y1, x2, y2 = map(int, row)
            rois[str(idx)] = [(x1, y1), (x2, y2)]
    return(rois)


#select folder containing sounds
def select_sound_folder():
    root = tk.Tk()
    root.withdraw() 
    sound_folder = filedialog.askdirectory(title='Select Folder Containing Sound Files')  # Prompt user to select directory
    root.destroy()  # Destroy the root window after selection

#map sounds to rois





#convert video to gray scale and display it

#display rois
def display_rois():
    pass
#motion detection in rois
def motion_detection(roi):

    pass

