from __future__ import print_function
import tkinter as tk
from tkinter import Scale
import cv2 as cv
from PIL import Image, ImageTk
import supFun as sf
import cv2 as cv
import numpy as np
import argparse


def video_calibration(cap):
    # Create a Tkinter window
    window = tk.Tk()
    window.title("Video Calibration")

    # Variables to store threshold values
    thresh1 = tk.IntVar(value=180)
    thresh2 = tk.IntVar(value=255)

    # Function to update the video feed
    def update_video():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            window.quit()
            return

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, thresh_frame = cv.threshold(
            gray, thresh1.get(), thresh2.get(), cv.THRESH_BINARY
        )

        # Convert to a format Tkinter can use
        cvimage = cv.cvtColor(thresh_frame, cv.COLOR_GRAY2RGB)
        img = Image.fromarray(cvimage)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(10, update_video)

    # Function to save the threshold values and close the window
    def save_and_close():
        window.destroy()

    # Create scale widgets for threshold values
    scale1 = Scale(window, variable=thresh1, from_=0, to_=255, label="Threshold1")
    scale1.pack()
    scale2 = Scale(window, variable=thresh2, from_=0, to_=255, label="Threshold2")
    scale2.pack()

    # Create a label for displaying the video
    lmain = tk.Label(window)
    lmain.pack()

    # Create a button to save the threshold values and close the window
    save_button = tk.Button(window, text="Save and Close", command=save_and_close)
    save_button.pack()

    # Start the video update loop
    update_video()

    # Start the GUI event loop
    window.mainloop()

    return thresh1.get(), thresh2.get()


cap = cv.VideoCapture(0)


# Call the video calibration function
thresh1, thresh2 = video_calibration(cap)

# check that camera is opened correctly
if not cap.isOpened():
    raise IOError("cannot open camera")

while True:
    ret, frame = (
        cap.read()
    )  # ret = boolean returned by read function. it tells us if the frame was captured successfully.
    # If it is, it is stored in variable frame
    frame = cv.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)

    # apply thresholds from other script
    grayOriginal, valid = sf.grab_n_convert_frame(cameraHandle=cap)
    ret, gray = cv.threshold(grayOriginal, thresh1, thresh2, cv.THRESH_BINARY)

    cv.imshow("Input", gray)

    # press esc to exit
    c = cv.waitKey(1)
    if c == 27:  # ASCII of esc is 27
        break
