import tkinter as tk
from tkinter import Scale
import cv2 as cv
from PIL import Image, ImageTk

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
        ret, thresh_frame = cv.threshold(gray, thresh1.get(), thresh2.get(), cv.THRESH_BINARY)

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
