import tkinter as tk
from tkinter import simpledialog
from PIL import Image, ImageTk
import cv2
import os

class BackgroundSubtractionApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.vid = cv2.VideoCapture(0)

        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.canvas = tk.Canvas(window, width=self.width, height=self.height)
        self.canvas.pack()

        self.backSub = None
        # Keep a reference list for the photo images to prevent garbage collection
        self.photo_images = []
        self.update_interval = 20 # milliseconds, adjust based on performance
        self.update_video_frame()


        self.select_algo_button = tk.Button(window, text="Select Background Subtraction Method", command=self.select_algo)
        self.select_algo_button.pack(anchor=tk.CENTER, expand=True)

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update_video_frame(self):
        ret, frame = self.vid.read()
        if ret:
            if self.backSub:
                fgMask = self.backSub.apply(frame)
                frame_display = cv2.bitwise_and(frame, frame, mask=fgMask)
            else:
                frame_display = frame

            # Convert to grayscale for consistency in display
            frame_display = cv2.cvtColor(frame_display, cv2.COLOR_BGR2GRAY)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_display))
            # Append the photo image to the list to keep the reference
            self.photo_images.append(self.photo)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        # Call this method again after a short delay to update the frame
        self.window.after(self.update_interval, self.update_video_frame)

    def select_algo(self):
        algo_win = tk.Toplevel(self.window)
        algo_win.title('Select algorithm')
        algo_win.geometry('300x100')

        label = tk.Label(algo_win, text='Please select the background subtraction method:')
        label.pack()

        var = tk.StringVar(algo_win)
        var.set('MOG2')  # default value

        options = ['KNN', 'MOG2']
        menu = tk.OptionMenu(algo_win, var, *options)
        menu.pack()

        button = tk.Button(algo_win, text='OK', command=lambda: self.set_algo(var.get(), algo_win))
        button.pack()

    def set_algo(self, choice, algo_win):
        if choice == 'MOG2':
            self.backSub = cv2.createBackgroundSubtractorMOG2()
        else:
            self.backSub = cv2.createBackgroundSubtractorKNN()
        algo_win.destroy()

    def on_closing(self):
        self.vid.release()
        cv2.destroyAllWindows()
        self.window.destroy()  # Add parentheses to correctly call the method

if __name__ == "__main__":
    root = tk.Tk()
    app = BackgroundSubtractionApp(root, "Background Subtraction App")
    root.mainloop()
