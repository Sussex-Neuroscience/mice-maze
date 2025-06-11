import tkinter as tk
from PIL import Image, ImageTk
import cv2
import csv
import os

class WebcamApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # OpenCV: 0 is usually the ID of the built-in webcam
        self.vid = cv2.VideoCapture(0)

        # Setting webcam resolution
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()

        # Button to select ROI
        self.btn_select_roi = tk.Button(window, text="Select ROI", width=15, command=self.select_roi)
        self.btn_select_roi.pack(anchor=tk.CENTER, expand=True)

        # Selected ROIs
        self.rois = []
        self.load_rois()

        self.delay = 10
        self.update()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)  # Handle window closing

        self.window.mainloop()

    def select_roi(self):
        # Reset current ROI selection
        self.current_roi = None
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

    def on_mouse_down(self, event):
        self.current_roi = [(event.x, event.y)]  # Starting point

    def on_mouse_move(self, event):
        if self.current_roi:
            temp_frame = self.current_frame.copy()
            pt1 = self.current_roi[0]
            pt2 = (event.x, event.y)
            cv2.rectangle(temp_frame, pt1, pt2, (0, 0, 255), 2)
            self.display_frame(temp_frame)  # Display the frame with the temporary rectangle

    def on_mouse_up(self, event):
        if self.current_roi:
            pt2 = (event.x, event.y)
            self.rois.append((self.current_roi[0], pt2))
            self.canvas.unbind("<ButtonPress-1>")
            self.canvas.unbind("<B1-Motion>")
            self.canvas.unbind("<ButtonRelease-1>")
            if len(self.rois) == 6:  # If three ROIs have been selected, stop allowing new selections
                self.btn_select_roi.config(state="disabled")

    def on_closing(self):
        self.vid.release()
        self.window.destroy()

    def load_rois(self):
        # Check if "auditory_rois.csv" exists
        if os.path.exists("auditory_rois.csv"):
            with open("auditory_rois.csv", newline='') as csvfile:
                roi_reader = csv.reader(csvfile, delimiter=',')
                for row in roi_reader:
                    # Each row in the CSV should contain x1, y1, x2, y2 which are the coordinates for a ROI
                    self.rois.append(((int(row[0]), int(row[1])), (int(row[2]), int(row[3]))))
            # Disable button as ROIs are loaded and no more selection is needed
            self.btn_select_roi.config(state="disabled")
        else:
            # Enable button to allow ROI selection if file does not exist
            self.btn_select_roi.config(state="normal")

    def save_rois(self):
        # Save the selected ROIs to "auditory_rois.csv"
        with open("auditory_rois.csv", mode='w', newline='') as csvfile:
            roi_writer = csv.writer(csvfile, delimiter=',')
            for roi in self.rois:
                pt1, pt2 = roi
                roi_writer.writerow([pt1[0], pt1[1], pt2[0], pt2[1]])

    def update(self):
        # Get a frame from the webcam
        ret, frame = self.vid.read()
        if ret:
            self.current_frame = frame  # Store the current frame
            self.display_frame(frame)
        self.window.after(self.delay, self.update)

    def display_frame(self, frame):
        # Draw the ROIs on the frame before displaying
        for roi in self.rois:
            pt1, pt2 = roi
            cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    # The rest of the class remains the same as previously provided, with the addition of self.save_rois() call in on_mouse_up

    def on_mouse_up(self, event):
        if self.current_roi:
            pt2 = (event.x, event.y)
            self.rois.append((self.current_roi[0], pt2))
            self.save_rois()  # Save ROIs after each selection
            self.canvas.unbind("<ButtonPress-1>")
            self.canvas.unbind("<B1-Motion>")
            self.canvas.unbind("<ButtonRelease-1>")
            if len(self.rois) == 6:  # If three ROIs have been selected, stop allowing new selections
                self.btn_select_roi.config(state="disabled")

    def on_closing(self):
        self.vid.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    WebcamApp(root, "ROIs")
