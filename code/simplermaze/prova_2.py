import tkinter as tk
from PIL import Image, ImageTk
import cv2
import csv
import os
#this does bg subtraction

class WebcamApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.vid = cv2.VideoCapture(0)
        
        self.canvas = tk.Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        self.btn_select_roi = tk.Button(window, text="Select ROI", width=15, command=self.select_roi)
        self.btn_select_roi.pack(anchor=tk.CENTER, expand=True)

        # Button for selecting the background subtraction algorithm
        self.btn_select_algo = tk.Button(window, text="Select Background Subtraction Method", width=30, command=self.select_algo)
        self.btn_select_algo.pack(anchor=tk.CENTER, expand=True)

        self.rois = []
        self.load_rois()
        self.backSub = None

        self.delay = 15
        self.update()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)


    def load_rois(self):
        if os.path.exists("auditory_rois.csv"):
            with open("auditory_rois.csv", newline='') as csvfile:
                roi_reader = csv.reader(csvfile, delimiter=',')
                for row in roi_reader:
                    self.rois.append(((int(row[0]), int(row[1])), (int(row[2]), int(row[3]))))
            self.btn_select_roi.config(state="disabled")
        else:
            self.btn_select_roi.config(state="normal")

    def select_roi(self):
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.current_roi = None

    def on_mouse_down(self, event):
        self.current_roi = [(event.x, event.y)]

    def on_mouse_move(self, event):
        if self.current_roi:
            self.canvas.delete("roi-rect")
            self.current_roi[1:] = [(event.x, event.y)]
            self.canvas.create_rectangle(self.current_roi[0], self.current_roi[1], outline="red", tags="roi-rect")

    def on_mouse_up(self, event):
        if self.current_roi:
            pt2 = (event.x, event.y)
            self.rois.append((self.current_roi[0], pt2))
            self.save_rois()
            self.canvas.unbind("<ButtonPress-1>")
            self.canvas.unbind("<B1-Motion>")
            self.canvas.unbind("<ButtonRelease-1>")
            self.canvas.delete("roi-rect")
            self.select_algo()

    def save_rois(self):
        with open("auditory_rois.csv", mode='w', newline='') as csvfile:
            roi_writer = csv.writer(csvfile, delimiter=',')
            for roi in self.rois:
                roi_writer.writerow([roi[0][0], roi[0][1], roi[1][0], roi[1][1]])

    def select_algo(self):
        self.algo_win = tk.Toplevel(self.window)
        self.algo_win.title('Select algorithm')
        self.algo_win.geometry('300x100')
        
        label = tk.Label(self.algo_win, text='Please select the background subtraction method:')
        label.pack()

        var = tk.StringVar(self.algo_win)
        var.set('MOG2')  # default value

        options = ['KNN', 'MOG2']
        menu = tk.OptionMenu(self.algo_win, var, *options)
        menu.pack()

        button = tk.Button(self.algo_win, text='OK', command=lambda: self.set_algo(var.get()))
        button.pack()

    def set_algo(self, choice):
        if choice == 'MOG2':
            self.backSub = cv2.createBackgroundSubtractorMOG2()
        else:
            self.backSub = cv2.createBackgroundSubtractorKNN()
        self.algo_win.destroy()

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            if self.backSub:
                fgMask = self.backSub.apply(frame)
                frame_display = cv2.bitwise_and(frame, frame, mask=fgMask)
            else:
                frame_display = frame

            for roi in self.rois:
                cv2.rectangle(frame_display, roi[0], roi[1], (0, 255, 0), 2)

            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)

    def on_closing(self):
        self.vid.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = WebcamApp(root, "Webcam with Tkinter and OpenCV")
    root.mainloop()
