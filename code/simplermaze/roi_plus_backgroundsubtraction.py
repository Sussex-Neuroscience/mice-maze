import tkinter as tk
from tkinter import simpledialog
from PIL import Image, ImageTk
import cv2
import csv
import os
import threading


class WebcamApp:
    def __init__(self, window, window_title):
        # Define the main application window
        self.window = window
        self.window.title(window_title)
        self.vid = cv2.VideoCapture(0)

        # Set canvas to the size of the video
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.canvas = tk.Canvas(window, width=self.width, height=self.height)
        self.canvas.pack()

        # Button to select the initial ROI
        self.btn_select_roi = tk.Button(window, text="Select initial ROI", width=15, command=self.enable_roi_selection)
        self.btn_select_roi.pack(anchor=tk.CENTER, expand=True)

        # Button to move the ROI
        self.btn_move_roi = tk.Button(window, text="Move ROI", width=15, command=self.toggle_move_mode)
        self.btn_move_roi.pack(anchor=tk.CENTER, expand=True)

        # Button to delete an ROI
        self.btn_delete_roi = tk.Button(window, text="Delete ROI", width=15, command=self.enable_roi_deletion)
        self.btn_delete_roi.pack(anchor=tk.CENTER, expand=True)

        # Button to finish ROI selection
        self.btn_finish_roi = tk.Button(window, text="Finish ROI Selection", width=20, command=self.finish_roi_selection)
        self.btn_finish_roi.pack(anchor=tk.CENTER, expand=True)

        # Initialize variables
        self.rois = []
        self.load_rois()
        self.backSub = None
        self.moving_mode = False
        self.selected_roi_index = None
        self.initial_click_position = None
        self.current_roi = None
        self.show_background_subtraction = False

        # Start the video frame update loop
        self.update_video_frame()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def enable_roi_selection(self):
        self.moving_mode = False
        self.selected_roi_index = None
        self.current_roi = None
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

    def toggle_move_mode(self):
        self.moving_mode = not self.moving_mode
        self.current_roi = None
        if self.moving_mode:
            self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
            self.canvas.bind("<B1-Motion>", self.on_mouse_move)
            self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
            self.btn_move_roi.config(text="Select new ROI")
        else:
            self.enable_roi_selection()
            self.btn_move_roi.config(text="Move ROI")

    def load_rois(self):
        if os.path.exists("auditory_rois.csv"):
            with open("auditory_rois.csv", newline='') as csvfile:
                roi_reader = csv.reader(csvfile, delimiter=',')
                for row in roi_reader:
                    self.rois.append(((int(row[0]), int(row[1])), (int(row[2]), int(row[3]))))
            self.btn_select_roi.config(state="disabled")
        else:
            self.btn_select_roi.config(state="normal")

    def on_mouse_down(self, event):
        if self.moving_mode:
            self.initial_click_position = (event.x, event.y)
            for i, roi in enumerate(self.rois):
                if roi[0][0] <= event.x <= roi[1][0] and roi[0][1] <= event.y <= roi[1][1]:
                    self.selected_roi_index = i
                    return
        else:
            self.current_roi = [(event.x, event.y)]
            self.selected_roi_index = None

    def on_mouse_move(self, event):
        if self.moving_mode and self.selected_roi_index is not None:
            dx = event.x - self.initial_click_position[0]
            dy = event.y - self.initial_click_position[1]
            self.move_selected_roi(dx, dy)
            self.initial_click_position = (event.x, event.y)
        elif not self.moving_mode and self.current_roi:
            self.canvas.delete("roi-rect")
            self.current_roi[1:] = [(event.x, event.y)]
            self.canvas.create_rectangle(self.current_roi[0], self.current_roi[1], outline="green", width= 4, tags="roi-rect")

    def on_mouse_up(self, event):
        if self.moving_mode and self.selected_roi_index is not None:
            self.save_rois()
            self.selected_roi_index = None
        elif not self.moving_mode and self.current_roi:
            pt2 = (event.x, event.y)
            self.rois.append((self.current_roi[0], pt2))
            self.save_rois()
            self.current_roi = None
            self.canvas.delete("roi-rect")

    def move_selected_roi(self, dx, dy):
        roi = self.rois[self.selected_roi_index]
        moved_roi = ((roi[0][0] + dx, roi[0][1] + dy), (roi[1][0] + dx, roi[1][1] + dy))
        self.rois[self.selected_roi_index] = moved_roi
        self.redraw_rois()

    def save_rois(self):
        with open("auditory_rois.csv", mode='w', newline='') as csvfile:
            roi_writer = csv.writer(csvfile, delimiter=',')
            for roi in self.rois:
                roi_writer.writerow([roi[0][0], roi[0][1], roi[1][0], roi[1][1]])

    def update_video_frame(self):
        ret, frame = self.vid.read()
        if ret:
            # If background subtraction is enabled, apply it
            if self.backSub:
                fgMask = self.backSub.apply(frame)
                frame_display = cv2.bitwise_and(frame, frame, mask=fgMask)
            else:
                frame_display = frame

            # Convert the frame to a format suitable for Tkinter to display
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame_display, cv2.COLOR_BGR2GRAY)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

            # Redraw the ROIs on the updated frame
            self.redraw_rois()

        # Schedule the next frame update
        self.window.after(10, self.update_video_frame)#self.vid was defined in the class at first, with the properties of the webcam video
    
    #ret is a boolean that returns True if the frame was successfully captured
    
            

    def redraw_rois(self):
        self.canvas.delete("roi-rect")
        for index, roi in enumerate(self.rois):
            self.canvas.create_rectangle(roi[0], roi[1], outline="green", width=4, tags="roi-rect")
        # Calculate the position for the text. This places the text in the top-left corner of the ROI.
            text_position = (roi[0][0] -6, roi[0][1] - 6)
        # Draw the text. You can change the font size as needed.
            self.canvas.create_text(text_position, text="{}".format(index + 1), fill="red", font=('Helvetica', '10'), tags="roi-rect")
        
    def enable_roi_deletion(self):
        self.moving_mode = False  # Ensure moving mode is disabled
        self.current_roi = None  # Clear current ROI selection if any
        self.selected_roi_index = None  # Clear any selected ROI index
        # Unbind previous bindings that are not related to deletion
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")
        # Bind left mouse click to the deletion function
        self.canvas.bind("<ButtonPress-1>", self.on_roi_delete_click)

    def on_roi_delete_click(self, event):
        for i, roi in enumerate(self.rois):
            if roi[0][0] <= event.x <= roi[1][0] and roi[0][1] <= event.y <= roi[1][1]:
                del self.rois[i]  # Delete the selected ROI
                self.save_rois()  # Save the updated list of ROIs
                self.redraw_rois()  # Redraw the ROIs on the canvas
                break  # Exit the loop once the ROI is found and deleted

    def finish_roi_selection(self):
        # Unbind the ROI selection events
        self.canvas.unbind("<ButtonPress-1>")
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")

        self.show_background_subtraction = True  # Now ready to show the background subtraction
        # Optionally, you can clear the current ROI to prevent accidental inclusion if partially drawn
        self.current_roi = None

            # Hide the buttons
        self.btn_select_roi.pack_forget()
        self.btn_move_roi.pack_forget()
        self.btn_delete_roi.pack_forget()
        self.btn_finish_roi.pack_forget()
        # Open the background subtraction method selection window
        self.select_algo()

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

        #after algo is chosen, destroy the window    
        self.algo_win.destroy()

        # Add a button to save the video
        self.btn_save_video = tk.Button(self.window, text="Save Video", width=15, command=self.save_video)
        self.btn_save_video.pack(anchor=tk.CENTER, expand=True)



    def on_closing(self):
        self.vid.release()
        cv2.destroyAllWindows()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = WebcamApp(root, "Fucking work please")
    root.mainloop()