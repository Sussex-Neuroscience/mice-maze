import numpy as np
import cv2 as cv
import pandas as pd
import time
import os
import os.path

# Removed unnecessary imports for clarity

# Remove or comment out custom start_camera function as its implementation is unknown
# Assuming cap = cv.VideoCapture(0) is sufficient for starting the camera

#set variable for defining if this is an habituation run:
habituation = True

#set if the video should be recorded during the session
#recordVideo = False
#set filename for video recording

# get the current date and time
local_time = time.localtime()


# format the date and time as YYYY-MM-DD_hh-mm-ss
date_time = str(time.strftime('%Y-%m-%d', local_time))

animal_ID = str(input("Insert animal ID: \n")).upper()
session_ID = str(input("Insert Session ID: \n")).lower()

# Construct the path to the desktop for the current user
base_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'maze_recordings')

# Ensure the base directory exists
if not os.path.exists(base_path):
    os.makedirs(base_path)
    print(f"Base directory created: {base_path}")
else:
    print(f"Base directory already exists: {base_path}")

# Construct the path for the new directory
new_directory = f"{date_time}_{animal_ID}_{session_ID}"
new_dir_path = os.path.join(base_path, new_directory)

# Ensure the session directory exists
if not os.path.exists(new_dir_path):
    os.makedirs(new_dir_path)
    print(f"Session directory created: {new_dir_path}")
else:
    print(f"Session directory already exists: {new_dir_path}")

specific_time = str(time.strftime('%H_%M_%S', local_time)) # Changed to use dashes for compatibility
rec_name = f"{animal_ID}_{specific_time}.avi"
recordings = os.path.join(new_dir_path, rec_name)

#if name is trial based, I need to insert this in the for loop

# Confirming the final output path
print(f"Video will be saved to: {recordings}")

#look into parallel processing to let the user fill the data form as the video goes
#metadata .csv 

data = {
  "date": (),
  "animal ID": animal_ID,
  "duration": [],
  "ear mark identifier" : (str(input("ear mark identifiers? (y \ n) : \n")).lower()),
  "session ID": session_ID,
  "session start" :(),
  "session end":(),
  "animal weight": (str(input("insert animal weight (g): \n")).lower()),
  "animal birth date":(str(input("insert animal birth date (dd\mm\yyyy): \n"))),
  "animal gender": (str(input("insert animal assumed gender (m \ f): \n")).lower()),
  #"rewards":(),
  #"punishment wrong location":(),
  #"reward probability": []

}

#load data into a DataFrame object:
df = pd.DataFrame(data)
csv_name=  f"{animal_ID}_{specific_time}.csv"

df.to_csv(os.path.join(new_dir_path, csv_name))


cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera could not be opened.")
    exit()

# Retrieve and print camera properties
frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv.CAP_PROP_FPS))
frameSize = (frame_width, frame_height)

cc = cv.VideoWriter_fourcc(*'XVID')
videoFile = cv.VideoWriter(recordings, cc, fps, frameSize)

# Check if the VideoWriter object was successfully created
if not videoFile.isOpened():
    print("Error: Failed to create VideoWriter object.")
    cap.release()
    exit()

while True: 
    ret, frame = cap.read()
    if ret:
        videoFile.write(frame)
        cv.imshow("Frame", frame)
        if cv.waitKey(1) & 0xFF in [ord('q'), 27]:  # Quit on 'q' or ESC
            break
    else:
        print("Error: Failed to read frame from camera.")
        break

# Release resources
videoFile.release()
cap.release()
cv.destroyAllWindows()
