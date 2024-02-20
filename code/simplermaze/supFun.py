import pandas as pd
import cv2 as cv
import numpy as np
import pandas as pd
import time
import os
from tkinter import filedialog as fd
from tkinter import *

def get_metadata():
    anID = input("enter animal identification:")
    date = input("enter date:")
    sessionID = input("enter session identification")
    metadata = [anID, date, sessionID]
    #figure out what else users want to store
    return metadata

def write_text(text="string",window="noWindow"):
    font                   = cv.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,500)
    fontScale              = 1
    fontColor              = (255,255,255)
    thickness              = 1
    lineType               = 2
    
    cv.putText(window,text, 
    bottomLeftCornerOfText, 
    font, 
    fontScale,
    fontColor,
    thickness,
    lineType)
    
    return


def start_camera():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        exit()
    return cap


def record_video(cap, recordings, frame_width, frame_height, fps):
    cc = cv.VideoWriter_fourcc(*'XVID')
    videoFile = cv.VideoWriter(recordings, cc, fps, (frame_width, frame_height))

    if not videoFile.isOpened():
        print("Error: Failed to create VideoWriter object.")
        cap.release()
        exit()

    while True:
        ret, frame = cap.read()
        if ret:
            videoFile.write(frame)
            cv.imshow("Frame", frame)
            if cv.waitKey(1) & 0xFF in [ord('q'), 27]:
                break
        else:
            print("Error: Failed to read frame from camera.")
            break

    videoFile.release()

def grab_n_convert_frame(cameraHandle):
    #capture a frame
    ret, frame = cameraHandle.read()
    #print(ret)
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    return gray,ret 

def csv_to_dict(fileName="rois.csv"):
    output = dict()
    temp =  pd.read_csv(fileName,index_col=0)
    temp = temp.transpose()
    output = temp.to_dict()
    return output


def define_rois(roiNames = ["entrance1","entrance2",
                             "rew1","rew2","rew3","rew4"],
                outputName = "rois.csv"):
    

    cap = cv.VideoCapture('/home/andre/Desktop/maze_test.mp4')
    #cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        #break
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # Display the resulting frame
    #cv.imshow('frame', gray)

    #release the capture
    cap.release()

    rois = {}
    for entry in roiNames:
        print("please select location of "+str(entry))
        rois[entry] = cv.selectROI('frame', gray)
        #print(rois[entry])
    
    df = pd.DataFrame(rois)
    #print(df)
    #print(roiNames)
    df.index = ["xstart","ystart","xlen","ylen"]
    df.to_csv(outputName,index=["xstart","ystart","xlen","ylen"])
    #when all done destroy windows
    cv.destroyAllWindows()

    return df

#def draw_on_video(frame=gray,roi=(0,0,0,0)):
    #cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (0, 0, 0), -1)
    

def grab_cut(frame,xstart,ystart,xlen,ylen):
    
    cut = frame[ystart:ystart+ylen,
                xstart:xstart+xlen]
    return cut

def get_current_time_formatted():
    return time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime())

def get_user_inputs():
    animal_ID = input("Insert animal ID: \n").upper()
    session_ID = input("Insert Session ID: \n").lower()
    return animal_ID, session_ID

def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created: {path}")
    else:
        print(f"Directory already exists: {path}")

def setup_directories(base_path, date_time, animal_ID, session_ID):
    new_directory = f"{date_time}_{animal_ID}_{session_ID}"
    new_dir_path = os.path.join(base_path, new_directory)
    ensure_directory_exists(new_dir_path)
    return new_dir_path

def collect_metadata(animal_ID, session_ID):
    ear_mark = input("Ear mark identifiers? (y/n): \n").lower()
    weight = input("Insert animal weight (g): \n").lower()
    birth_date = input("Insert animal birth date (dd/mm/yyyy): \n")
    gender = input("Insert animal assumed gender (m/f): \n").lower()

    data = {
        "animal ID": animal_ID,
        "session ID": session_ID,
        "ear mark identifier": ear_mark,
        "animal weight": weight,
        "animal birth date": birth_date,
        "animal gender": gender,
    }
    
    return data

def save_metadata_to_csv(data, new_dir_path, file_name):
    df = pd.DataFrame([data])  # Convert dict to dataframe
    csv_path = os.path.join(new_dir_path, file_name)
    df.to_csv(csv_path, index=False)
    print(f"Metadata saved to: {csv_path}")

def choose_csv():
    root=Tk()
    # Show the file dialog and get the selected file name
    filename = fd.askopenfilename()
    # Print the file name to the console
    #print (filename)
    root.destroy()
    return filename

