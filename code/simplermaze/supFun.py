import pandas as pd
import cv2 as cv
import numpy as np

def write_text(text="string",window="noWindow"):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,500)
    fontScale              = 1
    fontColor              = (255,255,255)
    thickness              = 1
    lineType               = 2
    
    cv2.putText(window,text, 
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
        print("Cannot open camera")
        exit()
    return cap

def grab_n_convert_frame(cameraHandle):
    #capture a frame
    ret, frame = cameraHandle.read()
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    return gray,ret 

def read_rois(fileName="rois.csv"):
    rois = dict()
    temp =  pd.read_csv(fileName,index_col=0)
    temp = temp.transpose()
    rois = temp.to_dict()
    return rois

def define_rois():
    pass

    
