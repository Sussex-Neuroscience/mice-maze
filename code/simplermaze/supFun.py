import pandas as pd
import cv2 as cv
import numpy as np

def get_metadata():
    anID = input("enter animal identification:")
    date = input("enter date:")
    sessionID = input("enter session identification")
    metatada = [anID, date, sessionID]
    #figure out what else users want to store
    return metadata

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