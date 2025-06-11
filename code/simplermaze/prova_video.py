from __future__ import print_function
#from threshold_calibration import *
from threshold_calibration import video_calibration 
import supFun as sf
import cv2 as cv
import numpy as np
import argparse




cap= cv.VideoCapture(0)


# Call the video calibration function
thresh1, thresh2 = video_calibration(cap)

#check that camera is opened correctly
if not cap.isOpened():
    raise IOError("cannot open camera")

while True:
    ret, frame = cap.read() #ret = boolean returned by read function. it tells us if the frame was captured successfully. 
    #If it is, it is stored in variable frame
    frame = cv.resize(frame, None, fx= 0.5, fy= 0.5, interpolation= cv.INTER_AREA) 

    #apply thresholds from other script
    grayOriginal, valid = sf.grab_n_convert_frame(cameraHandle=cap)
    ret, gray = cv.threshold(grayOriginal, thresh1, thresh2, cv.THRESH_BINARY)

    cv.imshow("Input", gray)

    #press esc to exit
    c=cv.waitKey(1)
    if c==27: #ASCII of esc is 27
        break

'''

parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')

args = parser.parse_args()
if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()
    
capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)
while True:
    ret, frame = capture.read()
    if frame is None:
        break
    
    fgMask = backSub.apply(frame)
    
    
    cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    
    
    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)
    
    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break


'''
cap.release()
cv.destroyAllWindows()
