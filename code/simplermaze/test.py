import numpy as np
import cv2 as cv
import pandas as pd
import supFun as sf


 
cap = cv.VideoCapture('/home/andre/Desktop/maze_test.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    ret,thresh1 = cv.threshold(gray,180,255,cv.THRESH_BINARY)
    
    cv.imshow('frame', thresh1[:,:,2])
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()

#sf.define
#pd.read_csv("roi.csv",index_col=0)

# 
# fileName = "trials_2columns.csv"
# #load ROI information
# rois = sf.csv_to_dict("rois.csv")
# 
# #set number of trials
# nTrials = 100
# 
# #start the camera object
# cap = sf.start_camera()
# 
# #get one frame from camera
# gray,valid = sf.grab_n_convert_frame(cameraHandle=cap)
# 
# #get areas defined in the rois csv:
# areas = dict()
# for item in rois:
#     areas[item] = gray[rois[item]["ystart"]:rois[item]["ystart"]+rois[item]["ylen"],
#                        rois[item]["xstart"]:rois[item]["xstart"]+rois[item]["xlen"],]
# 
# 
# 
# # When everything done, release the capture
# cap.release()
# cv.destroyAllWindows()
# 
# 
# #load the trials file (description of each trial)
# rois = sf.csv_to_dict("rois.csv")
# 
# thresholds = dict()
# for item in rois:
#     thresholds[item] = 900
# 
# mousePresent=dict()
# 
# for item in rois:
#     areas[item] = sf.grab_cut(gray,
#                 ystart = rois[item]["ystart"],
#                     ylen =  rois[item]["ylen"],
#                     xstart = rois[item]["xstart"],
#                     xlen =  rois[item]["xlen"],
#                     )
#                 
#     mousePresent[item] = np.sum(areas[item])>thresholds[item]
# 

#code dump


#             endGate2 = time.time()
#             gate2Durations.append((trial,endGate2-startGate2))
#             startTrialTime = time.time()
#             enteredMaze = 1

            
#         if gate2History[0] and not gate2History[1]:
#             print("mouse just entered entrance 2")
#             #endGate2 = time.time()
#             #gate1Durations.append((trial,endGate1-startGate1))

            
#         if  gate2History[0] and gate2History[1]:
#             print("mouse still on entrance 2 area")

#         if mousePresent["entrance1"]:
#             endGate1 = time.time()
#             if stillOnGate1==0:
#                 gate1 = gate1 + 1
#                 stillOnGate1=1
#         gate1Durations.append((trial,endGate1-startGate1))
#         
        
            
#         while mousePresent["entrance2"]==1:
#             endGate2 = time.time()
#             if stillOnGate2==0:
#                 gate2 = gate2 + 1
#                 stillOnGate2=1
#          
         
#          
#         if gate1>0 and gate2>0 :
#             if gate1>gate2:
#                 print("mouse has entered the maze")
#             
# 
#         #startTrial = 0
#         if gate1%2 != 0:
#             print("mouse has entered the maze")
#             #startTrial = 1
#             rewarded = 0
#         
#         if gate%2 == 0:
#             print("mouse has exited the maze")
#             
#         #if startTrial == 1: