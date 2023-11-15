import numpy as np
import cv2 as cv
import pandas as pd
import supFun as sf
#from 


#load ROI information
rois = sf.read_rois("rois.csv")

#set number of trials
nTrials = 100

#start the camera object
cap = sf.start_camera()


#infinite loop (needs to be changed later)
while True:
    gray,valid = sf.grab_n_convert_frame(cameraHandle=cap)
    if not valid:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Display the resulting frame
    cv.imshow('frame', gray)
    
    
    #crop roi from original image
    entrance1Crop=gray[rois["entrance1"]["ystart"]:rois["entrance1"]["ystart"]+rois["entrance1"]["ylen"],
                       rois["entrance1"]["xstart"]:rois["entrance1"]["xstart"]+rois["entrance1"]["xlen"],]

    entrance2Crop=gray[rois["entrance2"]["ystart"]:rois["entrance2"]["ystart"]+rois["entrance2"]["ylen"],
                       rois["entrance2"]["xstart"]:rois["entrance2"]["xstart"]+rois["entrance2"]["xlen"],]
    
    rewArea1Crop=gray[rois["rewArea1"]["ystart"]:rois["rewArea1"]["ystart"]+rois["rewArea1"]["ylen"],
                      rois["rewArea1"]["xstart"]:rois["rewArea1"]["xstart"]+rois["rewArea1"]["xlen"],]
    rewArea2Crop=gray[rois["rewArea2"]["ystart"]:rois["rewArea2"]["ystart"]+rois["rewArea2"]["ylen"],
                      rois["rewArea2"]["xstart"]:rois["rewArea2"]["xstart"]+rois["rewArea2"]["xlen"],]
    rewArea3Crop=gray[rois["rewArea3"]["ystart"]:rois["rewArea3"]["ystart"]+rois["rewArea3"]["ylen"],
                      rois["rewArea3"]["xstart"]:rois["rewArea3"]["xstart"]+rois["rewArea3"]["xlen"],]
    rewArea4Crop=gray[rois["rewArea4"]["ystart"]:rois["rewArea4"]["ystart"]+rois["rewArea4"]["ylen"],
                      rois["rewArea4"]["xstart"]:rois["rewArea4"]["xstart"]+rois["rewArea4"]["xlen"],]
    
    
    for trial in range(nTrials):
        #print("start trial "+str(trial))
        #show cropped image
        cv.imshow("entrace1Crop",entrance1Crop)
        cv.imshow("entrace2Crop",entrance2Crop)
        cv.imshow("rewArea1Crop",rewArea1Crop)
        cv.imshow("rewArea2Crop",rewArea2Crop)
        cv.imshow("rewArea3Crop",rewArea3Crop)
        cv.imshow("rewArea4Crop",rewArea4Crop)
    
    
    
        crossE1Value = 900
        crossE2Value = 900
        #detect maze entrance:
        while crossE1Value<1000:
            crossE1Value = np.sum(entrance1Crop)
        crossE1 = True
    
        while crossE2Value<1000:
            crossE2Value = np.sum(entrance2Crop)
        crossE2 = True
        
        
    
    
    
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

