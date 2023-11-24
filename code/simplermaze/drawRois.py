import numpy as np
import cv2 as cv


#"":,
rois = { "entrance1":(),"entrance2":(),"rew1":(),"rew2":(),"rew3":(),"rew4":()}

cap = cv.VideoCapture(0)
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
gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
# Display the resulting frame
cv.imshow('frame', gray)

#release the capture
cap.release()

for entry in rois:
    print("please select location of "+str(entry))
    rois[entry] = cv.selectROI('frame', gray)    
    
#when all done destroy windows
cv.destroyAllWindows()

