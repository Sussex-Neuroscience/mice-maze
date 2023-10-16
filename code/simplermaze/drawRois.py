import numpy as np
import cv2 as cv




cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

firstFrame=1
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv.imshow('frame', gray)
    if firstFrame==1:
        entrance1=cv.selectROI('frame', gray)
        firstFrame=0
    
    #crop roi from original image
    entrance1Crop=gray[entrance1[0]:entrance1[0]+entrance1[2],
                     entrance1[1]:entrance1[1]+entrance1[3]]

    #show cropped image
    cv.imshow("entrace1Crop",entrance1Crop)

    
    
    
    
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

