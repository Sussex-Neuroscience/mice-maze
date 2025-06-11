import cv2 as cv
import numpy as np
'''import pygame

# Initialize Pygame for sound
pygame.init()
pygame.mixer.init()
sound = pygame.mixer.Sound('alert.wav')  # Replace 'alert.wav' with your sound file
'''
def select_rois(frame, num_rois):
    rois = [cv.selectROI("Frame", frame, False, False) for _ in range(num_rois)]
    cv.destroyAllWindows()
    return rois

'''
 #grab each area of interest and store them in a dictionary
        #this will be used to detect if the animal was there or not
        for item in rois:
            areas[item] = sf.grab_cut(gray,
                xstart = rois[item]["xstart"],
                ystart =  rois[item]["ystart"],
                xlen = rois[item]["xlen"],
                ylen =  rois[item]["ylen"],
                )
            
            cv.imshow(item,areas[item])
            #print(item+" " + str(np.sum(areas[item])))
    
            mousePresent[item] = np.sum(areas[item])<thresholds[item]/2
            
            if mousePresent[item]:
                hasVisited[item] = True
                #timeSpentAreas[item] = time.time()
            #print(item+" "+str(np.sum(areas[item])))
        
'''

def is_motion_detected(roi, frame, threshold=50):
    x, y, w, h = roi
    roi_frame = frame[y:y+h, x:x+w]
    avg_intensity = np.mean(roi_frame)
    return avg_intensity > threshold

def main():
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame")
        return

    # Convert to grayscale for ROI selection
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Let the user select a number of ROIs
    num_rois = int(input("Enter the number of ROIs: "))
    rois = select_rois(gray_frame, num_rois)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame")
            break

        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Check each ROI for motion
        for roi in rois:
            if is_motion_detected(roi, gray_frame):
                sound.play()

        cv.imshow("Frame", gray_frame)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
