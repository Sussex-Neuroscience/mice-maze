import cv2 
import numpy as np
import sys
import pickle

#slightly modified roi selection. This one prompts the user to select the specific reward rois (line 52 and 60)

def select_rois():
    global pt1, pt2, topLeft_clicked, botRight_clicked
    n_ROI = 4  # how many ROIs you want to select 

    # mouse callback function to select the ROIs
    def draw_rectangle(event, x, y, flags, param):
        global pt1, pt2, topLeft_clicked, botRight_clicked

        # get mouse click
        if event == cv2.EVENT_LBUTTONDOWN:

            if topLeft_clicked == True and botRight_clicked == True:
                topLeft_clicked = False
                botRight_clicked = False
                pt1 = (0, 0)
                pt2 = (0, 0)

            if topLeft_clicked == False:
                pt1 = (x, y)
                topLeft_clicked = True
                
            elif botRight_clicked == False:
                pt2 = (x, y)
                botRight_clicked = True

    # Haven't drawn anything yet!
    reward_pts = list()  # np.zeros(2) # 2 because there are 2 points to identify each region
    pt1 = (0, 0)
    pt2 = (0, 0)
    topLeft_clicked = False
    botRight_clicked = False

    cap = cv2.VideoCapture(0)  # add a 0 to connect to the camera 

    # Create a named window for connections
    cv2.namedWindow('Test')

    # Bind draw_rectangle function to mouse clicks
    cv2.setMouseCallback('Test', draw_rectangle) 

    # Initialize a counter variable
    counter = 0

    # Create a list of column names for the ROIs
    columns = ['LL ROI', 'LR ROI', 'RL ROI', 'RR ROI']

    while counter < len(columns):

        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display the message on the frame
        cv2.putText(frame, f"Please select {columns[counter]}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if topLeft_clicked:
            cv2.circle(frame, center=pt1, radius=5, color=(0, 0, 255), thickness=-1)

        # drawing rectangle
        if topLeft_clicked and botRight_clicked:
            cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)
            pts = list([pt1, pt2])
            # Add another condition to check if the width and height of the rectangle are positive
            if pt2[0] - pt1[0] > 0 and pt2[1] - pt1[1] > 0:
                # Append the pts to the reward_pts list
                reward_pts.append(pts)
                topLeft_clicked = False
                botRight_clicked = False
                counter += 1

        # Display the resulting frame
        cv2.imshow('Test', frame)

        # This command let's us quit with the "q" button on a keyboard.
        # Simply pressing X on the window won't work!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    reward_pts = np.unique(reward_pts, axis=0)
    pickle.dump(reward_pts, sys.stdout)
    return reward_pts


