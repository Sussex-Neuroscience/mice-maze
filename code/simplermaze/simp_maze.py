import numpy as np
import cv2 as cv
import pandas as pd
import supFun as sf
import time
import serial
import os

# Set variable for defining if this is a habituation run:
habituation = True

# Set if the video should be recorded during the session
recordVideo = False

# Set filename for video recording
recVideoName = ""

# Get the current date and time
local_time = time.localtime()

# Format the date and time as YYYY-MM-DD_hh-mm-ss
date_time = str(time.strftime('%d %b %Y %H:%M:%S', local_time))

# Append the date and time to the file name
recVideoName = "test_" + date_time + ".mp4"

# Set variable for defining if animal should be rewarded in the case
# a wrong location is visited before visiting a correct location:
considerWrongLocations = False

# Create a serial object and connect to it
# ser = serial.Serial("/dev/ttyACM0")

drawRois = False

if drawRois:
    sf.define_rois(roiNames=["entrance1", "entrance2", "rew1", "rew2", "rew3", "rew4"],
                    outputName="rois1.csv")

# Load ROI information
rois = pd.read_csv("rois1.csv", index_col=0)

# Load the trials file (description of each trial)
trialsIDs = pd.read_csv("trials_2columns.csv")

# Set number of trials
nTrials = len(trialsIDs)

# Get the identification of each grating
gratingID = [match for match in list(trialsIDs.keys()) if "motor" in match]

# Get the number of reward areas
nRewAreas = trialsIDs.ra1.max()

# Set the port/address for each motor
gratingMotors = {"motor 1": 1, "motor 2": 2, "motor 3": 3, "motor 4": 4, "motor 5": 5, "motor 6": 6}

rewardMotors = {"motor 1": 7, "motor 2": 8, "motor 3": 9, "motor 4": 10}

# Create dictionaries to store threshold values and time spent in areas
thresholds = {}
timeSpentAreas = {}

for item in rois:
    # Create threshold values for each area
    thresholds[item] = 0
    # Create all ROI variables in the dictionary and attach an empty list to them
    timeSpentAreas[item] = np.zeros(nTrials)

# Create variables that will store session data
areasRewarded = []
hits = []
miss = []
incorrect = []

# Preload variables
hasVisited = {}
mousePresent = {}

# Start the camera object
cap = cv.VideoCapture(0)

# START FOR LOOP TO AVERAGE N FRAMES FOR THRESHOLDING

# Grab one frame to adjust thresholds of empty spaces:
gray, valid = sf.grab_n_convert_frame(cameraHandle=cap)
ret, gray = cv.threshold(gray, 180, 255, cv.THRESH_BINARY)

# Run a loop to catch each area and sum the pixel values on that area of the frame
areas = {}

for item in rois:
    areas[item] = sf.grab_cut(gray,
                              xstart=rois[item]["xstart"],
                              ystart=rois[item]["ystart"],
                              xlen=rois[item]["xlen"],
                              ylen=rois[item]["ylen"])
    thresholds[item] = np.sum(areas[item])

# Make directory where videos will be saved
if not os.path.exists("~/Desktop/maze_recordings"):
    os.makedirs("~/Desktop/maze_recordings")

    # Expand the tilde to the full path of the home directory
    homeDir = os.path.expanduser("~")

    # Construct the file path from the directory and the file name using forward slashes
    filePath = homeDir + "/Desktop/maze_recordings/" + recVideoName

else:
    # Expand the tilde to the full path of the home directory
    homeDir = os.path.expanduser("~")
    # Construct the file path from the directory and the file name using forward slashes
    filePath = homeDir + "/Desktop/maze_recordings/" + recVideoName


# cap = sf.start_camera()
cap = cv.VideoCapture(filePath)
sessionStartTime = time.time()

if recordVideo:
    frame_width = int(cap.get(3))  # Can also use cam.get (cv.CAP_PROP_FRAME_WIDTH)
    frame_height = int(cap.get(4))  # Can also use cam.get (cv.CAP_PROP_FRAME_HEIGHT)
    fps = int(cap.get(5))  # Can also use cam.get (cv.CAP_PROP_FPS)

    cc = cv.VideoWriter_fourcc(*'XVID')  # Can also use mp4v or MJPG for .avi, must check with maze camera
    frameSize = (frame_width, frame_height)

    videoFile = cv.VideoWriter(recVideoName, cc, fps, frameSize)

trial_durations = []  # List that will contain the durations of every trial. Maybe this can be stored in another CSV file?

for trial in range(nTrials):
    # Time at the beginning of the trial
    start_trial_time = time.time()
    print("starting trial " + str(trial + 1))

    # Which area should be rewarded
    rewardTarget = trialsIDs["ra1"][trial]

    # Define which servo motor should be used for this specific area
    for item in rewardMotors.keys():
        if str(rewardTarget) in item:
            rewMotor = rewardMotors[item]
            break
    else:
        rewMotor = 7

    # Run code to start the gratings
    for grating in gratingID:
        print(grating)
        # Add Arduino code here so that all gratings turn to neutral position
        # ser.print(xxxxxxxx)
        # Maybe add a pause so that the servo motors have time to catch up

    for grating in gratingID:
        if habituation:
            position = 0
        else:
            position = trialsIDs[grating][trial]

        print(position)
        # Then add code so that they turn to the trial specific position
        # Maybe add a pause so that the servo motors have time to catch up

    for item in rois:
        hasVisited[item] = False

    trialOngoing = True
    rewarded = False
    enteredMaze = False
    mistake = False
    hasLeft1 = False
    hasLeft2 = False
    e2Aftere1 = False
    e1Aftere2 = False

    ent1History = [False, False]
    ent2History = [False, False]

    while trialOngoing:
        # The line below needs to be commented out when running the actual experiments
        time.sleep(0.05)

        grayOriginal, valid = sf.grab_n_convert_frame(cameraHandle=cap)
        ret, gray = cv.threshold(grayOriginal, 180, 255, cv.THRESH_BINARY)

        if recordVideo:
            videoFile.write(grayOriginal)

        if not valid:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Display ROIs on top of the frame for user feedback
        for item in rois:
            cv.rectangle(gray, (rois[item]["xstart"], rois[item]["ystart"]),
                         (rois[item]["xstart"] + rois[item]["xlen"], rois[item]["ystart"] + rois[item]["ylen"]),
                         color=(0, 0, 0), thickness=2)

        # Display the resulting frame
        cv.imshow('frame', gray)

        if cv.waitKey(1) == ord('q'):
            break

        # Grab each area of interest and store them in a dictionary
        # This will be used to detect if the animal was there or not
        for item in rois:
            areas[item] = sf.grab_cut(gray,
                                       xstart=rois[item]["xstart"],
                                       ystart=rois[item]["ystart"],
                                       xlen=rois[item]["xlen"],
                                       ylen=rois[item]["ylen"])

            cv.imshow(item, areas[item])

            mousePresent[item] = np.sum(areas[item]) < thresholds[item] / 2

            if mousePresent[item]:
                hasVisited[item] = True

        if not ent1History[0] and ent1History[1]:
            endEnt1 = time.time()
            hasLeft1 = True
            if hasLeft2:
                e1Aftere2 = True
                e2Aftere1 = False

        if not ent2History[0] and ent2History[1]:
            hasLeft2 = True
            if hasLeft1:
                e1Aftere2 = False
                e2Aftere1 = True

        if e2Aftere1 and not enteredMaze:
            print("Mouse has entered the maze")
            trialStartTime = time.time()
            enteredMaze = True

        if e1Aftere2:
            print("Mouse has left the maze")
            trialOngoing = False
            enteredMaze = False
            if not rewarded:
                if not mistake:
                    miss.append(trial)
                if mistake:
                    incorrect.append(trial)

        if enteredMaze:
            for item in mousePresent:
                if habituation:
                    pass
                elif str(rewardTarget) in item and "entrance" not in item:
                    if mousePresent[item] and not rewarded:
                        if considerWrongLocations:
                            pass
                        else:
                            print(hasVisited)
                            print("Animal has reached reward zone")
                            rewarded = True
                            hits.append(trial)
                            areasRewarded.append((trial, rewardTarget))
                            time2Reward = time.time()

        end_trial_time = time.time()
        trial_duration = end_trial_time - start_trial_time  # Duration of the trial
        trial_durations.append(trial_duration)

# Record end time of the session
sessionDuration = time.time() - sessionStartTime

# When everything is done, release the capture
cap.release()
cv.destroyAllWindows()

if recordVideo:
    # videoFile.release()
    pass
