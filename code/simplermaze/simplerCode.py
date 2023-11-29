import numpy as np
import cv2 as cv
import pandas as pd
import supFun as sf
import time
import serial


#create a serial object and connect to it
#ser = Serial.serial("/dev/ttyACM0")

drawRois = 0
if drawRois==1:

    sf.define_rois(roiNames = ["entrance1","entrance2",
                            "rew1","rew2","rew3","rew4"],
                    outputName = "rois1.csv")

#load ROI information
rois = pd.read_csv("rois1.csv",index_col=0)

#load the trials file (description of each trial)
trialsIDs = pd.read_csv("trials_2columns.csv")

#set number of trials
nTrials = len(trialsIDs)

#get the identification of each grating
gratingID = [match for match in list(trialsIDs.keys()) if "motor" in match]


#get the number of reward areas
nRewAreas = trialsIDs.ra1.max()


#the maze only uses servo motors. To simplify their control, we are relying on
# an adafruit board (based on the PCA9685 IC), which allows us to flexibly
#increase/decrease the number of motors, without using more ports on a microcontroller,
#since the Adafruit board communicates with a microcontroller using two digital lines, but is
#able to control more than 100 servo motors.

#The board also has C++ and micropython libraries, which makes it easy for users to control them.


# set the port/address for each motor
motor1Grating = 1
motor2Grating = 2
motor3Grating = 3
motor4Grating = 4
motor5Grating = 5
motor6Grating = 6

motor1Reward = 7
motor2Reward = 8
motor3Reward = 9
motor4Reward = 10



thresholds = dict()
timeSpentAreas = dict()
for item in rois:
    #create threshold values for each area
    thresholds[item] = 0
    #create all roi variables in the dictionary and attach an empty list to them
    timeSpentAreas[item] = []


#create variables that will store session data
 

areasRewarded = list()

hits = 0
miss = 0
incorrect = 0

#preload variables
areas = dict()
enteredMaze=False
#set variable for defining if animal should be rewarded in the case
#a wrong location is visited before visiting a correct location:
considereWrongLocations = False


#start the camera object
cap = cv.VideoCapture('/home/andre/Desktop/maze_test.mp4')

#grab one frame to adjust tresholds of empty spaces:
gray,valid = sf.grab_n_convert_frame(cameraHandle=cap)
ret,gray = cv.threshold(gray,180,255,cv.THRESH_BINARY)
#gray = gray[:,:,0]
#run a loop to catch each area and sum the pixel values on that area of the frame
for item in rois:
    areas[item] = sf.grab_cut(gray,
                xstart = rois[item]["xstart"],
                ystart =  rois[item]["ystart"],
                xlen = rois[item]["xlen"],
                ylen =  rois[item]["ylen"],
                )
    
    thresholds[item] = np.sum(areas[item])

print(thresholds)

#cap = sf.start_camera()
#cap = cv.VideoCapture("~/Desktop/maze_test.mp4")



sessionStartTime = time.time()

for trial in range(nTrials):
    #if cv.waitKey(1) == ord('q'):
    #    break
    #which area should be rewarded:
    rewardTarget = trialsIDs["ra1"][trial]
    #define which servo motor should be used for this specific area
    #(eventually find a better way to code this with a loop or something)
    if rewardTarget == 1:
        rewMotor = motor1Reward
    elif rewardTarget == 2:
        rewMotor = motor2Reward
    elif rewardTarget == 3:
        rewMotor = motor3Reward
    elif rewardTarget == 4:
        rewMotor = motor4Reward
    else:
        rewMotor = 1
        print("correct reward motor not found!! defaulting to motor 1")
        
    
    #run code to start the gratings
    for grating in gratingID:
        print(grating)
        #add arduino code here so that all gratings turn to neutral position"
        #ser.print(xxxxxxxx)
    
        #maybe add a pause? so that the servo motors have time to catch up
        
    for grating in gratingID:
        position = trialsIDs[grating][trial]
        print(position)
        #than add code so that they turn to the trial specific position"
        #maybe add a pause? so that the servo motors have time to catch up
        
    
    #gate1 = 0
    #gate2 = 0
    trialOngoing = 1
    rewarded = False
    enteredMaze = False
    left1 = 0
    left2 = 0
    g2Afterg1 = False
    g1Afterg2 = False
    hasVisited = dict()
    mousePresent = dict()
    gate1History = [False,False]
    gate2History = [False,False]
    while trialOngoing == 1:
        
        time.sleep(0.05)
        
        gray,valid = sf.grab_n_convert_frame(cameraHandle=cap)
        ret,gray = cv.threshold(gray,180,255,cv.THRESH_BINARY)
        binGray = gray[:,:,2]
        
        if not valid:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        for item in rois:
            cv.rectangle(gray, (rois[item]["xstart"],
                                rois[item]["ystart"]),
                               (rois[item]["xstart"]+rois[item]["xlen"],
                                rois[item]["ystart"]+rois[item]["ylen"]),
                           color=(0,0,0), thickness=2)

        # Display the resulting frame
        cv.imshow('frame', gray)
        
        if cv.waitKey(1) == ord('q'):
            break
        
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
            #print(item+" "+str(np.sum(areas[item])))
        
        #here we define the logic for this training step
        #this step only cares if the animal reaches the correct area
        #no matter if it entered a wrong area before
        #stillOnGate1 = 0
        #stillOnGate2 = 0
        #startGate1 = time.time()
        
    
        #at each new frame, add the information on whether the mouse
        #was in the gate1 area to a two element list
        #every new added item goes in front of the list, and the last
        #element is removed.
        gate1History.insert(0,mousePresent["entrance1"])
        gate1History.pop(-1)
        print(gate1History)
        #at each new frame, add the information on whether the mouse
        #was in the gate2 area to a two element list
        #every new added item goes in front of the list, and the last
        #element is removed.
        gate2History.insert(0,mousePresent["entrance2"])
        gate2History.pop(-1)

        
            
        #calculate the mouse movement based on the gate1 array history
        # and determine if it has passes gate1 and/or 2 and in which direction
        if gate1History[1] and not gate1History[0]:
            print("mouse just left entrance1")
            endGate1 = time.time()
            #timeSpentAreas["entrance1"].append(trial,endGate1-startGate1)
            #gate1Durations.append((trial,endGate1-startGate1))
            left1 = True
            if left2:
                g1Afterg2 = True
                g2Afterg1 = False
                
            
#         if gate1History[0] and not gate1History[1]:
#             print("mouse just entered entrance 1")
#             
#         if  gate1History[0] and gate1History[1]:
#             print("mouse still on entrance 1 area")
        
        #calculate the mouse movement based on the gate2 array history
        if gate2History[1] and not gate2History[0]:
            print("mouse just left entrance2")
            left2 = True
            if left1:
                g1Afterg2 = False
                g2Afterg1 = True
                

        
        if g2Afterg1 and enteredMaze==False:
            print("mouse has entered the maze")
            trialStartTime = time.time()
            enteredMaze = True
        if g1Afterg2:
            print("mouse has left the maze")
            trialOngoing=0
            #trialDurations.append((trial,time.time()-trialStartTime))
            enteredMaze = False


            

            
        if enteredMaze:         
            for item in mousePresent:
                if mousePresent[item]:
                    pass
                    #hasVisited[item]==True
                    
                if str(rewardTarget) in item and "entrance" not in item:
                    if mousePresent[item] and not rewarded:
                        #this if statement is a placeholder for the training phase where
                        #animals cannot visit a wrong location before visiting the
                        #target location
                        if considerWrongLocations:
                            pass
                        
                        else:
                            print("animal has reached reward zone")
                            rewarded = True
                            #store the reward area
                            areasRewarded.append((trial,rewardTarget))
                            #TODO
                            #do calculations on time to reward
                            #add code to start proper reward motor
            
            #if trialEnd==1:
            #    trialOngoing = 0
        
        
        
        
sessionDuration = time.time()-sessionStarTime    
    

    
    #add some timing metric so we have info on how much time each trial took


    
    




# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

