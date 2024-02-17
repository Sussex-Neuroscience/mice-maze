import numpy as np
import cv2 as cv
import pandas as pd
import supFun as sf
import time
import serial
import os
import pandas as pd
from tkinter import filedialog as fd
from tkinter import *


#line 16, line 130, 133, 197, 241, 274, 339
#line 375

#set if the video should be recorded during the session
recordVideo = False
#set filename for video recording

# get the current date and time
local_time = time.localtime()

# format the date and time as YYYY-MM-DD_hh-mm-ss
date_time= str(time.strftime('%d %b %Y %H:%M:%S', local_time))


# append the date and time to the file name
recVideoName = "test_" + date_time + ".mp4"


#set variable for defining if animal should be rewarded in the case
#a wrong location is visited before visiting a correct location:
considerWrongLocations = False


#create a serial object and connect to it
ser = serial.Serial("/dev/ttyACM0")


drawRois = False
if drawRois:

    sf.define_rois(roiNames = ["entrance1","entrance2",
                            "rewA","rewB","rewC","rewD"],
                    outputName = "rois1.csv")

#load ROI information
rois = pd.read_csv("rois1.csv",index_col=0)


sessionStage = 1
##### begin block #######

### this needs to be optimised so we don't have to create, write and read every time

#create  trials and save them to csv (later this csv needs to go to the appropriate session folder)
trials = sf.create_trials(numTrials = 100, sessionStage=sessionStage)
trials.to_csv("trials.csv")

#load the trials file (description of each trial)
#trialsIDs = pd.read_csv(sf.choose_csv())
#trialsIDs = trials
###### end block ##############


#set number of trials
nTrials = len(trials)

#get the identification of each grating
gratingID = pd.read_csv("grating_maps.csv",index_col=0)
#gratingID = [match for match in list(trialsIDs.keys()) if "motor" in match]


#the maze only uses servo motors. To simplify their control, we are relying on
# an adafruit board (based on the PCA9685 IC), which allows us to flexibly
#increase/decrease the number of motors, without using more ports on a microcontroller,
#since the Adafruit board communicates with a microcontroller using two digital lines, but is
#able to control more than 100 servo motors.

#The board also has C++ and micropython libraries, which makes it easy for users to control them.


# set the port/address for each motor
gratingMotors = {"L":0,"R":1,
                 "LR":2,"RL":3,
                 "LL":4,"RR":5,                 
                 }

rewardMotors = {"A":8,"B":9,
                "C":10,"D":11}




thresholds = dict()
#timeSpentAreas = dict()
for item in rois:
    #create threshold values for each area
    thresholds[item] = 0
    #create all roi variables in the dictionary and attach an empty list to them
    #timeSpentAreas[item] = np.zeros(nTrials)


#create variables that will store session data
areasRewarded = list()
hits = list()
miss = list()
incorrect = list()

#preload variables
hasVisited = dict()
mousePresent = dict()




#start the camera object
#cap = sf.start_camera()
cap = cv.VideoCapture(0)

### START FOR LOOP TO AVERAGE N FRAMES FOR THRESHOLDING

#grab one frame to adjust tresholds of empty spaces:
gray,valid = sf.grab_n_convert_frame(cameraHandle=cap)
ret,gray = cv.threshold(gray,180,255,cv.THRESH_BINARY)
#gray = gray[:,:,0]
#run a loop to catch each area and sum the pixel values on that area of the frame
areas = dict()
for item in rois:
    areas[item] = sf.grab_cut(gray,
                xstart = rois[item]["xstart"],
                ystart =  rois[item]["ystart"],
                xlen = rois[item]["xlen"],
                ylen =  rois[item]["ylen"],
                )
    
    thresholds[item] = np.sum(areas[item])
cap.release()
#cv.destroyAllWindows()
#print(thresholds)

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


cap = sf.start_camera()
#cap = cv.VideoCapture(filePath)



sessionStartTime = time.time()

if recordVideo:
    frame_width = int(cap.get(3)) # can also use cam.get (cv.CAP_PROP_FRAME_WIDTH)
    frame_height = int(cap.get(4)) # can also use cam.get (cv.CAP_PROP_FRAME_HEIGHT)
    fps = int(cap.get(5)) #can also use cam.get (cv.CAP_PROP_FPS)

    cc = cv.VideoWriter_fourcc(*'XVID') #Can also use mp4v or MJPG for .avi, must check with maze camera
    frameSize = (frame_width,frame_height)

    videoFile = cv.VideoWriter(recVideoName, cc, fps, frameSize)
    
#trial_durations = [] #list that will contain the durations of every trial. Maybe this can be stored in another csv file?

rewLocation=0
usedTrial = list()

#create a dataframe filled with nans that will later be updated with data from session
data = sf.empty_frame(rows = len(trials.index))

for trial in trials.index:
    #remove trials that are the exact same as the previous one
    
    if trials.loc[trial].rewlocation==rewLocation and sessionStage==3:
        print (rewLocation + " " + trials.loc[trial].rewlocation) 
        usedTrial.append(False)
    else:
        usedTrial.append(True)
        rewLocation = trials.loc[trial].rewlocation
            

        # time at the beginning of the trial 
        start_trial_time = time.time()
        print ("starting trial "+str(trial+1))
        if cv.waitKey(1) == ord('q'):
            break

            
        
        #move all gratings to their neutral positions
        print("set all grating motors to neutral position")
        for motor in gratingMotors:
            message = 'grt'+motor+' 45'
            ser.write(message.encode('UTF-8')) 
            #maybe add a pause? so that the servo motors have time to catch up
            
        
        
        #if trials.givereward[trial]: - don't know if this is necessary
            #maybe it is ok for motors to be placed in position even though
            #there will be no reward?
        for grtPosition in gratingID.loc[trials.loc[trial].rewlocation]:       
            print(grtPosition)
            message = 'grt'+grtPosition
            ser.write(message.encode('UTF-8'))
            
                
            #maybe add a pause? so that the servo motors have time to catch up
       
        for item in rois:
            hasVisited[item] = False 
        
        #ent1 = 0
        #ent2 = 0
        trialOngoing = True
        rewarded = False
        enteredMaze = False
        mistake = False
        hasLeft1 = False
        hasLeft2 = False
        e2Aftere1 = False
        e1Aftere2 = False
        
        
        ent1History = [False,False]
        ent2History = [False,False]
        while trialOngoing:
            
            #the line below needs to be commented out when running the actual experiments
            time.sleep(0.05)
            
            grayOriginal,valid = sf.grab_n_convert_frame(cameraHandle=cap)
            ret,gray = cv.threshold(grayOriginal,180,255,cv.THRESH_BINARY)

            if recordVideo:
                videoFile.write(grayOriginal)  
            
            
            #binGray = gray[:,:,2]
            
            if not valid:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            
            #display rois on top of the frame for user feedback
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
                
                if mousePresent[item]:
                    hasVisited[item] = True
                    logTime = time.time()
                    #timeSpentAreas[item] = time.time()
                #print(item+" "+str(np.sum(areas[item])))
            
        
            #at each new frame, add the information on whether the mouse
            #was in the Ent1 area to a two element list
            #every new added item goes in front of the list, and the last
            #element is removed.
            ent1History.insert(0,mousePresent["entrance1"])
            ent1History.pop(-1)
            #print(ent1History)
            
            #at each new frame, add the information on whether the mouse
            #was in the Ent2 area to a two element list
            #every new added item goes in front of the list, and the last
            #element is removed.
            ent2History.insert(0,mousePresent["entrance2"])
            ent2History.pop(-1)

            
                
            #calculate the mouse movement based on the Ent1 array history
            # and determine if it has passed Ent1 and/or 2 and in which direction
            if not ent1History[0] and ent1History[1]:
                #print("mouse just left entrance1")
                endEnt1 = time.time()
                #timeSpentAreas["entrance1"].append(trial,endEnt1-startEnt1)
                #Ent1Durations.append((trial,endEnt1-startEnt1))
                hasLeft1 = True
                if hasLeft2:
                    e1Aftere2 = True
                    e2Aftere1 = False
                    
                
            #calculate the mouse movement based on the Ent2 array history
            if not ent2History[0] and ent2History[1]:
                #print("mouse just left entrance2")
                hasLeft2 = True
                if hasLeft1:
                    e1Aftere2 = False
                    e2Aftere1 = True
                    

            
            if e2Aftere1 and enteredMaze==False:
                print("mouse has entered the maze")
                trialStartTime = time.time()
                enteredMaze = True
            if e1Aftere2:
                print("mouse has left the maze")
                trialOngoing=False
                #trialDurations.append((trial,time.time()-trialStartTime))
                enteredMaze = False
                if not rewarded:
                    if not mistake:
                        miss.append(trial)
                    if mistake:
                        incorrect.append(trial)
                


                

                
            if enteredMaze:
                for item in mousePresent:
                    
                    if not trials.givereward[trial]:#this is for habituation routine
                        pass
                    
                    #if trials.loc[trial].givereward:
                    elif trials.rewlocation[trial] in item:
                        if mousePresent[item] and not rewarded:
                            #if the animal is allowed to visit a wrong location first
                            if trials.wrongallowed[trial]:
                                message = "rew"+trials.rewlocation[trial]
                                ser.write(message.encode('UTF-8'))
                                print(hasVisited)
                                print("animal has reached reward zone")
                                rewarded = True
                                hits.append(trial)
                                #store the reward area
                                areasRewarded.append((trial,trials.loc[trial].rewlocation))
                                #TODO
                                time2Reward = time.time()
                                #do calculations on time to reward
                                #add code to start proper reward motor
                            
                            #this if statement is a placeholder for the training phase where
                            #animals cannot visit a wrong location before visiting the
                            #target location
                            else:
                                for rewLoc in hasVisited:
                                    if hasVisited[rewLoc] and trials.rewlocation[trial] not in rewLoc:
                                        mistake=True
        

                            

                
                #if trialEnd==1:
                #    trialOngoing = False

        #record end time of the trial
        end_trial_time=  time.time()
        trial_duration= end_trial_time - start_trial_time   #duration of the trial
        trial_durations.append(trial_duration)

        #create csv file that puts the durations per trial + the whole sesh duration
            
            
            
sessionDuration = time.time()-sessionStartTime




# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
if recordVideo:
    #videoFile.release()
    pass

