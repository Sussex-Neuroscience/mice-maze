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



#define where the video is coming from. Use 0 for the first camera on the computer,
#or a complete file path to use a pre-recorded video
videoInput = '/home/andre/Desktop/maze_test.mp4'



#get the identification of each grating
gratingID = pd.read_csv("grating_maps.csv",index_col=0)
#gratingID = [match for match in list(trialsIDs.keys()) if "motor" in match]




testing = True
if testing:
    recordVideo = False    
    new_dir_path = "/home/andre/Desktop/maze_recordings/"
    sessionStage = 2
    #create  trials and save them to csv (later this csv needs to go to the appropriate session folder)
    trials = sf.create_trials(numTrials = 100, sessionStage=sessionStage)

    trials.to_csv(os.path.join(new_dir_path,"trials_before_session.csv"))

    #load the trials file (description of each trial)
    print("choose the file containing trials (default: 'trials_before_session.csv'")
    trialsIDs = trials
else:
    recordVideo = True

    date_time = sf.get_current_time_formatted()
    animal_ID, session_ID = sf.get_user_inputs()
    base_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'maze_recordings')
    sf.ensure_directory_exists(base_path)

    new_dir_path = sf.setup_directories(base_path, date_time, animal_ID, session_ID)
    rec_name = f"{animal_ID}_{date_time}.avi"
    recordings = os.path.join(new_dir_path, rec_name)
    print(f"Video will be saved to: {recordings}")

    metadata = sf.collect_metadata(animal_ID, session_ID)
    sf.save_metadata_to_csv(metadata, new_dir_path, f"{animal_ID}_{date_time}.csv")
    trials = sf.create_trials(numTrials = 100, sessionStage=sessionStage)

    trials.to_csv(os.path.join(new_dir_path,"trials_before_session.csv"))

    #load the trials file (description of each trial)
    print("choose the file containing trials (default: 'trials_before_session.csv'")
    trialsIDs = trialspd.read_csv(sf.choose_csv())

#create a serial object and connect to it
ser = serial.Serial("/dev/ttyUSB0",115200)
print(ser.in_waiting)
while ser.in_waiting>0:
    ser.readline()
ser.flush()

drawRois = False
if drawRois:
    sf.define_rois(videoInput = videoInput,
                    roiNames = ["entrance1","entrance2",
                            "rewA","rewB","rewC","rewD"],
                    outputName = new_dir_path+"rois1.csv")

#load ROI information
rois = pd.read_csv(new_dir_path+"rois1.csv",index_col=0)


##### begin block #######

### this needs to be optimised so we don't have to create, write and read every time

thresholds = dict()
#timeSpentAreas = dict()
for item in rois:
    #create threshold values for each area
    thresholds[item] = 0



#create variables that will store session data
areasRewarded = list()
hits = list()
miss = list()
incorrect = list()

#preload variables
hasVisited = dict()
mousePresent = dict()



cap = sf.start_camera(videoInput=videoInput)

frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv.CAP_PROP_FPS))

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
#cap.release()
#cv.destroyAllWindows()
#print(thresholds)


sessionStartTime = time.time()

if recordVideo:
    sf.record_video(cap, recordings, frame_width, frame_height, fps)


rewLocation=0
usedTrial = list()

#create a dataframe filled with nans that will later be updated with data from session
data = sf.empty_frame(rows = len(trials.index))

for trial in trials.index:
    #remove trials that are the exact same as the previous one
    if trials.rewlocation[trial]==rewLocation and sessionStage==3:
        #print (rewLocation + " " + trials.rewlocation[trial]) 
        usedTrial.append(False)
    else:
        usedTrial.append(True)
        rewLocation = trials.rewlocation[trial]

    # time at the beginning of the trial 
    start_trial_time = time.time()
    print("starting trial "+str(trial+1))
    print("trial profile:")
    print("rew Location: "+trials.rewlocation[trial])
    print("give reward: "+str(trials.givereward[trial]))
    print("wrong allowed: "+str(trials.wrongallowed[trial]))
    

    #move all gratings to their neutral positions
    print("set all grating motors to neutral position")
    for motor in gratingID:
        motor1 = motor[motor.find(" ")+1:]
        message = 'grt{0} 45\n'.format(motor1)
        #print(message)
        ser.write(message.encode('utf-8'))
        time.sleep(0.1)
        #maybe add a pause? so that the servo motors have time to catch up


    #if trials.givereward[trial]: - don't know if this is necessary
        #maybe it is ok for motors to be placed in position even though
        #there will be no reward?
    print("now move motors to cueing positions for this trial")
    for grtPosition in gratingID.loc[trials.rewlocation[trial]]:       
        #print(grtPosition)
        message = 'grt{0}\n'.format(grtPosition)
        #print (message)
        ser.write(message.encode('utf-8'))    
    #maybe add after all commands have been sent to motors?


    # time at the beginning of the trial 
    start_trial_time = time.time()
    print ("starting trial "+str(trial+1))
    if cv.waitKey(1) & 0xFF in [ord('q'), 27]:  # Quit on 'q' or ESC
        break

            

       
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
        #binGray = gray[:,:,2]
        
        if not valid:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        if recordVideo:
            recordings.write(grayOriginal)  
        
        
        
        
        #display rois on top of the frame for user feedback
        for item in rois:
            cv.rectangle(gray, (rois[item]["xstart"],
                                rois[item]["ystart"]),
                               (rois[item]["xstart"]+rois[item]["xlen"],
                                rois[item]["ystart"]+rois[item]["ylen"]),
                           color=(0,0,0), thickness=2)

        # Display the resulting frame
        cv.imshow('frame', gray)
        
        if cv.waitKey(1) & 0xFF in [ord('q'), 27]:  # Quit on 'q' or ESC
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
                #timeSpentAreas[item] = time.time()
            #print(item+" "+str(np.sum(areas[item])))
        
            #here we define the logic for this training step
            #this step only cares if the animal reaches the correct area
            #no matter if it entered a wrong area before
            #stillOnEnt1 = 0
            #stillOnEnt2 = 0
            #startEnt1 = time.time()
        
    
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
                        
                        #if the animal is not allowed to visit a wrong location first
                        if not trials.wrongallowed[trial]:
                            for rewLoc in hasVisited:
                                if hasVisited[rewLoc] and trials.rewlocation[trial] not in rewLoc:
                                    mistake=True
                        
                        #if the animal is in the right reward zone and there was no mistake
                        if not mistake:
                            message = 'rew{0}\n'.format(trials.rewlocation[trial])
                            ser.write(message.encode('utf-8'))
                            #print(hasVisited)
                            print("animal has reached reward zone")
                            rewarded = True
                            hits.append(trial)
                            #store the reward area
                            areasRewarded.append((trial,trials.loc[trial].rewlocation))
                            #TODO
                            time2Reward = time.time()
                            #do calculations on time to reward
                            #add code to start proper reward motor
                        
                        
    

                        

            
            #if trialEnd==1:
            #    trialOngoing = False

    #record end time of the trial
    #end_trial_time=  time.time()
    #trial_durations= end_trial_time - start_trial_time   #duration of the trial
    #trial_durations.append(trial_durations)

    
            
            
            
sessionDuration = time.time()-sessionStartTime




# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
if recordVideo:
    #videoFile.release()
    pass

