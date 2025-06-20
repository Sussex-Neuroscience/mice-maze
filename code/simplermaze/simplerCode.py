<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
#import all needed libraries
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
import csv
import copy

#for code inspection and testing of the code purposes we add a small pause in between frames in
#the main code loop... this variable just below this needs to be set to False if one is running the actual experiments
pause_between_frames = False

<<<<<<< Updated upstream
#whenever working without the actual servos and ESP32 set the next variable to False
serialOn = True

#if running experiments "testing" should be False (related to testing the code)
testing = False
#If ROIs need to be drawn 
# by experiementer, set the next variable to TRUE
drawRois = False

#If just testing and no video needs to be recorded, set the next variable to FALSE
recordVideo = True
#define where the video is coming from. Use 0 for the first camera on the computer,
#or a complete file path to use a pre-recorded video
videoInput = 0
=======

#define where the video is coming from. Use 0 for the first camera on the computer,
#or a complete file path to use a pre-recorded video
videoInput = 0#'/home/andre/Desktop/maze_test.mp4'

>>>>>>> Stashed changes


#get the identification of each grating
gratingID = pd.read_csv("grating_maps.csv",index_col=0)
#gratingID = [match for match in list(trialsIDs.keys()) if "motor" in match]


<<<<<<< Updated upstream
#get the current date and time, so all files created do not overwrite existing data
date_time = sf.get_current_time_formatted()

if testing:
    base_path = "C:/Users/labadmin/Desktop/maze_recordings/"
    new_dir_path = "C:/Users/labadmin/Desktop/maze_recordings/"
    #new_dir_path = "C:/Users/labadmin/Desktop/maze_recordings/"
    experiment_phase = 3
    #create  trials and save them to csv (later this csv needs to go to the appropriate session folder)
    trials = sf.create_trials(numTrials = 100, sessionStage=experiment_phase, nonRepeat=True)
    trials.to_csv(os.path.join(new_dir_path,f"trials_before_session_{date_time}.csv"))
    recordFile = os.path.join(new_dir_path, f"test_{date_time}.mp4")
    #load the trials file (description of each trial)
    print("choose the file containing trials (default: 'trials_before_session.csv'")
    trialsIDs = trials
else:
    animal_ID, session_ID, experiment_phase = sf.get_user_inputs()
    base_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'maze_recordings')
    sf.ensure_directory_exists(base_path)
    
    new_dir_path = sf.setup_directories(base_path, date_time,animal_ID,session_ID)
    rec_name = f"{animal_ID}_{date_time}.mp4"
    recordFile = os.path.join(new_dir_path, rec_name)
    print(f"Video will be saved to: {recordFile}")

    metadata = sf.collect_metadata(animal_ID, session_ID)
    sf.save_metadata_to_csv(metadata, new_dir_path, f"{animal_ID}_{date_time}.csv")
    trials = sf.create_trials(numTrials = 150, sessionStage=experiment_phase, nonRepeat=True)

    trials.to_csv(os.path.join(new_dir_path,"trials_before_session.csv"))

    #load the trials file (description of each trial)
    print("choose the file containing trials (default: 'trials_before_session.csv'")
    trialsIDs = pd.read_csv(sf.choose_csv())



if serialOn:
    #create a serial object and connect to it

    ser = serial.Serial("COM4",115200)
=======
    trials.to_csv(os.path.join(new_dir_path,"trials_before_session.csv"))
    recordings = os.path.join(new_dir_path, "test.mp4")
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
    trialsIDs = pd.read_csv(sf.choose_csv())

if serialOn:
    #create a serial object and connect to it

    ser = serial.Serial("/dev/ttyUSB0",115200)
>>>>>>> Stashed changes
    print(ser.in_waiting)
    while ser.in_waiting>0:
        ser.readline()
    ser.flush()
<<<<<<< Updated upstream
=======

drawRois = True
if drawRois:
    sf.define_rois(videoInput = videoInput,
                    roiNames = ["entrance1","entrance2",
                            "rewA","rewB","rewC","rewD"],
                    outputName = new_dir_path+"rois1.csv")
    
    rois = pd.read_csv(new_dir_path+"rois1.csv",index_col=0)
else:
    rois = pd.read_csv("rois1.csv",index_col=0)
#load ROI information
>>>>>>> Stashed changes


if drawRois:
    sf.define_rois(videoInput = videoInput,
                    roiNames = ["entrance1","entrance2",
                            "rewA","rewB","rewC","rewD"],
                    outputName = base_path+"/"+"rois1.csv")
    rois = pd.read_csv(base_path+ "/"+"rois1.csv",index_col=0)
else:
    rois = pd.read_csv(base_path+"/"+ "rois1.csv",index_col=0)
    #rois_save = rois[:]
    #rois_save.to_csv(new_dir_path+"/"+"rois1.csv")
#load ROI information

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

<<<<<<< Updated upstream
=======

cap = sf.start_camera(videoInput=videoInput)

>>>>>>> Stashed changes
frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv.CAP_PROP_FPS))
if recordVideo:
    videoFileObject = sf.record_video(cap, recordFile, frame_width, frame_height, fps)


<<<<<<< Updated upstream
#grab one frame:
valid,gray = cap.read()
#quick and dirty method to grab more frames from the camera to see if there are any issues with
#stabilization in the beginning:
for i in range(5):
    valid,gray = cap.read()
    time.sleep(0.1)
ret,gray = cv.threshold(gray,100,255,cv.THRESH_BINARY)

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


sessionStartTime = time.time()

<<<<<<< Updated upstream
=======
if recordVideo:
    sf.record_video(cap, recordings, frame_width, frame_height, fps)


rewLocation=0
useTrial = list()
if sessionStage==3:
    for trial in trials.index:
        #remove trials that are the exact same as the previous one
        if trials.rewlocation[trial]==rewLocation:
            #print (rewLocation + " " + trials.rewlocation[trial]) 
            useTrial.append(False)
        else:
            useTrial.append(True)
            rewLocation = trials.rewlocation[trial]
        rewLocation=trials.rewlocation[trial]
else:
    useTrial = [True]*len(trials.index)

>>>>>>> Stashed changes
#create a dataframe filled with nans that will later be updated with data from session
data = sf.empty_frame(rows = len(trials.index))

#write header of panda dataframe
sf.write_data(file_name=os.path.join(new_dir_path,f"session_data_{date_time}.csv"),
              mode="w",
              data=data.head(n=0))

back_sub = cv.createBackgroundSubtractorMOG2()
#create two windows to show the animal movement while in maze:
cv.namedWindow('binary maze plus ROIs', cv.WINDOW_NORMAL)
cv.namedWindow('original image', cv.WINDOW_NORMAL)
cv.namedWindow('frame_diff', cv.WINDOW_NORMAL)
absolute_time_start = sf.time_in_millis()
 
 


for trial in trials.index:
<<<<<<< Updated upstream
    if cv.waitKey(1) & 0xFF in [ord('q'), 27]:  # Quit on 'q' or ESC
        break
    # time at the beginning of the trial 
    trial_start_time = sf.time_in_millis()-absolute_time_start
    if trial == 0:
        time_old_frame = sf.time_in_millis()-trial_start_time
        

    data.loc[trial,"trial_start_time"] = trial_start_time
    data.loc[trial,"rew_location"] = trials.rewlocation[trial]
    #print information for users
    print("preparing maze"+str(trial+1))
    print("trial profile:")
    print("rew Location "+trials.rewlocation[trial])
    print("give reward "+str(trials.givereward[trial]))
    print("wrong allowed "+str(trials.wrongallowed[trial]))
    

    #move all gratings to their neutral position
    print("set all grating motors to neutral position")
    for motor in gratingID:
        motor1 = motor[motor.find(" ")+1:]
        message = 'grt{0} 0\n'.format(motor1)
        #print(message)
        if serialOn:
            ser.write(message.encode('utf-8'))
            ser.flush()           
        

    print("now move motors to cueing positions for this trial")
    
    for grtPosition in gratingID.loc[trials.rewlocation[trial]]:
        if experiment_phase!=1:
            message = 'grt{0}\n'.format(grtPosition)
            #print (message)
        else:
            message = 'grt{0} 20\n'.format(grtPosition[0:2])

        if serialOn:
            ser.write(message.encode('utf-8'))
            ser.flush()   
    

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
    
    visited_any_rew_area_flag = False
    first_rew_area = "X"
    
    #quick and dirty method to grab more frames from the camera to see if there are any issues with
    #stabilization in the beginning:
    if trial == 0:

        for i in range(5):
            valid,gray = cap.read()
            time.sleep(0.1)
    
    while trialOngoing:
        
        if pause_between_frames:
            time.sleep(0.05)
        
        valid,grayOriginal = cap.read()
        if not valid:
            print("Failed to read frame")
            continue
        
        ## play around with the value after grayOriginal to set the threshold for the pixels

        ret,gray = cv.threshold(grayOriginal,160,255,cv.THRESH_BINARY)
        fg_mask = back_sub.apply(gray)
        # cv.imshow('Foreground Mask', fg_mask)
        # cv.waitKey(0)
        #ret,gray_inv = cv.threshold(grayOriginal,180,255,cv.THRESH_BINARY_INV)
        
        try:
            contours,hierarchy = cv.findContours(fg_mask, cv.RETR_TREE,
                                          cv.CHAIN_APPROX_NONE)
            if contours:
                cv.drawContours(fg_mask, contours[0], -1, 255, 4)
                cv.drawContours(gray, contours[0], 0, (0,255,255), 2)
            else:
                print("No contours found")
        except:
            print("no")
        #cv.drawContours(gray,contours[0],0,(0,255,255),2)
        #binGray = gray[:,:,2]
        time_frame=sf.time_in_millis()-absolute_time_start
        #if 'old_frame' in locals():
        #    frame_diff = gray-old_frame
        #    cv.imshow('frame_diff', frame_diff)
        
        if not valid:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        if recordVideo:
            videoFileObject.write(grayOriginal)

        
        #display rois on top of the frame for user feedback
        for item in rois:
            cv.rectangle(gray, (rois[item]["xstart"],
                                rois[item]["ystart"]),
                               (rois[item]["xstart"]+rois[item]["xlen"],
                                rois[item]["ystart"]+rois[item]["ylen"]),
                           color=(120,0,0), thickness=2)
                        # Add the name of the ROI above the rectangle
            cv.putText(gray, item,
               (rois[item]["xstart"], rois[item]["ystart"] - 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv.LINE_AA)

        # Display the resulting frame
        cv.imshow('binary maze plus ROIs', gray)
        cv.imshow('original image', grayOriginal)
        #
        #contours, hierarchy = cv.findContours(fg_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
       # cv.imshow('frame_diff', fg_mask) 
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
            
            #cv.imshow(item,areas[item])
            #print(item+" " + str(np.sum(areas[item])))
    
            mousePresent[item] = np.sum(areas[item])<thresholds[item]/2
            
            if mousePresent[item]:
                print(item)
                #store what was the first reward area visited by the animal
                if "rew" in item and not visited_any_rew_area_flag:
                    visited_any_rew_area_flag=True
                    first_rew_area = item
                    data.loc[trial,"first_reward_area_visited"] = first_rew_area        
                    if trials.rewlocation[trial] not in first_rew_area:
                        data.loc[trial,"incorrect"] = 1
                        print("early mistake detection")
                        mistake=True
                        #print("first visit to:",item)
                
                hasVisited[item] = True

                duration=time_frame-time_old_frame
                #print(duration)
                if np.isnan(data.loc[trial,item]):
                    data.loc[trial,item] = duration
                else:
                    data.loc[trial,item] = data[item][trial]+duration


            else:
                hasVisited[item]= False 

            
                
        #time_old_frame=time_frame
        
    
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
=======
    if useTrial[trial]:
        # time at the beginning of the trial 
        start_trial_time = time.time()
        #print information for users
        print("starting trial "+str(trial+1))
        print("trial profile:")
        print("rew Location "+trials.rewlocation[trial])
        print("give reward "+str(trials.givereward[trial]))
        print("wrong allowed "+str(trials.wrongallowecd[trial]))
        

        #move all gratings to their neutral positions
        print("set all grating motors to neutral position")
        for motor in gratingID:
            motor1 = motor[motor.find(" ")+1:]
            message = 'grt{0} 45\n'.format(motor1)
            #print(message)
            if serialOn:
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
            if serialOn:
                ser.write(message.encode('utf-8'))
                time.sleep(0.1)


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
>>>>>>> Stashed changes

            if recordVideo:
                recordings.write(grayOriginal)
                
<<<<<<< Updated upstream
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
            print("trial starting")
            data.loc[trial,"mouse_enter_time"]=time_frame
            enteredMaze = True
        if e1Aftere2:
            print("mouse has left the maze")
            trialOngoing=False
            data.loc[trial,"end_trial_time"]=time_frame
            #trialDurations.append((trial,time.time()-trialStartTime))
            enteredMaze = False
            if not rewarded:
                if not mistake:
                    data.loc[trial,"miss"] = 1
                    #data["miss"][trial] = 1
                if mistake:
                    data.loc[trial,"incorrect"] = 1
                    #data["incorrect"][trial] = 1
            
                
        if enteredMaze:
            for item in mousePresent:
                
                if not trials.givereward[trial]:#this is for habituation routine
                    pass
                
                    
                #if trials.loc[trial].givereward:
                elif trials.rewlocation[trial] in item:
                    if mousePresent[item] and not rewarded:                         
                        
                        #if the animal is not allowed to visit a wrong location first
                        #if not trials.wrongallowed[trial]:
                        #    if trials.rewlocation[trial] not in first_rew_area:
                        #        mistake=True
                        #       #data.loc[trial,"incorrect"] = 1
                        #        print("animal made a mistake")
                        
                        #if the animal is in the right reward zone and there was no mistake
                        if not mistake:
                            data.loc[trial,"hit"] = 1
                            message = 'rew{0}\n'.format(trials.rewlocation[trial])
                            if serialOn:
                                ser.write(message.encode('utf-8'))
                                ser.flush()
=======

            
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
                                if serialOn:
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
>>>>>>> Stashed changes
                            
                            #print(hasVisited)
                            print("information sent to reward motor: \n",message)
                            print("animal has reached reward zone")
                            rewarded = True
                            #hits.append(trial)
                            #store the reward area
                            data.loc[trial,"area_rewarded"] = trials.loc[trial].rewlocation
                            data.loc[trial,"time_to_reward"] = time_frame-trial_start_time
                            
                            #do calculations on time to reward
                    #elif mousePresent[item] and rewarded:
                    #    print ("animal is in the right zone but has been rewarded")
                    #elif enteredMaze==False:
                    #    print("mouse out of the maze")
                    #elif rewarded:
                    #    print("animal has been rewarded")
                    #else:
                    #    print("animal reached an unhandled condition")
                        
        time_old_frame=time_frame
        #old_frame=copy.deepcopy(gray)
        
    end_trial_time = sf.time_in_millis()-absolute_time_start
    data.loc[trial,"end_trial_time"] = end_trial_time
    sf.write_data(file_name=os.path.join(new_dir_path,f"session_data_{date_time}.csv"),
              mode="a",
              data=data.loc[trial].values)
                        

<<<<<<< Updated upstream
=======
                            

                
                #if trialEnd==1:
                #    trialOngoing = False

        #record end time of the trial
        #end_trial_time=  time.time()
        #trial_durations= end_trial_time - start_trial_time   #duration of the trial
        #trial_durations.append(trial_durations)

    
            
>>>>>>> Stashed changes
            
            #if trialEnd==1:
            #    trialOngoing = False

    #record end time of the trial
    #end_trial_time=  time.time()
    #trial_durations= end_trial_time - trial_start_time   #duration of the trial
    #trial_durations.append(trial_durations)

    
            
         
data.to_csv(os.path.join(new_dir_path,f"data_from_session_{date_time}.csv"))           
sessionDuration = time.time()-sessionStartTime




# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
if recordVideo:
<<<<<<< Updated upstream
    videoFileObject.release()#videoFile.release()
    #pass


=======
    #videoFile.release()
    pass
>>>>>>> Stashed changes
