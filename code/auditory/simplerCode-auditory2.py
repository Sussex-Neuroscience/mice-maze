#import all needed libraries
import numpy as np
import cv2 as cv
import pandas as pd
import supFun as sf
import time
#from serial import Serial
import serial
import os
import pandas as pd
from tkinter import filedialog as fd
from tkinter import *
import csv
import sounddevice as sd
import soundfile as sf1


#need to determine how long the trial is, idk if modifying line "while trialongoing" or around the "if entered maze"


#for code inspection and testing of the code purposes we add a small pause in between frames in
#the main code loop... this variable just below this needs to be set to False if one is running
#the actual experiments
pause_between_frames=True

#If ROIs need to be drawn by experiementer, set the next variable to TRUE
drawRois = False


#choose sounds folder
make_sounds = False


#If just testing and no video needs to be recorded, set the next variable to FALSE
recordVideo = False
#define where the video is coming from. Use 0 for the first camera on the computer,
#or a complete file path to use a pre-recorded video
videoInput =  "/home/andre/Desktop/maze_test.mp4"

#get the current date and time, so all files created do not overwrite existing data
date_time = sf.get_current_time_formatted()


animal_ID,  = sf.get_user_inputs()
base_path = os.path.join(os.path.expanduser('~'), 'OneDrive/Desktop/maze_experiments', 'maze_recordings')
print(base_path)
sf.ensure_directory_exists(base_path)

new_dir_path = sf.setup_directories(base_path, date_time, animal_ID)
rec_name = f"{animal_ID}_{date_time}.mp4"
recordFile = os.path.join(new_dir_path, rec_name)
print(f"Video will be saved to: {recordFile}")

metadata = sf.collect_metadata(animal_ID)
sf.save_metadata_to_csv(metadata, new_dir_path, f"{animal_ID}_{date_time}.csv")


#if we want to make new trials instead of reusing 
if make_sounds:
    #ask user for tones, volume and waveform 
    frequency, volume, waveform = sf.ask_music_info()
    

    for i in range(len(frequency)):
        sound_data = sf.generate_sound_data(frequency[i], volume[i], waveform[i])
        # sf.save_sound(sound_data, frequency[i], waveform[i])

    trials = sf.create_trials(frequency, volume, waveform)

    #save trials to csv
    trials.to_csv(os.path.join(new_dir_path,f"trials_{date_time}.csv"))



if drawRois:
    sf.define_rois(videoInput = videoInput,
                    roiNames = ["entrance1","entrance2",
                            "ROI1","ROI2","ROI3","ROI4"],
                    outputName = base_path+"/"+"rois1.csv")
    rois = pd.read_csv(base_path+"/"+"rois1.csv",index_col=0)
else:
    rois = pd.read_csv(base_path+"/"+"rois1.csv",index_col=0)



#load ROI information


thresholds = dict()
#timeSpentAreas = dict()
for item in rois:
    #create threshold values for each area
    thresholds[item] = 0

#preload variables
hasVisited = dict()
mousePresent = dict()

cap = sf.start_camera(videoInput=videoInput)

frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv.CAP_PROP_FPS))
if recordVideo:
    videoFileObject = sf.record_video(cap, recordFile, frame_width, frame_height, fps)


#grab one frame:
valid,gray = cap.read()
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


#create two windows to show the animal movement while in maze:
cv.namedWindow('binary maze plus ROIs', cv.WINDOW_NORMAL)
cv.namedWindow('original image', cv.WINDOW_NORMAL)

absolute_time_start = sf.time_in_millis()


#open sound folder in case the thingy with the playing from sound data in the csv doesn't work
#sound_folder= sf.select_sound_folder()

#load the trials file (description of each trial)
trials = pd.read_csv(sf.choose_csv())

unique_trials = trials['trial_ID'].unique()
trial_lengths = [1, 15, 2, 15, 2, 15, 2, 15, 2]

######## start working from here ###########
for trial in unique_trials:
    time_trial = trial_lengths[trial-1]
    print("trial_duration: ", time_trial)
    if cv.waitKey(1) & 0xFF in [ord('q'), 27]:  # Quit on 'q' or ESC
        break

    for item in rois:
        hasVisited[item] = False 
        
        #ent1 = 0
        #ent2 = 0
    trialOngoing = True
  
    enteredMaze = False

    hasLeft1 = False
    hasLeft2 = False
    e2Aftere1 = False
    e1Aftere2 = False
    
    
    ent1History = [False,False]
    ent2History = [False,False]
    started_trial_timer = False
    ongoing_trial_duration = 0
    time_old_frame= 0 


    while trialOngoing:
        #routine to check if enough time has elapsed
        #if started_trial_timer:
        #    #routine to updated ongoing_trial_duration
        #    #     
        if pause_between_frames:
            time.sleep(0.01)
        
        valid,grayOriginal = cap.read()
        ret,gray = cv.threshold(grayOriginal,180,255,cv.THRESH_BINARY)
        #binGray = gray[:,:,2]
        time_frame=time_frame=sf.time_in_millis()-absolute_time_start
        
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

        # Display the resulting frame
        cv.imshow('binary maze plus ROIs', gray)
        cv.imshow('original image', grayOriginal)
        
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
            
            rois_list= ['ROI1', 'ROI2', 'ROI3', 'ROI4']

            if mousePresent[item]:
                
                hasVisited[item] = True
                duration=time_frame-time_old_frame
                #print(duration)

                if item in rois_list:
                    condition = (trials['trial_ID'] == trial) & (trials['ROIs'] == item)
                    
                    if pd.isna(trials.loc[condition, 'time spent']).all():
                        trials.loc[condition, 'time spent'] = duration
                    else:
                        trials.loc[condition, 'time spent'] += duration

        time_old_frame = time_frame
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
            print(f"mouse entered the maze for trial {trial}")
            #if started_trial_timer==False:
                #starting time stamp
                #routine to start the timer
            #    started_trial_timer=True
            #start timer from the time the mouse has entered the maze
            print(f"Starting trial {trial} for {time_trial} minutes.")
            start_time = time.time()
            end_time = start_time + time_trial * 60 

            if time.time() >= end_time:
                trialOngoing=False

            #put in the  df the time the mouse entered the maze
            trials.loc[trials["trial_ID"] == trial,"mouse_enter_time"]=time_frame#-trial_start_time
            trials.loc[trials["trial_ID"] == trial,"trial_start_time"]=start_time
            trials.loc[trials["trial_ID"] == trial,"end_trial_time"]=end_time
            enteredMaze = True
        if e1Aftere2:
            print("mouse has left the maze")
            #trialOngoing=False
            #trialDurations.append((trial,time.time()-trialStartTime))
            enteredMaze = False

        #
        # while time.time() < end_time:        
        if enteredMaze:
        

            for item in mousePresent:
                if not mousePresent["ROI1"] and \
                   not mousePresent["ROI2"] and \
                   not mousePresent["ROI3"] and \
                   not mousePresent["ROI4"]:
                   sd.stop()

                
                    

                #if trials.loc[trial].givereward:
                

                if "1" in item or "2" in item or "3" in item or "4" in item:
                    #print("item",item)
                    #print("mouse present",mousePresent)
                    if mousePresent[item]:
                        if trials["waveform"][trials["trial_ID"][trial]]!="none":
                        #temp_roi= trials["ROIs"][trials["trial_ID"][trial]]
                        
                            
                            if item=="ROI1":    
                                sound= trials.loc[(trials['trial_ID'] == trial) & (trials['ROIs'] == 'ROI1'), 'wave_arrays'].item()
                                
                            if item=="ROI2" :
                                sound= trials.loc[(trials['trial_ID'] == trial) & (trials['ROIs'] == 'ROI2'), 'wave_arrays'].item()
                                
                            if item=="ROI3" :
                                sound= trials.loc[(trials['trial_ID'] == trial) & (trials['ROIs'] == 'ROI3'), 'wave_arrays'].item()
                                
                            if item=="ROI4" :
                                sound= trials.loc[(trials['trial_ID'] == trial) & (trials['ROIs'] == 'ROI4'), 'wave_arrays'].item()
                            
                            sf.play_sound(sound)
                        
                        
                            
                                
                            
        
                        print("animal has reached sound zone")
                                               
        time_old_frame=time_frame



    


# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
if recordVideo:
    videoFileObject.release()#videoFile.release()
    #pass
