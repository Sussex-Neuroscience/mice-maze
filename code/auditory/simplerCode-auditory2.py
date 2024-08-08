import numpy as np
import cv2 as cv
import pandas as pd
import supFun as sf
import time
#from serial import Serial
import serial
import os
import csv
import sounddevice as sd
import soundfile as sf1
from tkinter import filedialog as fd
from tkinter import *

# Variables
pause_between_frames = True
drawRois = False
make_sounds = False
#If we are recording a video, this needs to be true and videoInput needs to be set to 0 (or 1, depending on the camera)
recordVideo = False
videoInput = "C:/Users/aleja/Downloads/maze_test.mp4"

# Setup -- ask users for mouse info
date_time = sf.get_current_time_formatted()
animal_ID, = sf.get_user_inputs()
#create new directory
base_path = os.path.join(os.path.expanduser('~'), 'OneDrive/Desktop/maze_experiments', 'maze_recordings')
sf.ensure_directory_exists(base_path)
new_dir_path = sf.setup_directories(base_path, date_time, animal_ID)
#save recording in new directory
rec_name = f"{animal_ID}_{date_time}.mp4"
recordFile = os.path.join(new_dir_path, rec_name)
metadata = sf.collect_metadata(animal_ID)
sf.save_metadata_to_csv(metadata, new_dir_path, f"{animal_ID}_{date_time}.csv")

# Make sounds 
if make_sounds:
    frequency, volume, waveform = sf.ask_music_info()
    for i in range(len(frequency)):
        sound_data = sf.generate_sound_data(frequency[i], volume[i], waveform[i])
    trials, sound_arrays = sf.create_trials(frequency, volume, waveform)
    np.save(os.path.join(new_dir_path, f"trials_{date_time}.npy"), sound_arrays)
    trials.to_csv(os.path.join(new_dir_path, f"trials_{date_time}.csv"))

# Draw ROIs 
if drawRois:
    sf.define_rois(videoInput=videoInput,
                   roiNames=["entrance1", "entrance2", "ROI1", "ROI2", "ROI3", "ROI4"],
                   outputName=base_path + "/" + "rois1.csv")
    rois = pd.read_csv(base_path + "/" + "rois1.csv", index_col=0)
else:
    rois = pd.read_csv(base_path + "/" + "rois1.csv", index_col=0)

# Loading the ROI info and initialising the variables per roi
thresholds = {item: 0 for item in rois}
hasVisited = {item: False for item in rois}
mousePresent = {item: False for item in rois}
visitation_count = {item: 0 for item in rois}
previous_state = {item: False for item in rois}

cap = sf.start_camera(videoInput=videoInput)
frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv.CAP_PROP_FPS))
if recordVideo:
    videoFileObject = sf.record_video(cap, recordFile, frame_width, frame_height, fps)

valid, gray = cap.read()
ret, gray = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)

areas = {}
for item in rois:
    areas[item] = sf.grab_cut(gray,
                              xstart=rois[item]["xstart"],
                              ystart=rois[item]["ystart"],
                              xlen=rois[item]["xlen"],
                              ylen=rois[item]["ylen"])
    thresholds[item] = np.sum(areas[item])

sessionStartTime = time.time()
cv.namedWindow('binary maze plus ROIs', cv.WINDOW_NORMAL)
cv.namedWindow('original image', cv.WINDOW_NORMAL)
absolute_time_start = sf.time_in_millis()

base_name = sf.choose_csv()
base_name = base_name[:base_name.find(".")]
trials = pd.read_csv(base_name + ".csv")
sound_array = np.load(base_name + ".npy")
unique_trials = trials['trial_ID'].unique()
trial_lengths = [1, 1, 1, 1, 1, 1, 1, 1, 1]

for trial in unique_trials:
    time_trial = trial_lengths[trial - 1]
    print("trial_duration: ", time_trial)
    if cv.waitKey(1) & 0xFF in [ord('q'), 27]:  # Quit on 'q' or ESC
        break

# initialising the variables for the trials
    for item in rois:
        hasVisited[item] = False
    
    trialOngoing = True
    enteredMaze = False
    hasLeft1 = False
    hasLeft2 = False
    e2Aftere1 = False
    e1Aftere2 = False
    ent1History = [False, False]
    ent2History = [False, False]
    started_trial_timer = False
    ongoing_trial_duration = 0
    time_old_frame = 0
    start_new_time = True

    while trialOngoing:
        if pause_between_frames:
            time.sleep(0.01)

        #read the video 
        valid, grayOriginal = cap.read()
        ret, gray = cv.threshold(grayOriginal, 180, 255, cv.THRESH_BINARY)
        time_frame = sf.time_in_millis() - absolute_time_start

        if not valid:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        if recordVideo:
            videoFileObject.write(grayOriginal)

        # show the rois on the video
        for item in rois:
            # draw the rectangle per ROI
            cv.rectangle(gray, (rois[item]["xstart"],
                                rois[item]["ystart"]),
                         (rois[item]["xstart"] + rois[item]["xlen"],
                          rois[item]["ystart"] + rois[item]["ylen"]),
                         color=(120, 0, 0), thickness=2)
            
            # Add the name of the ROI above the rectangle
            cv.putText(gray, item,
               (rois[item]["xstart"], rois[item]["ystart"] - 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv.LINE_AA)

            

        cv.imshow('binary maze plus ROIs', gray)
        cv.imshow('original image', grayOriginal)

        if cv.waitKey(1) & 0xFF in [ord('q'), 27]:  # Quit on 'q' or ESC
            break

        for item in rois:
            areas[item] = sf.grab_cut(gray,
                                      xstart=rois[item]["xstart"],
                                      ystart=rois[item]["ystart"],
                                      xlen=rois[item]["xlen"],
                                      ylen=rois[item]["ylen"])

            mousePresent[item] = np.sum(areas[item]) < thresholds[item] / 2
            rois_list = ['ROI1', 'ROI2', 'ROI3', 'ROI4']

            
            if mousePresent[item]:
                print(f"mouse in {item}")
                hasVisited[item] = True
                visitation_count[item] +=1
                duration = time_frame - time_old_frame

                if item in rois_list:
                    condition = (trials['trial_ID'] == trial) & (trials['ROIs'] == item)

                    #trials.loc[condition, 'visitation count'] = visitation_count[item]

                    if pd.isna(trials.loc[condition, 'time spent']).all():
                        trials.loc[condition, 'time spent'] = duration
                    else:
                        trials.loc[condition, 'time spent'] += duration

                    # Detect Transitions
                    if not previous_state[item] and mousePresent[item]:  # Mouse just entered the ROI
                        visitation_count[item] += 1
                        trials.loc[condition, 'visitation count'] = visitation_count[item]


                    

        time_old_frame = time_frame

        ent1History.insert(0, mousePresent["entrance1"])
        ent1History.pop(-1)
        ent2History.insert(0, mousePresent["entrance2"])
        ent2History.pop(-1)

        if not ent1History[0] and ent1History[1]:
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
            print(f"mouse entered the maze for trial {trial}")
            print(f"Starting trial {trial} for {time_trial} minutes.")
            if start_new_time:
                start_time = time.time()
                start_new_time = False
                sound1 = sound_array[trials.loc[(trials['trial_ID'] == trial) & (trials['ROIs'] == 'ROI1')].index[0]]
                sound2 = sound_array[trials.loc[(trials['trial_ID'] == trial) & (trials['ROIs'] == 'ROI2')].index[0]]
                sound3 = sound_array[trials.loc[(trials['trial_ID'] == trial) & (trials['ROIs'] == 'ROI3')].index[0]]
                sound4 = sound_array[trials.loc[(trials['trial_ID'] == trial) & (trials['ROIs'] == 'ROI4')].index[0]]

            end_time = start_time + time_trial * 60
            print(end_time)

            #trials.loc[trials["trial_ID"] == trial, "mouse_enter_time"] = start_time #add to supfun code again if we want mouse enter time
            trials.loc[trials["trial_ID"] == trial, "trial_start_time"] = start_time
            trials.loc[trials["trial_ID"] == trial, "end_trial_time"] = end_time
            enteredMaze = True

        if e1Aftere2 and enteredMaze:
            print("mouse has left the maze")
            if time.time() >= end_time:
                print(time.time() >= end_time)
                trialOngoing = False
            enteredMaze = False

        if enteredMaze:
            if time.time() >= end_time:
                trialOngoing = False

            for item in mousePresent:
                if not mousePresent["ROI1"] and \
                   not mousePresent["ROI2"] and \
                   not mousePresent["ROI3"] and \
                   not mousePresent["ROI4"]:
                    reset_play = True
                    sd.stop()

                if "1" in item or "2" in item or "3" in item or "4" in item:
                    if mousePresent[item]:
                        if trials[trials["trial_ID"] == trial]["waveform"].values[0] != "none":
                            if item == "ROI1" and reset_play:
                                reset_play = False
                                sf.play_sound(sound1)
                            if item == "ROI2" and reset_play:
                                reset_play = False
                                sf.play_sound(sound2)
                            if item == "ROI3" and reset_play:
                                reset_play = False
                                sf.play_sound(sound3)
                            if item == "ROI4" and reset_play:
                                reset_play = False
                                sf.play_sound(sound4)

        time_old_frame = time_frame

        # Update previous_state dictionary with current state
        for item in rois_list:
            previous_state[item] = mousePresent[item]

    # Save the updated trials DataFrame to CSV after each trial
    print(f"Saving trials data for trial {trial} to CSV")
    trials.to_csv(base_name + ".csv", index=False)

cap.release()
cv.destroyAllWindows()
if recordVideo:
    videoFileObject.release()

print("Experiment completed and data saved.")
