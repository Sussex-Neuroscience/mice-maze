import numpy as np
import cv2 as cv
import pandas as pd
import supfun_sequences as sf
import time
#from serial import Serial
import serial
import os
import csv
import sounddevice as sd
import soundfile as sf1
from tkinter import filedialog as fd
from tkinter import *

#set the default sample rate and channel for the sound card
sd.default.samplerate = 192000
#python -m sounddevice
sd.default.device = 12

#vol needs to be around 50dB


# Variables
testing= False
#testing with longer silence in the middle of the sequence
longer_silence= False

pause_between_frames = False

#set number of ROIs 
rois_number = 8



#set to true to create/modify ROIs .csv file
drawRois = False
#set to true to make individual sine sounds
make_simple_sounds = False
# set to true to make sequences of tones
make_sequences = False

#set true to make individual complex sounds. This contains 2 intervals, 1 vocalisation and one pure tone
make_complex_sounds= True



#If we are recording a video, this needs to be true and videoInput needs to be set to 0 (or 1, depending on the camera)
recordVideo = True
videoInput = 0
#"C:/Users/labadmin/Desktop/auditory_maze_experiments/maze_recordings/maze_recordings_sequences/time_2024-09-25_11_47_12mouse6705/mouse6705_time_2024-09-25_11_47_12.mp4"
#"C:/Users/aleja/OneDrive/Desktop/maze_experiments/maze_recordings/2024-08-15_16_12_305872/5872_2024-08-15_16_12_30.mp4"
#"C:/Users/aleja/Downloads/maze_test.mp4"

# Setup -- ask users for mouse info
date_time = sf.get_current_time_formatted()
date_time= f"time_{date_time}"
animal_ID, = sf.get_user_inputs()
animal_ID= f"mouse{animal_ID}"
#create new directory
base_path = os.path.join(os.path.expanduser('~'), 'Desktop/auditory_maze_experiments/maze_recordings', 'maze_recordings_sequences')
sf.ensure_directory_exists(base_path)
new_dir_path = sf.setup_directories(base_path, date_time, animal_ID)
#save recording in new directory
rec_name = f"{animal_ID}_{date_time}.mp4"
recordFile = os.path.join(new_dir_path, rec_name)
metadata = sf.collect_metadata(animal_ID)
sf.save_metadata_to_csv(metadata, new_dir_path, f"{animal_ID}_{date_time}.csv")

#define rois
entrance_rois= ["entrance1", "entrance2"]
rois_list = sf.get_rois_list(rois_number)

roiNames= entrance_rois + rois_list

#set base_name for trials list and sound array to be saved as 
base_name = f"trials_{date_time}"

#make just sounds
if make_simple_sounds:
    # frequency= sf.ask_music_info_simple_sounds(rois_number)
    frequency = [10000, 12000, 14000, 16000, 18735, 20957, 22543, 24065]
    # for i in range(len(frequency)):
    #     sound_data = sf.generate_sound_data(frequency[i])
    trials, sound_array = sf.create_simple_trials(rois_list, frequency)
    np.save(os.path.join(new_dir_path, f"{base_name}.npy"), np.array(sound_array, dtype=object))
    trials.to_csv(os.path.join(new_dir_path, f"{base_name}.csv"))

# Make sequences of sounds
if make_sequences:
    frequency, patterns= sf.ask_music_info_sequences(rois_number)
    trials, sound_array = sf.create_trials(rois_list, frequency, patterns)
    #make all arrays the same size because otherwise it won't save sound arrays
    min_length = min(len(arr) for arr in sound_array)
     # Trim the arrays to the minimum length
    trimmed_sound_arrays = sf.trim_arrays(sound_array, min_length)
    np.save(os.path.join(new_dir_path, f"{base_name}.npy"), np.array(trimmed_sound_arrays, dtype=object))
    trials.to_csv(os.path.join(new_dir_path, f"{base_name}.csv"))


#make complex sounds
elif make_complex_sounds:
    #essentially this gives me the option to be prompted what I want as intervals vs hard coding them
    manual = False
    if manual:
        frequency, intervals, intervals_names = sf.ask_info_intervals(rois_number)
    else: 
        tonal_centre = 10000 #Hz
        intervals_list = ["perf_5", "perf_4", "maj_6", "tritone", "min_2", "maj_7"]
        frequency, intervals, intervals_names = sf.info_intervals_hc(rois_number, tonal_centre, intervals_list)

    trials, sound_array = sf.create_trials_for_intervals(rois_list, frequency, intervals, intervals_names)
    np.save(os.path.join(new_dir_path, f"{base_name}.npy"), np.array(sound_array, dtype=object))
    trials.to_csv(os.path.join(new_dir_path, f"{base_name}.csv"))


# base_name = sf.choose_csv()
# base_name = base_name[:base_name.find(".")]
# trials = pd.read_csv(base_name + ".csv")
# sound_array = np.load(base_name + ".npy", allow_pickle=True)
unique_trials = trials['trial_ID'].unique()


# Draw ROIs 
if drawRois:

    sf.define_rois(videoInput=videoInput,
                roiNames=roiNames,
                outputName=base_path + "/" + "rois1.csv")
    rois = pd.read_csv(base_path + "/" + "rois1.csv", index_col=0)
else:
    rois = pd.read_csv(base_path + "/" + "rois1.csv", index_col=0)

# Loading the ROI info and initialising the variables per roi
thresholds = {item: 0 for item in rois}
hasVisited = {item: False for item in rois}
mousePresent = {item: False for item in rois}
visitation_count = {item: 0 for item in rois}


cap = sf.start_camera(videoInput=videoInput)
frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv.CAP_PROP_FPS))


if recordVideo:
    videoFileObject = sf.record_video(cap, recordFile, frame_width, frame_height, fps)

valid, gray = cap.read()
ret, gray = cv.threshold(gray, 160,255, cv.THRESH_BINARY)


# back_sub = cv.createBackgroundSubtractorMOG2()
# fg_mask = back_sub.apply(gray)
# # cv.imshow('Foreground Mask', fg_mask)
# # cv.waitKey(0)
# #ret,gray_inv = cv.threshold(grayOriginal,180,255,cv.THRESH_BINARY_INV)

# try:
#     contours,hierarchy = cv.findContours(fg_mask, cv.RETR_TREE,
#                                 cv.CHAIN_APPROX_NONE)
#     if contours:
#         cv.drawContours(fg_mask, contours[0], -1, 255, 4)
#         cv.drawContours(gray, contours[0], 0, (0,255,255), 2)
#         print("contours found")
#     else:
#         print("No contours found")
# except:
#     print("no")


areas = {}
for _ in range(10):
    valid, gray = cap.read()
    if not valid:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    for item in rois:
        areas[item] = sf.grab_cut(gray,
                                xstart=rois[item]["xstart"],
                                ystart=rois[item]["ystart"],
                                xlen=rois[item]["xlen"],
                                ylen=rois[item]["ylen"])
        thresholds[item] = np.sum(areas[item])
        print(f"ROI: {item}, Threshold: {thresholds[item]}")

sessionStartTime = time.time()
cv.namedWindow('binary maze plus ROIs', cv.WINDOW_NORMAL)
cv.namedWindow('original image', cv.WINDOW_NORMAL)
absolute_time_start = sf.time_in_millis()



if testing:
    trial_lengths = [0.1, 0.2, 0.2, 2, 0.2, 2, 0.2, 2, 0.2]

    #inserted a longer silence trial in the middle, to check if the mice prefer the maze in silence
elif not testing and longer_silence:
    trial_lengths = [15, 15, 2, 15, 15, 15, 2, 15, 2]
else:
    trial_lengths = [15, 15, 2, 15, 2, 15, 2, 15, 2]



for trial in unique_trials:
    time_trial = trial_lengths[trial - 1]
    print("trial_duration: ", time_trial)
    if cv.waitKey(1) & 0xFF in [ord('q'), 27]:  # Quit on 'q' or ESC
        break

# initialising the variables for each trial
    for item in rois:
        hasVisited[item] = False
        visitation_count[item]= 0
    
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
        ret, gray = cv.threshold(grayOriginal, 160,255, cv.THRESH_BINARY)
        time_frame = sf.time_in_millis() - absolute_time_start # constantly updating - represents how many ms into the trial the frame(?) is

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
        #cv.imshow('frame_diff', fg_mask) 

        if cv.waitKey(1) & 0xFF in [ord('q'), 27]:  # Quit on 'q' or ESC
            break

        for item in rois:
            areas[item] = sf.grab_cut(gray,
                                      xstart=rois[item]["xstart"],
                                      ystart=rois[item]["ystart"],
                                      xlen=rois[item]["xlen"],
                                      ylen=rois[item]["ylen"])

            mousePresent[item] = np.sum(areas[item]) < thresholds[item] * 0.5
            
            
            rois_list = ['ROI1', 'ROI2', 'ROI3', 'ROI4', "ROI5", "ROI6", "ROI7", "ROI8"]

            
            if mousePresent[item]:
                # print(f"mouse in {item}")
                duration = time_frame - time_old_frame #
                
            

                if item in rois_list:
                    
                    condition = (trials['trial_ID'] == trial) & (trials['ROIs'] == item)

                    if not hasVisited[item]: 
                        visitation_count[item]+= 1
                        print(f'visitation count for {item} : {visitation_count[item]}')
                        hasVisited[item] = True
                        trials.loc[condition, 'visitation_count'] = visitation_count[item]
                        

                    #trials.loc[condition, 'visitation count'] = visitation_count[item]

                    if pd.isna(trials.loc[condition, 'time_spent']).all():
                        trials.loc[condition, 'time_spent'] = duration
                    else:
                        trials.loc[condition, 'time_spent'] += duration       
            else:
                hasVisited[item]= False            


                    

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
                # Create a dictionary to store sounds dynamically for each ROI
                sounds = {}

                # Loop through the dynamically generated ROIs
                for roi in rois_list:
                    sound = sound_array[trials.loc[(trials['trial_ID'] == trial) & (trials['ROIs'] == roi)].index[0]]
                    sounds[roi] = sound


            end_time = start_time + time_trial * 60
            remaining_total_time= end_time - time.time()
            remaining_minutes= int((end_time - time.time())/60)
            remaining_seconds= int((end_time - time.time())%60)
            print(f"Remaining Time: {remaining_total_time}\nApprox: {remaining_minutes}:{remaining_seconds}")


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
            #if time.time() >= end_time:
                #trialOngoing = False

            for item in mousePresent:
                # Dynamically check if all ROIs are False
                if all(not mousePresent[roi] for roi in rois_list):
                    reset_play = True
                    sd.stop()

                # Check if the item belongs to the dynamically created ROIs
                if item in rois_list:
                    if mousePresent[item]:
                        frequency_value = trials[trials["trial_ID"] == trial]["frequency"].values[0]
                        zeros = ["0", 0, [0, 0]]
                        
                        if not any(np.array_equal(frequency_value, zero) for zero in zeros):
                            if reset_play:
                                reset_play = False
                                roi_index = rois_list.index(item)

                                # Dynamically play the corresponding sound
                                if make_complex_sounds:
                                    sf.play_interval(sounds[item][0][0], sounds[item][1][0])
                                else:
                                    sf.play_sound(sounds[item])
                                
                                print(f"Playing sound for {item}")


        time_old_frame = time_frame


    # Save the updated trials DataFrame to CSV after each trial
    print(f"Saving trials data for trial {trial} to CSV")
    trials.to_csv(os.path.join(new_dir_path, f"{base_name}.csv"), index=False)

cap.release()
cv.destroyAllWindows()
if recordVideo:
    videoFileObject.release()

print("Experiment completed and data saved.")
