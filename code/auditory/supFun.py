import pandas as pd
import cv2 as cv
import numpy as np
import time
import os
import tkinter as tk
from tkinter import filedialog as fd
from tkinter import simpledialog, Scale, HORIZONTAL, Label, ttk, filedialog, messagebox, Entry, font
import csv
import random
import wave


def collect_metadata(animal_ID):
    ear_mark = input("Ear mark identifiers? (y/n): \n").lower()
    birth_date = input("Insert animal birth date (dd/mm/yyyy): \n")
    gender = input("Insert animal assumed gender (m/f): \n").lower()


    data = {
    "animal ID": animal_ID,
    "ear mark identifier": ear_mark,
    "animal birth date": birth_date,
    "animal gender": gender,
    }

    return data

def save_metadata_to_csv(data, new_dir_path, file_name):
    df = pd.DataFrame([data])  # Convert dict to dataframe
    csv_path = os.path.join(new_dir_path, file_name)
    df.to_csv(csv_path, index=False)
    print(f"Metadata saved to: {csv_path}")


def setup_directories(base_path, date_time, animal_ID):
    new_directory = f"{date_time}{animal_ID}"
    new_dir_path = os.path.join(base_path, new_directory)
    ensure_directory_exists(new_dir_path)
    return new_dir_path

def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created: {path}")
    else:
        print(f"Directory already exists: {path}")

def get_user_inputs():
    animal_ID = input("Insert animal ID: \n").upper()

    return animal_ID, 

def get_current_time_formatted():
    return time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime())

def get_metadata():
    anID = input("enter animal identification:")
    date = input("enter date:")
    metadata = [anID, date]
    #figure out what else users want to store
    return metadata

def write_text(text="string",window="noWindow"):
    font                   = cv.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,500)
    fontScale              = 1
    fontColor              = (255,255,255)
    thickness              = 1
    lineType               = 2
    
    cv.putText(window,text, 
    bottomLeftCornerOfText, 
    font, 
    fontScale,
    fontColor,
    thickness,
    lineType)
    
    return


def start_camera(videoInput=0):
    cap = cv.VideoCapture(videoInput)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    return cap

def record_video(cap, recordFile, frame_width, frame_height, fps):
    cc = cv.VideoWriter_fourcc(*'XVID')
    videoFileObject = cv.VideoWriter(recordFile, cc, fps, (frame_width, frame_height))
    return videoFileObject
    #if not videoFile.isOpened():
    #    print("Error: Failed to create VideoWriter object.")
    #    cap.release()
    #    exit()

    #while True:
    #    ret, frame = cap.read()
    #    if ret:
    #        videoFile.write(frame)
    #        cv.imshow("Frame", frame)
    #        if cv.waitKey(1) & 0xFF in [ord('q'), 27]:
    #            break
    #    else:
    #        print("Error: Failed to read frame from camera.")
    #        break

    #videoFile.release()


def grab_n_convert_frame(cameraHandle):
    #capture a frame
    ret, frame = cameraHandle.read()
    #print(ret)
    # Our operations on the frame come here
    #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray=frame
    return gray,ret 

def csv_to_dict(fileName="rois.csv"):
    output = dict()
    temp =  pd.read_csv(fileName,index_col=0)
    temp = temp.transpose()
    output = temp.to_dict()
    return output


def define_rois(videoInput = 0,
                roiNames = ["entrance1","entrance2",
                             "ROI1","ROI2","ROI3","ROI4"],
                outputName = "rois.csv"):
    

    cap = cv.VideoCapture(videoInput)
    #cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        #break
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # Display the resulting frame
    #cv.imshow('frame', gray)

    #release the capture
    cap.release()

    rois = {}
    for entry in roiNames:
        print("please select location of "+str(entry))
        rois[entry] = cv.selectROI('frame', gray)
        #print(rois[entry])
    
    df = pd.DataFrame(rois)
    #print(df)
    #print(roiNames)
    df.index = ["xstart","ystart","xlen","ylen"]
    df.to_csv(outputName,index=["xstart","ystart","xlen","ylen"])
    #when all done destroy windows
    cv.destroyAllWindows()

    return df


#def draw_on_video(frame=gray,roi=(0,0,0,0)):
    #cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (0, 0, 0), -1)
def read_rois_from_csv(csvfile):
    with open(csvfile, newline='') as file:
        reader = csv.reader(file)
        headers = next(reader)
        roi_data = {header: [] for header in headers}
        
        for row in reader:
            for header, value in zip(headers, row):
                roi_data[header].append(int(value))
                
    return roi_data
    

def grab_cut(frame,xstart,ystart,xlen,ylen):
    
    cut = frame[ystart:ystart+ylen,
                xstart:xstart+xlen]
    return cut


def choose_csv(path='/home/andre/Desktop/maze_recordings/'):
    root=tk()
    # Show the file dialog and get the selected file name
    filename = fd.askopenfilename(initialdir=path)
    # Print the file name to the console
    #print (filename)
    root.destroy()
    return filename

def empty_frame(rows=25,roi_names=["entrance1","entrance2","ROI1","ROI2","ROI3","ROI3"]):
    #create a dataframe that will contain only nans for all things to be measured.
    #during the session we will fill up this df with data
    
    columns = ["ROIs", "frequency", "waveform", "volume", "time spent", "visitation count",
           "trial_start_time","end_trial_time","mouse_enter_time"]+roi_names
    data = pd.DataFrame(None, index=range(rows), columns=columns)
    return data


def ask_music_info():
    frequency = []
    volume = []
    waveform = []

    for i in range(1, 5): 
        freqs = int(input(f"Insert frequency for sound #{i}\n"))
        frequency.append(freqs)

        vols = int(input(f"Insert volume for sound #{i} [1-100]\n"))
        volume.append(vols)

        waves = input(f"Insert waveform for sound #{i} [sine, square, sawtooth, triangle, pulse wave, white noise]\n").lower()
        waveform.append(waves)

    return frequency, volume, waveform

def generate_sound_data(frequency, volume, waveform):
    duration = 10  # seconds
    fs = 44100  # Sample rate
    t = np.linspace(0, duration, int(fs * duration), False)

    sound = np.zeros_like(t)

    if waveform == 'sine':
        sound = np.sin(frequency * t * 2 * np.pi)
    elif waveform == 'square':
        sound = np.sign(np.sin(frequency * t * 2 * np.pi))
    elif waveform == 'sawtooth':
        sound = 2 * (t * frequency % 1) - 1
    elif waveform == 'triangle':
        sound = 2 * np.abs(2 * (t * frequency - np.floor(0.5 + t * frequency))) - 1
    elif waveform == 'pulse wave':
        sound = np.where((t % (1 / frequency)) < (1 / frequency) * 0.5, 1.0, -1.0)  # 0.5 in this case is the duty cycle
    elif waveform == 'white noise':
        samples = int(fs * duration)
        sound = np.random.uniform(low=-1.0, high=1.0, size=samples)

    return sound * volume

def create_trials(frequency, volume, waveform):
    # Number of repetitions for each ROI
    total_repetitions = 5
    zero_repetitions = 1
    # ROIs to be used
    rois = ["ROI1", "ROI2", "ROI3", "ROI4"]
    # Create a list of ROIs repeated total_repetitions times
    rois_repeated = rois * total_repetitions

    # Initialize lists to store the shuffled frequency, volume, and waveform
    frequency_final = []
    volume_final = []
    waveform_final = []
    wave_arrays = []
    repetition_numbers = []

    # Total number of sound data sets
    num_sounds = len(frequency)
    previous_trials = set()

    def get_trial_tuple(frequency, volume, waveform):
        return tuple(zip(frequency, volume, waveform))

    def shuffle_data(frequency, volume, waveform):
        combined = list(zip(frequency, volume, waveform))
        random.shuffle(combined)
        return zip(*combined)

    # Create trials
    for i in range(total_repetitions):
        for j in range(len(rois)):
            repetition_numbers.append(i + 1)  # Add repetition number
            if i < zero_repetitions:
                # Add zero data for the specified repetitions
                frequency_final.append(0)
                volume_final.append(0)
                waveform_final.append('none')
                wave_arrays.append(np.zeros(441000))  # Assuming 10 seconds of silence
            elif i == 1:  # Trial 2 should use the initial list
                frequency_final.append(frequency[j])
                volume_final.append(volume[j])
                waveform_final.append(waveform[j])
                wave_arrays.append(generate_sound_data(frequency[j], volume[j], waveform[j]))
            else:  # Trials 3 to 5 should be shuffled
                if j == 0:
                    # Shuffle the sound data at the beginning of each trial
                    while True:
                        shuffled_frequency, shuffled_volume, shuffled_waveform = shuffle_data(frequency, volume, waveform)
                        trial_tuple = get_trial_tuple(shuffled_frequency, shuffled_volume, shuffled_waveform)
                        if trial_tuple not in previous_trials:
                            previous_trials.add(trial_tuple)
                            break
                frequency_final.append(shuffled_frequency[j])
                volume_final.append(shuffled_volume[j])
                waveform_final.append(shuffled_waveform[j])
                wave_arrays.append(generate_sound_data(shuffled_frequency[j], shuffled_volume[j], shuffled_waveform[j]))

    # Create the DataFrame with the repetition numbers, repeated ROIs, and final data
    df = pd.DataFrame({
        "trial": repetition_numbers,
        "ROIs": rois_repeated,
        "frequency": frequency_final,
        "volume": volume_final,
        "waveform": waveform_final,
        "wave_arrays": wave_arrays
    })

    # Add other necessary columns filled with NaNs or default values
    df["time spent"] = [None] * len(df)
    df["visitation count"] = [None] * len(df)
    df["trial_start_time"] = [None] * len(df)
    df["end_trial_time"] = [None] * len(df)
    df["mouse_enter_time"] = [None] * len(df)

    return df

def save_sound(sound_data, frequency, waveform):
    parent_folder_selected = filedialog.askdirectory()

    if parent_folder_selected:
        local_time = time.localtime()
        date_time = time.strftime('%d-%m-%Y_%H', local_time)  # adding hour in case of trial/error

        # Create a new subdirectory within the selected folder that takes the day
        subfolder_path = os.path.join(parent_folder_selected, date_time)

        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

        # Use the frequency as the name for the sound
        file_name = f"{frequency}.wav"

        # Save in the folder
        full_file_path = os.path.join(subfolder_path, file_name)
        save_sound_to_file(sound_data, full_file_path)

def save_sound_to_file(sound_data, file_path):
    '''Modify this if need different audio formats'''
    # Normalize sound_data to 16-bit PCM format for WAV file
    sound_data_normalized = np.int16((sound_data / np.max(np.abs(sound_data))) * 32767)
    with wave.open(file_path, 'w') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16 bits
        wf.setframerate(44100)  # Sampling rate
        wf.writeframes(sound_data_normalized.tobytes())

        
def select_sound_folder(): 
    root = tk.Tk()
    sound_folder = filedialog.askdirectory(title="Select Folder Containing Sound Files")
    if sound_folder:
        root.mainloop()  # This will not do anything visible since the window is withdrawn
    else:
        root.destroy()
    
    return sound_folder



def time_in_millis():
    millis=round(time.time() * 1000)
    return millis

def write_data(file_name="tests.csv",mode="a",data=["test","test","test"]):
    data_fh = open(file_name,mode)
    data_writer = csv.writer(data_fh, delimiter=',')
    data_writer.writerow(data)
    data_fh.close()