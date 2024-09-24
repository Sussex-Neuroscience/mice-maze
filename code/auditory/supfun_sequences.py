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
import sounddevice as sd


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
    cc = cv.VideoWriter_fourcc(*'mp4v')
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


def define_rois(videoInput=0,
                roiNames=["entrance1", "entrance2", "ROI1", "ROI2", "ROI3", "ROI4"],
                outputName="rois.csv"):
    
    cap = cv.VideoCapture(videoInput)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        cap.release()
        exit()

    # Convert frame to grayscale for display
    gray = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    # Dictionary to store ROIs
    rois = {}

    for entry in roiNames:
        print(f"Please select location of {entry}")

        # Display the current frame with already selected ROIs
        for name, roi in rois.items():
            x, y, w, h = roi
            cv.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv.putText(gray, name, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv.LINE_AA)
        
        # Show the frame and select ROI
        cv.imshow('frame', gray)
        rois[entry] = cv.selectROI('frame', gray, fromCenter=False, showCrosshair=True)
        
        # Draw the selected ROI on the frame
        x, y, w, h = rois[entry]
        cv.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(gray, entry, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(rois)
    df.index = ["xstart", "ystart", "xlen", "ylen"]
    
    # Save the DataFrame to a CSV file
    df.to_csv(outputName, index=["xstart", "ystart", "xlen", "ylen"])
    
    # Release the capture and destroy windows
    cap.release()
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


def choose_csv():
    root=tk.Tk()
    # Show the file dialog and get the selected file name
    filename = fd.askopenfilename()
    # Print the file name to the console
    #print (filename)
    root.destroy()
    return filename

def empty_frame(rows=25,roi_names=["entrance1","entrance2","ROI1","ROI2","ROI3","ROI3"]):
    #create a dataframe that will contain only nans for all things to be measured.
    #during the session we will fill up this df with data
    
    columns = ["ROIs", "frequency", "waveform", "volume", "time spent", "visitation count",
           "trial_start_time","end_trial_time"]+roi_names
    data = pd.DataFrame(None, index=range(rows), columns=columns)
    return data


def ask_music_info():
    freqs = []
    events = ["A", "B", "C", "o"]

    for i in range(1,4): 

        freq = int(input(f"Insert frequency for sound #{i}\n"))
        freqs.append(freq)
    freqs.append(0)

    events_dict = dict(zip(events, freqs))
    #each sound is 0.1 seconds, sequences need to be 10 seconds long, so 100 notes (or 50 for 0.2))

    frequency1= []
    frequency2=[]
    frequency3=[]

    for i in range(25):
        frequency1.append("A")
        frequency1.append("o")
        frequency2.append("A")
        frequency2.append("B")
        frequency3.append("A")
        frequency3.append("B")
        frequency3.append("C")
        
        
    frequency4=[random.choice(events) for _ in range(100)]

        
    f1= [events_dict[i] for i in frequency1]
    f2= [events_dict[i] for i in frequency2]
    f3= [events_dict[i] for i in frequency3]
    f4= [events_dict[i] for i in frequency4]


    patterns = [frequency1, frequency2, frequency3, frequency4]
    frequency = [f1, f2, f3, f4]

    return frequency, patterns

def generate_sound_data(frequency, volume=100, waveform="sine", duration = 0.2,fs = 44100):
      # seconds
      # Sample rate
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

def get_trial_tuple(frequency, volume, waveform):
    return tuple(zip(frequency, volume, waveform))

def shuffle_data(frequency, volume, waveform):
    combined = list(zip(frequency, volume, waveform))
    random.shuffle(combined)
    return zip(*combined)


def create_trials(frequency, patterns, volume=100, waveform="sine",
                  total_repetitions = 9,
                  zero_repetitions = 5,
                  rois = ["ROI1", "ROI2", "ROI3", "ROI4"]):
    
    # Create a list of ROIs repeated total_repetitions times
    rois_repeated = rois * total_repetitions

    # Initialize lists to store the shuffled frequency, volume, and waveform
    frequency_final = []
    wave_arrays = []
    repetition_numbers = []
    patterns_final= []

    # Total number of sound data sets
    num_sounds = len(frequency)
    previous_trials = set()

    # Create trials
    for i in range(total_repetitions):
        if i % 2 == 0:  # Alternate zero data repetitions
            for j in range(len(rois)):
                repetition_numbers.append(i + 1)  # Add repetition number
                frequency_final.append(0)
                patterns_final.append(0)
                wave_arrays.append(np.zeros(441000))  # Assuming 10 seconds of silence
        else:
            while True:
                if i == 1:  # Trial 2 should use the initial list
                    trial_list = list(zip(frequency, patterns))

                else:  # Other trials should be unique and shuffled
                    trial_list = list(zip(frequency, patterns))
                    random.shuffle(trial_list)


                # Convert trial_list to a tuple of tuples ##add wavefrom as a tuple
                trial_tuple_as_tuple = tuple((tuple(freq), tuple(pat)) for freq, pat in trial_list)
                #print(type(trial_tuple_as_tuple))

                if trial_tuple_as_tuple not in previous_trials:
                    previous_trials.add(trial_tuple_as_tuple)
                    if i == 1:
                        for j in range(len(rois)):
                            repetition_numbers.append(i + 1)  # Add repetition number
                            frequency_final.append(frequency[j])
                            patterns_final.append(patterns[j])

                            #to generate the sounds data in a way that they play continuously
                            concatenated_array = []
                            for k in range(len(frequency[j])):
                                sounds = generate_sound_data(frequency[j][k])  # Generate sound data
                                concatenated_array.append(sounds)
                            concatenated_sounds= np.concatenate(concatenated_array)
                            wave_arrays.append(concatenated_sounds)


                    else:
                        for j in range(len(rois)):
                            repetition_numbers.append(i + 1)  # Add repetition number
                            freq, pat = trial_list[j]
                            frequency_final.append(freq)
                            patterns_final.append(pat)
                            concatenated_array = []
                            
                            for k in range(len(freq)):
                                sounds = generate_sound_data(freq[k])  # Generate sound data
                                #print(trial_list[j][k])
                                concatenated_array.append(sounds)
                            concatenated_sounds= np.concatenate(concatenated_array)
                            wave_arrays.append(concatenated_sounds)



                    break

    # Create the DataFrame with the repetition numbers, repeated ROIs, and final data
    df = pd.DataFrame({
        "trial_ID": repetition_numbers,
        "ROIs": rois_repeated,
        "pattern": patterns_final,
        "frequency_seq": frequency_final,
        "wave_arrays": wave_arrays
    })

    # Add other necessary columns filled with NaNs or default values
    df["time_spent"] = [None] * len(df)
    df["visitation_count"] = [None] * len(df)
    df["trial_start_time"] = [None] * len(df)
    df["end_trial_time"] = [None] * len(df)

    return df, wave_arrays

def play_sound(sound_data, fs=44100):
    """Play sound using sounddevice library."""
    sound_data_normalised = np.int16((sound_data / np.max(np.abs(sound_data))) * 32767)
    sd.play(sound_data_normalised, fs)

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