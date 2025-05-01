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
from sympy import *
from sympy.plotting import plot_parametric
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d



def get_rois_list(rois_number):
    rois_list = []
    for i in range(rois_number):
        rois_list.append(f"ROI{i+1}")
    return rois_list

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

def read_wav_file(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        # Extract parameters
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        framerate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        
        # Read frames and convert to numpy array
        frames = wav_file.readframes(n_frames)
        sound_data = np.frombuffer(frames, dtype=np.int16)
        
        # If stereo, reshape to two columns
        if n_channels == 2:
            sound_data = sound_data.reshape(-1, 2)
        
        return sound_data, framerate

def get_interval(interval_name):

    intervals_names = ["unison", "min_2", "maj_2", "min_3", "maj_3", "perf_4", "tritone", 
             "perf_5", "min_6", "maj_6", "min_7", "maj_7", "octave"]
    intervals_values = [1/1, 16/15, 9/8, 6/5, 5/4, 4/3, 64/45, 3/2, 8/5, 5/3, 16/9, 15/8, 2]
    intervals_values_strings= ["1/1", "16/15", "9/8", "6/5", "5/4", "4/3", "45/32", "3/2", "8/5", "5/3", "16/9", "15/8", "2/1"]

    intervals = dict(zip(intervals_names, intervals_values))
    intervals_strings = dict(zip(intervals_names, intervals_values_strings))
    
    return intervals[interval_name], intervals_strings[interval_name]

def get_chord(chord):
    chords_dict= {
        "major": ["unison", "maj_3", "perf_5"],
        "minor": ["unison", "min_3", "perf_5"],
        "maj_6th": ["unison", "maj_3", "perf_5", "maj_6"],
        "min_6th": ["unison", "min_3", "perf_5", "maj_6"],
        "sus4": ["unison", "perf_4", "perf_5"],
        "dim": ["unison", "min_3", "tritone"],
        "aug": ["unison", "maj_3", "min_6"],
        "dominant_7th": ["unison", "maj_3", "perf_5", "min_7"],
        "min_7th": ["unison", "min_3", "perf_5", "min_7"],
        "maj_7th": ["unison", "maj_3", "perf_5", "maj_7"]
    }
    
    return chords_dict[chord]

def get_intervals_from_chord(root_frequency, chord):
    sequence_of_intervals = get_chord(chord)

    interval_values = []
    interval_strings = []
    frequencies = []
    for i in sequence_of_intervals: 
        interval_value, interval_string = get_interval(i)
        interval_values.append(interval_value)
        interval_strings.append(interval_string)
        frequencies.append(int(interval_value * root_frequency))

    return sequence_of_intervals, interval_strings, frequencies

def check_if_lower(pattern):
    lower = ["random", "RANDOM", "Random", "ran", "rdm", "Ran", "RDM", "silence", "SILENCE", "sil", "Sil", "Silence"]
    mixed = ["AoAo", "AOAO", "aoao", "A0A0", "a0a0"]
    
    if pattern in mixed: 
        pattern = "AoAo"
    else:
        if pattern in lower:
            pattern = pattern.lower()
        else:
            pattern = pattern.upper()
    return pattern

def ask_pattern(rois_number):
    pattern_list= []
    for i in range(1, rois_number+1):        
        pattern = input(f"insert pattern #{i}\n(e.g. AoAo, ABAB, ABCABC, ABCDABCD, random, silence, BABA, CBACBA)\n")
        pattern = check_if_lower(pattern)
        pattern_list.append(pattern)
    return (pattern_list)

def ask_pattern_chords(number_consonants, number_dissonants):
    consonant_patterns = []
    dissonant_patterns= []
    
    if number_consonants == number_dissonants: 
        for i in range(1, number_consonants+1):
            pattern = (input(f"insert pattern #{i}\n(ABAB, ABCABC, ABCDABCD, random, BABA, CBACBA):\n"))
            pattern = check_if_lower(pattern)
            consonant_patterns.append(pattern)
            dissonant_patterns.append(pattern)
            
    elif number_consonants >  number_dissonants : 
        difference= number_consonants -  number_dissonants
        same_n_rois = number_consonants - (difference)

        for i in range(1, same_n_rois+1):
            pattern = (input(f"insert pattern #{i}\n(ABAB, ABCABC, ABCDABCD, random, BABA, CBACBA):\n"))
            pattern = check_if_lower(pattern)
            consonant_patterns.append(pattern)
            dissonant_patterns.append(pattern)

        for i in range(1, difference+1):
            pattern = (input(f"insert the additional consonant pattern #{i}\n(ABAB, ABCABC, ABCDABCD, random, BABA, CBACBA):\n"))
            pattern = check_if_lower(pattern)
            consonant_patterns.append(pattern)

    else: 
        difference=  number_dissonants - number_consonants
        same_n_rois =  number_dissonants - (difference)

        for i in range(1, same_n_rois+1):
            pattern = (input(f"insert pattern #{i}\n(ABAB, ABCABC, ABCDABCD, random, BABA, CBACBA):\n"))
            pattern = check_if_lower(pattern)
            consonant_patterns.append(pattern)
            dissonant_patterns.append(pattern)

        for i in range(1, difference+1):
            pattern = (input(f"insert the additional dissonant pattern #{i}\n(ABAB, ABCABC, ABCDABCD, random, BABA, CBACBA):\n"))
            pattern = check_if_lower(pattern)
            dissonant_patterns.append(pattern)
            
    return consonant_patterns, dissonant_patterns

def ask_music_info_sequences(rois_number):

    intervals_vs_custom= input("would you like to add custom values or to generate sequences based on intervals? (custom/intervals)")

    if intervals_vs_custom == "custom":
        pattern_list = []
        
        #for each ROI, ask the user for the desired pattern
        ask_for_pattern_input = (input("would you like to make new patterns?\n (select y if you want to insert the patterns, n if you want to hard code the patterns)")).lower()
        
        if ask_for_pattern_input == "y":
            pattern_list= ask_pattern(rois_number)
        else:
            pattern_list = ['BABA','AoAo','random', 'silence',  'AAAAA', 'ABCABC', 'BABA', 'CBACBA', 'ABCDEABCDE', 'DCBADCBA']   
        
        #to determine the individual events, exclude random and silence
        patterns_nonpatterns = ["random", "silence"]
        
        #get the sorted individual events in the sequences and map them to frequencies
        events = []
        freqs = []
        for i in sorted(pattern_list):
            if i not in patterns_nonpatterns: 
                for j in i:
                    if j not in events:
                        events.append(j)
                        #ask the user the frequency for the event
                        if j !='o':
                            freq = int(input(f"Insert frequency for sound {j}:\n"))
                            freqs.append(freq)
                        else:
                            freqs.append(0)

        
        #map frequency to event in a dictionary
        sound_dict = dict(zip(events,freqs))
        
        sequence_from_pattern= []
        repetitions = 50
        #generate the sequences of events from patterns 
        for index, item in enumerate(pattern_list):
            #print (index+1, item)
            if item not in patterns_nonpatterns: 
                #print (index, item)
                sequence_from_pattern.append(list(item*repetitions))
            elif item == "random": 
                sequence_from_pattern.append([random.choice(events) for _ in range(200)])
            else:
                sequence_from_pattern.append(['o' for _ in range(200) ])
            
    #print(sequence_from_pattern)
        sequence_of_frequencies = [[sound_dict[char] for char in sublist] for sublist in sequence_from_pattern]

    elif intervals_vs_custom == "intervals":
        pass

    #
    #
    #
    #
    #
    #
    #
    #
    #
    ###continue with the intervals patterns one #### 

    return sequence_of_frequencies, pattern_list
    

def ask_music_info_simple_sounds(rois_number):
    frequency = []

    for i in range(1, rois_number+1): 
        freqs = int(input(f"Insert frequency for sound #{i}\n"))
        frequency.append(freqs)

    return frequency

def ask_info_intervals(rois_number):
    
    # create a dictionary with interval names and values
        
    #define intervals to get consonant and dissonant intervals
    intervals_names = ["unison", "min_2", "maj_2", "min_3", "maj_3", "perf_4", "tritone", 
                       "perf_5", "min_6", "maj_6", "min_7", "maj_7", "octave"]
    
    consonant_intervals = [intervals_names[i] for i in (3,4,5,7,8,9,12)]
    dissonant_intervals = [intervals_names[i] for i in (1,2,6,10,11)]
    
    

    usable_rois = rois_number - 2
    #one is going to be unison, the other one is going to be silence
    rois_per_nance= int(usable_rois/2)
    
    number_consonants = rois_per_nance
    number_dissonants = rois_per_nance
    
    if rois_number %2 !=0:
        
        choice_odd_roi = (input('''you have an odd number of ROIs.\nWould you like to have an extra consonant or dissonant ROI?\n''')).lower()
        if choice_odd_roi == "consonant":
            number_consonants +=1
            
        elif choice_odd_roi == "dissonant":
            number_dissonants +=1

            
    tonal_centre = int(input("insert the frequency that will be the tonal centre:\n"))
    tonal_centre_interval, tonal_centre_string = get_interval("unison")
    
    #the frequencies list will contain lists containing the 2 frequencies that make up the interval. 
    frequencies =[[tonal_centre, int(tonal_centre*tonal_centre_interval)]]
    interval_numerical_list = [tonal_centre_string]
    interval_string_names = ["unison"]
    
    
    
    for i in range(number_consonants):
        consonant_choice= input(f"insert the consonant interval of choice {consonant_intervals}:\n")
        consonant_choice= consonant_choice.lower()
        interval, interval_as_string = get_interval(consonant_choice)
        frequencies.append([tonal_centre, int(tonal_centre*interval)])
        interval_numerical_list.append(interval_as_string)
        interval_string_names.append(consonant_choice)
    
    for i in range(number_dissonants):
        dissonant_choice= input(f"insert the dissonant interval of choice {dissonant_intervals}:\n")
        dissonant_choice= dissonant_choice.lower()
        interval, interval_as_string = get_interval(dissonant_choice)
        frequencies.append([tonal_centre, int(tonal_centre*interval)])
        interval_numerical_list.append(interval_as_string)
        interval_string_names.append(dissonant_choice)
        
        #ask user if there will be a vocalisation.
    vocalisation = (input("want to select a .wav file containing a sound vocalisation?(y/n)\nif n, one of the ROIs will be silent \n")).lower()
    
    if vocalisation == "y":
        wav_file = choose_csv()
        frequencies.append([wav_file, 0])
        interval_numerical_list.append(0)
        interval_string_names.append("vocalisation")

    else:
        frequencies.append([0,0])
        interval_numerical_list.append(["0"])
        interval_string_names.append("no_interval")
        
        
    return frequencies, interval_numerical_list, interval_string_names
    
def plotting_lissajous(interval):
    t = symbols('t')
    
    intervals_names = ["unison", "min_2", "maj_2", "min_3", "maj_3", "perf_4", "tritone", 
             "perf_5", "min_6", "maj_6", "min_7", "maj_7", "octave"]

    intervals_values = ["1/1", "16/15", "9/8", "6/5", "5/4", "4/3", "45/32", "3/2", "8/5", "5/3", "16/9", "15/8", "2/1"]


    intervals = dict(zip(intervals_names, intervals_values))
    
    if len(intervals[interval])==5: 
        a= int(intervals[interval][0:2])
        b= int(intervals[interval][3:5])
    elif len(intervals[interval])==3:
        a= int(intervals[interval][0])
        b= int(intervals[interval][2])
        
    
    
    #s = 0 because the interval (a,b) plays at the same time
    s = 0 #pi/2


    title = f'a = {a}, b = {b}'
    x = sin(a*t + s)
    y = sin(b*t)
    p = plot_parametric(x, y, (t, 0, 2*pi), title=title)
    
    p.save(f"{interval}_plot.png")    

def compute_gain(frequency_hz):

    data = pd.read_csv("C:/Users/aleja/Downloads/frequency_response_speaker.csv")
    # interpolation function from the frequency response data extrapolated from the graph on the website
    frequency = data['Frequency_kHz'].values * 1000  # Convert kHz to Hz
    attenuation = data['Attenuation_dB'].values
    interp_func = interp1d(frequency, attenuation, kind='cubic', fill_value="extrapolate")
    #get the interpolated attenuation
    attenuation_db = interp_func(frequency_hz)
    #from there calculate the gain 
    gain = 10 ** (-attenuation_db / 20)
    return gain


    

 

def generate_sound_data(frequency, waveform="sine", duration=15, fs=192000, volume=1.0, compensate=True):
    """Generate sound data for a given frequency and waveform."""
    t = np.linspace(0, duration, int(fs * duration), endpoint = False)
    gain = compute_gain(frequency) if compensate else 1.0
    adjusted_volume = volume * gain
    sound = np.zeros_like(t)
    


    if waveform == 'sine':
        sound = np.sin(frequency * t * 2 * np.pi)  * adjusted_volume

    elif waveform == 'square':
        sound = np.sign(np.sin(frequency * t * 2 * np.pi)) * adjusted_volume

    elif waveform == 'sawtooth':
        sound = (2 * (t * frequency % 1) - 1) * adjusted_volume

    elif waveform == 'triangle':
        sound = (2 * np.abs(2 * (t * frequency - np.floor(0.5 + t * frequency))) - 1) * adjusted_volume

    elif waveform == 'pulse wave':
        sound = np.where((t % (1 / frequency)) < (1 / frequency) * 0.5, 1.0, -1.0)  * adjusted_volume  # 0.5 in this case is the duty cycle
    
    elif waveform == 'white noise':
        samples = int(fs * duration)
        sound = np.random.uniform(low=-1.0, high=1.0, size=samples) * adjusted_volume


    return sound

def get_trial_tuple(frequency, volume, waveform):
    return tuple(zip(frequency, volume, waveform))

def shuffle_data(frequency, volume, waveform):
    combined = list(zip(frequency, volume, waveform))
    random.shuffle(combined)
    return zip(*combined)

def create_simple_trials(rois, frequency,
                  total_repetitions = 9,
                  zero_repetitions = 5,
                  sample_rate = 192000):

    
    # Create a list of ROIs repeated total_repetitions times
    rois_repeated = rois * total_repetitions

    # Initialize lists to store the shuffled frequency, volume, and waveform
    frequency_final = []
    # volume_final = []
    # waveform_final = []
    wave_arrays = []
    repetition_numbers = []

    # Total number of sound data sets
    num_sounds = len(frequency)
    previous_trials = set()

    

    # Create trials
    for i in range(total_repetitions):
        if i % 2 == 0:  # Alternate zero data repetitions
            for j in range(len(rois)):
                repetition_numbers.append(i + 1)  # Add repetition number
                frequency_final.append(0)
                # volume_final.append(0)
                # waveform_final.append('none')
                temp_array = np.zeros(sample_rate*10)
                wave_arrays.append(np.zeros(sample_rate*10))  # Assuming 10 seconds of silence
        else:
            while True:
                if i == 1:  # Trial 2 should use the initial list
                    trial_tuple = tuple(frequency)  #get_trial_tuple(frequency, volume, waveform)
                else:  # Other trials should be unique and shuffled 
                    trial_list = list(frequency)
                    random.shuffle(trial_list)
                    trial_tuple = tuple(trial_list)
                    


                if trial_tuple not in previous_trials:
                    previous_trials.add(trial_tuple)
                    if i == 1:
                        for j in range(len(rois)):
                            repetition_numbers.append(i + 1)  # Add repetition number
                            frequency_final.append(frequency[j])
                            # volume_final.append(volume[j])
                            # waveform_final.append(waveform[j])
                            wave_arrays.append(generate_sound_data(frequency[j]))
                    else:
                        for j in range(len(rois)):
                            repetition_numbers.append(i + 1)  # Add repetition number
                            frequency_final.append(trial_tuple[j])
                            # volume_final.append(shuffled_volume[j])
                            # waveform_final.append(shuffled_waveform[j])
                            wave_arrays.append(generate_sound_data(trial_tuple[j]))
                    break

    # Create the DataFrame with the repetition numbers, repeated ROIs, and final data
    df = pd.DataFrame({
        "trial_ID": repetition_numbers,
        "ROIs": rois_repeated,
        "frequency": frequency_final,
        # "volume": volume_final,
        # "waveform": waveform_final,
        "wave_arrays": wave_arrays
    })

    # Add other necessary columns filled with NaNs or default values
    df["time_spent"] = [None] * len(df)
    df["visitation_count"] = [None] * len(df)
    df["trial_start_time"] = [None] * len(df)
    df["end_trial_time"] = [None] * len(df)
    

    return df,wave_arrays

def create_trials_for_sequences(rois, frequency, patterns, volume=100, waveform="sine",
                  total_repetitions = 9,
                  zero_repetitions = 5,
                  sample_rate = 192000):
    
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
                wave_arrays.append(np.zeros(sample_rate*10))  # Assuming 10 seconds of silence
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
        "frequency": frequency_final,
        "wave_arrays": wave_arrays
    })

    # Add other necessary columns filled with NaNs or default values
    df["time_spent"] = [None] * len(df)
    df["visitation_count"] = [None] * len(df)
    df["trial_start_time"] = [None] * len(df)
    df["end_trial_time"] = [None] * len(df)

    return df, wave_arrays

def create_trials_for_intervals(rois, frequency, intervals, intervals_names,
                  total_repetitions = 9,
                  zero_repetitions = 5,
                  sample_rate=192000):

    # Create a list of ROIs repeated total_repetitions times
    rois_repeated = rois * total_repetitions

    # Initialize lists to store the shuffled frequency, volume, and waveform
    frequency_final = []
    intervals_final = []
    intervals_names_final = []
    wave_arrays = []
    repetition_numbers = []


    # Total number of sound data sets
    num_sounds = len(frequency)
    previous_trials = set()

    #generate the sounds and create an array containing 4 lists, each containing 2 arrays containing each sound
    dual_array_sounds = []
    for i in frequency: 

        frequencies_sound_data = []
        #run the loop for the first 3 items that definitely contain a frequency
        if i[1] != 0:
            for j in i:
                sound = generate_sound_data(j)
                frequencies_sound_data.append([sound])

    #     #now the fourth item always has i[1] == 0
        if i[1] == 0 :
            if i[0] == 0:
                #if the first item of the 4th array is 0, they are both 0s and we can proceed 
                for j in i:
                        sound = generate_sound_data(j)
                        frequencies_sound_data.append([sound])

            #if in the 4th item, the first element of the array is not 0, we need to convert the .wav into a sound data array
            else:
                sound_data, framerate = read_wav_file(i[0])
                sound1 = generate_sound_data(sound_data, framerate)
                frequencies_sound_data.append([sound1])
                sound2 = generate_sound_data(i[1])
                frequencies_sound_data.append([sound2])

        dual_array_sounds.append(frequencies_sound_data)

    # Create trials
    #first trial is going to be silent 
    #i is the trial_ID
    for i in range(total_repetitions):
        if i % 2 == 0:  # Alternate zero data repetitions
            for j in range(len(rois)):
                repetition_numbers.append(i + 1)  # Add repetition number
                frequency_final.append(0)
                intervals_final.append(0)
                intervals_names_final.append(0)
                wave_arrays.append(np.zeros(sample_rate*10))  # Assuming 10 seconds of silence
        else:
            while True:
                if i == 1:  # Trial 2 should use the initial list
                    trial_list = list(zip(frequency, intervals, intervals_names, dual_array_sounds))

                    #print(f"trial 1: {trial_list}")


                else:  # Other trials should be unique and shuffled
                    trial_list = list(zip(frequency, intervals, intervals_names, dual_array_sounds))
                    random.shuffle(trial_list)
                    #print(f"trial {i}: {trial_list}")



                # Convert trial_list to a tuple of tuples ##add wavefrom as a tuple
                trial_tuple_as_tuple = tuple(item[2] for item in trial_list)
                #print(trial_tuple_as_tuple)

                #trying to convert the whole list of freq, int, wave requires too much, try to see if just affing the freq works

                #try to use nested lists instead
                #print(type(trial_list))
                if trial_tuple_as_tuple not in previous_trials:
                    previous_trials.add(trial_tuple_as_tuple)
                    if i == 1:
                        for j in range(len(rois)):
                            repetition_numbers.append(i + 1)  # Add repetition number
                            frequency_final.append(frequency[j])
                            intervals_final.append(intervals[j])
                            intervals_names_final.append(intervals_names[j])
                            wave_arrays.append(tuple(dual_array_sounds[j]))


                    else:
                        for j in range(len(rois)):
                            repetition_numbers.append(i + 1)  # Add repetition number
                            freq, inter, inter_name, wave = trial_list[j]
                            frequency_final.append(freq)
                            intervals_final.append(inter)
                            intervals_names_final.append(inter_name)
                            wave_arrays.append(tuple(wave))



                    break

    # Create the DataFrame with the repetition numbers, repeated ROIs, and final data
    df = pd.DataFrame({
        "trial_ID": repetition_numbers,
        "ROIs": rois_repeated,
        "interval": intervals_names_final,
        "interval_ratio": intervals_final,
        "frequency": frequency_final,
        "wave_arrays": wave_arrays
    })

    # Add other necessary columns filled with NaNs or default values
    df["time_spent"] = [None] * len(df)
    df["visitation_count"] = [None] * len(df)
    df["trial_start_time"] = [None] * len(df)
    df["end_trial_time"] = [None] * len(df)

    return df, wave_arrays

def play_sound(sound_data, fs=192000):
    """Play sound using sounddevice library."""
    sd.play(sound_data, fs)

def play_interval(sound_data1, sound_data2, fs=44100):
    """Play two sounds simultaneously using the sounddevice library."""
    # Ensure both sound_data arrays are the same length
    min_length = min(len(sound_data1), len(sound_data2))
    sound_data1 = sound_data1[:min_length]
    sound_data2 = sound_data2[:min_length]
    
    # Sum the two sound data arrays
    combined_sound_data = sound_data1 + sound_data2
    
    # Normalize the combined sound data
    combined_sound_data_normalised = np.int16((combined_sound_data / np.max(np.abs(combined_sound_data))) * 32767)
    
    # Play the combined sound
    sd.play(combined_sound_data_normalised, fs)

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
