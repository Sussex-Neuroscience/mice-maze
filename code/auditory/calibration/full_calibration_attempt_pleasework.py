import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.io import wavfile
from scipy.fftpack import fft
from scipy.signal import get_window
import tkinter as tk
from tkinter import filedialog as fd
import pandas as pd
import fft_volume as fv



# File selection
choose_file = False
SPEED_UP_FFT = True
test1 = False
test2 = True
found_c = False

#set to false to visualise when the signals happen and define the range of time 
time_range_found = False

#set to 

def main():
    if choose_file:
        INPUT_FILE = fv.choose_csv()
        fv.find_time_range(INPUT_FILE)
    elif test1:
        for i in range (1,5):
            INPUT_FILE = f"C:/Users/aleja/OneDrive/Desktop/mice maze/recordings_calibration/latest_{i}.wav"
            freq = [15000, 16000, 17000, 18000, 19000, 20000]

            #every row stands for the volumes dbSPL per frequencies (aka row 0 is all the trials for freq 15kHz)
            spls_for_all= np.array([[68, 66, 67, 67.8],
                            [65.4, 64.8, 66, 66],
                            [63.3, 63.8, 63.8, 63.5],
                            [58.3, 59.4, 60.2, 60],
                            [55, 57.1, 56.8, 56],
                            [55, 54.7, 55, 54.8]])


            times = np.array([[[41,52], [57, 68], [74, 84], [92,100], [106, 115], [123, 135]],
                    [[27,40], [54,57], [62,74], [76,90], [97,107], [112,123]],
                    [[60,73], [77,91], [93,106], [110,123], [126, 140], [144,150]], 
                    [[40,52], [55,69], [72,85], [88,101], [107,119], [124,135]]])
            
            
            start_time, end_time, spls = fv.get_start_end_times(INPUT_FILE, freq, spls_for_all, times)
            calibration_constants_lists , Amps= fv.find_c_for_all (INPUT_FILE, freq, start_time, end_time, spls)

  
    elif test2:
        freq_list = []
        start_time_list = []
        end_time_list = []
        input_list = []
        for i in range (1,7):

            INPUT_FILE = f"C:/Users/aleja/OneDrive/Desktop/mice maze/recordings_calibration/pre_Malawi_{i}.wav"

            #get times of events, and visually check that everything is okay 
            start_times, end_times, freqs = fv.find_time_range(INPUT_FILE, show_plot = False)
            start_time_list.append(start_times)
            end_time_list.append(end_times)
            freq_list.append(freqs)
            input_list.append(INPUT_FILE)
            
            variables_sorted= sorted(zip(freq_list, input_list, start_time_list, end_time_list))
            

            files_sorted = []
            freq_sorted = []
            start_time_sorted = []
            end_time_sorted = []

            for i in range(len(freq_list)):
                
                freq_sorted.append(variables_sorted[i][0])
                files_sorted.append(variables_sorted[i][1])
                start_time_sorted.append(variables_sorted[i][2])
                end_time_sorted.append(variables_sorted[i][3])

            spls_for_all= np.array([67.98, 68.95, 67.32, 71.38, 67.77, 66.31, 65.66])

        #sorted frequencies        
        freq, input_list, start_time, end_time = freq_sorted, files_sorted, start_time_sorted, end_time_sorted 

        
        freq, amp, amp_raw, c_per_freq, c_per_freq_raw = find_c_2(freq, input_list, spls_for_all, start_time, end_time)
    
        print(freq, amp, amp_raw, c_per_freq, c_per_freq_raw)


        #Lraw(f) = 20*np.log10(A(f)) + C
        calibration_constant = -24
        Lraws = []
        Amps = []
        for i in range(len(freq)):
            Lraw = 20*np.log10(amp[i]) + calibration_constant
            Lraws.append(Lraw)
            Amps.append(amp)

        print(Lraws, Amps)








    


# calibration_constant = find_calibration_constant()


def find_c_2(freq, input_list, spls_for_all, start_time, end_time):
    c_per_freq = []
    c_per_freq_raw = []
    amp_avgd = []
    amp_raw_avgd = []
    for i in range(len(input_list)):
    
        if len(start_time[i])== len(end_time[i]):
            # print (i-1, freq[i-1], start_time[i-1], end_time[i], len(start_time[i-1]), len(end_time[i-1]))
            c_per_time = []
            c_per_time_raw =[]
            amp_per_time = []
            amp_raw_per_time = []
            for j in range(len(start_time[i])):
                amp, amp_raw = fv.find_amplitude(input_list[i], start_time[i][j], end_time[i][j], 8192, freq[i])
                cc= fv.find_calibration_constant(spls_for_all[i], amp)
                cc1= fv.find_calibration_constant(spls_for_all[i], amp_raw)
                

                print(freq[i],cc, cc1, spls_for_all[i], amp, amp_raw)
        
                c_per_time.append(cc)
                c_per_time_raw.append(cc1)
                amp_per_time.append(amp)
                amp_raw_per_time.append(amp_raw)

            avg_c = sum(c_per_time)/ len(c_per_time)
            avg_c_raw = sum(c_per_time_raw)/ len(c_per_time_raw)
            avg_amp = sum(amp_per_time) / len(amp_per_time)
            avg_raw_amp = sum(amp_raw_per_time) / len(amp_raw_per_time)




            amp_avgd.append(avg_amp)
            amp_raw_avgd.append(amp_raw_avgd)
            c_per_freq.append(avg_c)
            c_per_freq_raw.append(avg_c_raw)
    
    return freq, amp_avgd, amp_raw_avgd, c_per_freq, c_per_freq_raw








#this function finds the raw volumes that will have to be adjusted to the mic sensitivity 
#def find_raw_volumes(freq, start_time, end_time, calibration_constant): 
def find_raw_volumes(freq, start_time, end_time, calibration_constant, INPUT_FILE):
        #check that they are all the same length: 
    if len(freq) == len(start_time) and len(start_time) == len(end_time):

        #Lraw(f) = 20*np.log10(A(f)) + C
        Lraws = []
        Amps = []
        for i in range(len(freq)):
            amp, amp_raw = fv.find_amplitude(INPUT_FILE, start_time[i], end_time[i], 8192, freq[i])
            Lraw = 20*np.log10(amp) + calibration_constant
            Lraws.append(Lraw)
            Amps.append(amp)


        print(freq)
        print(Amps)
        print(Lraws)


        
    else: 
        raise Exception("the frequencies in the array need to match number of start time and end time")




if __name__ == "__main__":
    main()











