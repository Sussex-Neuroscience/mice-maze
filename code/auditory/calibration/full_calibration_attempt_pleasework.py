import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.io import wavfile
from scipy.fftpack import fft
from scipy.signal import get_window
import tkinter as tk
from tkinter import filedialog as fd

import fft_volume as fv
import find_amplitude as fa


# File selection
choose_file = False
SPEED_UP_FFT = True

#set to false to visualise when the signals happen and define the range of time 
time_range_found = False
if choose_file:
    INPUT_FILE = fv.choose_csv()
else:
    for i in range (1,5):
      
        INPUT_FILE = f"C:/Users/aleja/OneDrive/Desktop/mice maze/recordings_calibration/latest_{i}.wav"
  
        #check the spectrogram and normalised fft to determine the time window to extract the amplitude
        SPEED_RATE = 5
        if not SPEED_UP_FFT:
            SPEED_RATE = 1

        #if the time values have been found
            if not time_range_found:
                fv.spectrogram_n_fft(INPUT_FILE, SPEED_RATE)


       # calibration_constant = find_calibration_constant()



        def find_c_for_all (freq, start_time, end_time, spls):

            Amps = []
            calibration_constants_lists = []
            for i in range(len(freq)):
                amp, amp_raw = fa.find_amplitude(INPUT_FILE, start_time[i], end_time[i], 8192, freq[i])
                cc=fa.find_calibration_constant(spls[i], amp)

                Amps.append(amp)
                calibration_constants_lists.append(cc)

            return calibration_constants_lists #, Amps


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


        for i in range(1,5):
            if str(i) in INPUT_FILE:
                specific_times = times[i-1]
                start_time = specific_times[:,0]
                end_time = specific_times[:,1]

                start_time = [float(i) for i in start_time]
                end_time = [float(i) for i in end_time]

                spls = spls_for_all[:, i-1]


            


        print(find_c_for_all (freq, start_time, end_time, spls))



        #this function finds the raw volumes that will have to be adjusted to the mic sensitivity 
        #def find_raw_volumes(freq, start_time, end_time, calibration_constant): 

        #     #check that they are all the same length: 
        # if len(freq) == len(start_time) and len(start_time) == len(end_time):

        #     #Lraw(f) = 20*np.log10(A(f)) + C
        #     Lraws = []
        #     Amps = []
        #     for i in range(len(freq)):
        #         amp, amp_raw = fa.find_amplitude(INPUT_FILE, start_time[i], end_time[i], 8192, freq[i])
        #         Lraw = 20*np.log10(amp) + calibration_constant
        #         Lraws.append(Lraw)
        #         Amps.append(amp)


        #     print(freq)
        #     print(Amps)
        #     print(Lraws)


            
        # else: 
        #     raise Exception("the frequencies in the array need to match number of start time and end time")
















