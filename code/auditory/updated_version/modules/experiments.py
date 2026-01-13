# here we handle all things related to the trials generations

# basically anything that in version_1's supfun_sequences starts with create_ ..._trial or has info in their names

import numpy as np
import pandas as pd
import random
from typing import Tuple, List, Any, Dict, Union
from modules.audio import Audio
from config import ExperimentConfig

#this will be the structure of the output of the trial generation. A dataframe containing all the trials information + the list of the sound sound_arrays
TrialData = Tuple[pd.DataFrame, List[Any]]

class ExperimentFactory: 

    # this class will act like a "menu" that generates the correct trial structure based on the experiment mode in config.py
    # we generate the sound data that will go in the df with the functions we defined in audio.py

    @staticmethod
    def generate_trials(cfg: ExperimentConfig, audio:Audio) -> TrialData:

        generic_rois = [f"ROI{str(i+1)}" for i in range(cfg.rois_number)]
        rois_list = cfg.entrance_rois + generic_rois

        experiment_type = cfg.experiment_mode

        print(f"generating trials for {experiment_type}")

        if experiment_type == "simple_smooth":
            return ExperimentFactory._make_simple_smooth(generic_rois, cfg, audio)
        elif experiment_type == "simple_intervals":
            return ExperimentFactory._make_simple_intervals(generic_rois, cfg, audio)
        elif experiment_type == "temporal_envelope_modulation":
            return ExperimentFactory._make_temporal_envelope_modulation(generic_rois, cfg, audio)
        elif experiment_type == "complex_intervals":
            return 
        elif experiment_type == "sequences":
            return 
        elif experiment_type == "vocalisation":
            return 
        elif experiment_type == "semantic_predictive_complexity":
            return
        
    
    # logic specific to each experiment type

    @staticmethod
    def _make_simple_smooth(rois: List[str], cfg :ExperimentConfig, audio: Audio) -> TrialData:
        frequencies = [10000, 12000, 14000, 16000, 18735, 20957, 22543, 24065]

        if len(frequencies)< len(rois):
            print("not enough frequencies for ROIs. Recycling. If you want to add/modify, go to experiments.py,  _make_simple_smooth().")
            frequencies = (frequencies*2)[:len(rois)]
        
        #create trial structure
        return ExperimentFactory._create_simple_trials_logic(rois, frequencies, audio)
    
    @staticmethod
    def _make_simple_intervals(rois:List[str], cfg:ExperimentConfig, audio:Audio, manual:bool = False) -> TrialData: 
        rois_number = cfg.rois_number
        
        if manual: # if manual =True, prompt the user for the intervals
            frequency, intervals, intervals_names = ExperimentFactory._ask_info_intervals(rois_number)
        else: 
            tonal_centre = 10000 #Hz
            intervals_list = ["perf_5", "perf_4", "maj_6", "tritone", "min_2", "maj_7"]
            frequency, intervals, intervals_names = ExperimentFactory._get_info_intervals_hard_coded(rois_number, tonal_centre, intervals_list)

            return ExperimentFactory._create_intervals_trials_logic(rois, frequency, intervals, intervals_names, audio)
        
    @staticmethod
    def _make_temporal_envelope_modulation(rois:List[str], cfg:ExperimentConfig, audio:Audio) -> TrialData: 
        rois_number = cfg.rois_number

        # okay, so, this adds another layer of control. You user can choose which frequencies will be smooth, which with constant Amplitude Modulation, and which with variable AM 

        smooth_freqs = [10000, 20000]
        constant_rough_freqs = [10000, 20000]
        #constant temporal modulation
        ctemporal_modulation= 50 #Hz
        complex_rough_freqs = [10000, 20000]

        #complex temporal modulation
        complex_temporal_modulation = [30, 50, 70] # Hz
        
        controls = ["vocalisation", "silent"]

        path_to_vocalisation = cfg.path_to_vocalisation_control

        frequencies, temporal_modulation, sound_type, sound_arrays = ExperimentFactory._get_info_tem_hard_coded(rois_number,
                                                                                                controls, 
                                                                                                smooth_freqs, 
                                                                                                constant_rough_freqs, 
                                                                                                complex_rough_freqs, 
                                                                                                constant_rough_modulation= ctemporal_modulation, 
                                                                                                complex_rough_mod = complex_temporal_modulation, 
                                                                                                path_to_voc = path_to_vocalisation)
        
        return ExperimentFactory._create_tem_trials_logic(rois, frequencies, temporal_modulation, sound_type, sound_arrays, audio)
        

    @staticmethod
    def _make_complex_intervals(rois:List[str], cfg:ExperimentConfig, audio:Audio) -> TrialData: 
        experiment_day = cfg.complex_interval_day

        tonal_centre = 15000
        path_to_voc = cfg.path_to_vocalisation_control
        smooth_freq = False
        rough_freq = False
        controls = ["vocalisation", "silent"] #"vocalisation", "silent"

        #"w1day2", "w1day3", "w1day4", "another_day"
        if experiment_day == "w1day2":
            smooth_freq= True; rough_freq = True
            consonant_intervals = ["perf_5", "perf_4"] #"min_3", "maj_3", "perf_4", "perf_5", "min_6", "maj_6", "octave"
            dissonant_intervals = ["tritone",  "min_7"] # "min_2", "maj_2",  "tritone", "min_7", "maj_7"
            

        elif experiment_day =="w1day3": 
            smooth_freq= True; rough_freq = True
            consonant_intervals = ["maj_6", "min_3"] #"min_3", "maj_3", "perf_4", "perf_5", "min_6", "maj_6", "octave"
            dissonant_intervals = ["maj_7",  "min_2"] # "min_2", "maj_2",  "tritone", "min_7", "maj_7"
            
        
        elif experiment_day =="w1day4":
            consonant_intervals = ["maj_3", "perf_4", "perf_5", "min_6"] #"min_3", "maj_3", "perf_4", "perf_5", "min_6", "maj_6", "octave"
            dissonant_intervals = ["min_7",  "maj_2", "tritone", "maj_7"] # "min_2", "maj_2",  "tritone", "min_7", "maj_7"
            controls = [] #"vocalisation", "silent"

        elif experiment_day =="another_day":
            consonant_intervals = ["maj_3", "perf_4", "perf_5"] #"min_3", "maj_3", "perf_4", "perf_5", "min_6", "maj_6", "octave"
            dissonant_intervals = ["min_7",  "maj_2", "tritone"] # "min_2", "maj_2",  "tritone", "min_7", "maj_7"

        return ExperimentFactory._create_complex_intervals_trials_logic()







    



    ## create trials generators
    @staticmethod
    def _create_simple_trials_logic(rois, frequencies, audio):
        pass

    @staticmethod
    def _create_intervals_trials_logic(rois, frequency, intervals, intervals_names, audio):
        pass

    @staticmethod
    def _create_tem_trials_logic(rois, frequencies, temporal_modulation, sound_type, sound_arrays, audio):
        pass

    @staticmethod
    def _create_complex_intervals_trials_logic(rois, frequencies, intervals, intervals_names, sound_type, sound_arrays, audio):
        pass

    @staticmethod
    def _create_sequences_trials_logic():
        pass





    




    @staticmethod
    def _get_interval(interval_name: str):
        #function to retrieve numerical ratios and info from interval name

        intervals_names = ["unison", "min_2", "maj_2", "min_3", "maj_3", "perf_4", "tritone", 
                "perf_5", "min_6", "maj_6", "min_7", "maj_7", "octave"]
        intervals_values = [1/1, 16/15, 9/8, 6/5, 5/4, 4/3, 64/45, 3/2, 8/5, 5/3, 16/9, 15/8, 2]
        intervals_values_strings= ["1/1", "16/15", "9/8", "6/5", "5/4", "4/3", "45/32", "3/2", "8/5", "5/3", "16/9", "15/8", "2/1"]

        intervals = dict(zip(intervals_names, intervals_values))
        intervals_strings = dict(zip(intervals_names, intervals_values_strings))
        
        return intervals[interval_name], intervals_strings[interval_name]

    @staticmethod
    def _ask_info_intervals(rois_number: int):
            # create a dictionary with interval names and values
        
        #define intervals to get consonant and dissonant intervals
        intervals_names = ["unison", "min_2", "maj_2", "min_3", "maj_3", "perf_4", "tritone", 
                        "perf_5", "min_6", "maj_6", "min_7", "maj_7", "octave"]
        
        consonant_intervals = [intervals_names[i] for i in (0,3,4,5,7,8,9,12)]
        dissonant_intervals = [intervals_names[i] for i in (1,2,6,10,11)]
        
        print(f"You will now be prompted to select the stimuli for the {rois_number} ROIs")
        new_rois_number = rois_number

        
        # lists containing stimuli info that will be output

        frequencies = []
        interval_numerical_list = []
        interval_string_names = []

        #ask if vocalisation
        vocalisation = input("Do you want to include a vocalisation recording?(y/n)\n").strip().lower()
        if vocalisation == "y":
            frequencies.append("vocalisation")
            interval_numerical_list.append([9]) # arbitrary number so that I can identify this later on
            interval_string_names.append("vocalisation")

            
            new_rois_number -=1

        #ask if silence
        print(f"You have {new_rois_number} ROIs available")
        silence = input("do you want a Silent ROI?(y/n)\n").strip().lower()
        if silence == "y":
            frequencies.append([0,0])
            interval_numerical_list.append([0])
            interval_string_names.append("no_interval")
            new_rois_number -=1

        print(f"You have {new_rois_number} ROIs available")
        number_consonants = int("insert the number of consonant rois: \n").strip()
        new_rois_number = new_rois_number - number_consonants

        number_dissonants = new_rois_number

        print(f"your number of dissonant rois is: {number_dissonants}")

        tonal_centre = int(input("insert the frequency that will be the tonal centre:\n"))
        
        #the frequencies list will contain lists containing the 2 frequencies that make up the interval. 
        frequencies =[]
        interval_numerical_list = []
        interval_string_names = []
        
        for i in range(number_consonants):
            consonant_choice= input(f"insert the consonant interval of choice #{i+1} {consonant_intervals}:\n")
            consonant_choice= consonant_choice.lower()
            interval, interval_as_string = ExperimentFactory._get_interval(consonant_choice)
            frequencies.append([tonal_centre, int(tonal_centre*interval)])
            interval_numerical_list.append(interval_as_string)
            interval_string_names.append(consonant_choice)
        
        for i in range(number_dissonants):
            dissonant_choice= input(f"insert the dissonant interval of choice #{i+1} {dissonant_intervals}:\n")
            dissonant_choice= dissonant_choice.lower()
            interval, interval_as_string = ExperimentFactory._get_interval(dissonant_choice)
            frequencies.append([tonal_centre, int(tonal_centre*interval)])
            interval_numerical_list.append(interval_as_string)
            interval_string_names.append(dissonant_choice)
            
          
            
        return frequencies, interval_numerical_list, interval_string_names
    

    @staticmethod
     #hard code the freq and intervals list not to manually be prompted every time
    def _get_info_intervals_hard_coded(rois_number, tonal_centre, intervals_list):
        
        #usable_rois exclude the unison and silent arm
        usable_rois = rois_number - 2

        tonal_centre_interval, tonal_centre_string = ExperimentFactory._get_interval("unison")
        #the frequencies list will contain lists containing the 2 frequencies that make up the interval. 
        frequencies =[[tonal_centre, int(tonal_centre*tonal_centre_interval)]]
        interval_numerical_list = [tonal_centre_string]
        interval_string_names = ["unison"]


        if len(intervals_list) == usable_rois:

            for i in range(usable_rois):

        #check if we want a vocalisation as one of the inputs, if not, continues as normal
                if i != "vocalisation":
                    interval, interval_as_string = ExperimentFactory._get_interval(intervals_list[i])
                    frequencies.append([tonal_centre, int(tonal_centre*interval)])
                    interval_numerical_list.append(interval_as_string)
                    interval_string_names.append(intervals_list[i])

        # if there is a vocalisation, it retrieves the path from the main script and checks if the sample rate is correct
                else: 
                    frequencies.append("vocalisation")
                    interval_numerical_list.append([9]) # arbitrary number so that I can identify this later on
                    interval_string_names.append("vocalisation")


        # appends the silent frequency
            frequencies.append([0,0])
            interval_numerical_list.append(["0"])
            interval_string_names.append("no_interval")
        else:
            print("please check that the number of intervals is rois_number - 2")
        
        return frequencies, interval_numerical_list, interval_string_names
    
    @staticmethod
    def _get_info_tem_hard_coded(rois_number, controls, smooth_freqs, constant_rough_freqs, complex_rough_freqs, constant_rough_modulation = 50, complex_rough_mod = [30,50,70], audio = Audio, cfg = ExperimentConfig):
        freqs = controls + smooth_freqs + constant_rough_freqs + complex_rough_freqs
        frequencies = []
        temporal_modulation = []
        sound_type = []
        sound_arrays = []

        if len(freqs) != rois_number:
            print("Bestie, double check the number of stimuli and make sure they match the number of rois")

        path_to_voc = cfg.path_to_vocalisation_control
        
        for item in controls:
            if item == "silent":
                frequencies.append("silent_arm"); temporal_modulation.append("no_stimulus"); sound_type.append("control")
                sound_arrays.append(np.zeros(int(audio.fs * audio.default_duration)))
            else:
                frequencies.append("vocalisation"); temporal_modulation.append("vocalisation"); sound_type.append("control")
                sound_arrays.append(audio.load_wav(path_to_voc))
        
        for f in smooth_freqs:
            frequencies.append(f); temporal_modulation.append("none"); sound_type.append("smooth")
            sound_arrays.append(audio.generate_sound_data(f))
            
        for f in constant_rough_freqs:
            frequencies.append(f); temporal_modulation.append(constant_rough_modulation); sound_type.append("rough")
            # Generate AM tone
            sound_arrays.append(audio.generate_simple_tem_sound_data(f, modulated_frequency=constant_rough_modulation))
            
        for f in complex_rough_freqs:
            frequencies.append(f); temporal_modulation.append(complex_rough_mod); sound_type.append("rough_complex")
            # Generate Complex AM
            sound_arrays.append(audio.generate_complex_tem_sound_data(f, modulated_frequencies_list=complex_rough_mod))
            
        return frequencies, temporal_modulation, sound_type, sound_arrays


    @staticmethod
    def _get_info_complex_intervals_hard_coded(rois_number, controls, tonal_centre, smooth_freq, rough_freq, consonant_intervals, dissonant_intervals, audio = Audio, cfg = ExperimentConfig):

        all_intervals = consonant_intervals + dissonant_intervals
        
        #the frequencies list will contain lists containing the 2 frequencies that make up the interval. 
        frequencies =[]; interval_numerical_list = []; interval_string_names = []; sound_type = []; sounds_arrays = []

        for i in controls:
            interval_numerical_list.append(0)
            interval_string_names.append(i)          
            sound_type.append(i)

            if i == "silent":
                frequencies.append(0)
                # 10 s of silence at 192 kHz
                z = np.zeros(int(192000 * 10))
                sounds_arrays.append([z, z])
            else: 
                frequencies.append(i)
                voc = generate_voc_array(path_to_voc, 192000)
                silence = np.zeros_like(voc)
                sounds_arrays.append([voc, silence])

        if smooth_freq:
            tonal_centre_interval, tonal_centre_string = ExperimentFactory._get_interval("unison") 
            frequencies.append([tonal_centre, int(tonal_centre*tonal_centre_interval)])
            interval_numerical_list.append(tonal_centre_string)
            interval_string_names.append("unison")
            sound_data_1= generate_sound_data(tonal_centre)
            sound_data_2= generate_sound_data(int(tonal_centre*tonal_centre_interval))

            sound_type.append("smooth")
            sounds_arrays.append([sound_data_1, sound_data_2])

        if rough_freq:

            tonal_centre_interval, tonal_centre_string = get_interval("unison")

            t_1, sound_data_1 = generate_sound_data(tonal_centre, give_t = True)
            t_2, sound_data_2 = generate_sound_data(int(tonal_centre*tonal_centre_interval), give_t = True)
            
            frequencies.append([tonal_centre, int(tonal_centre*tonal_centre_interval)])
            interval_numerical_list.append(tonal_centre_string)
            interval_string_names.append("unison")
            sound_type.append("rough")
            modulated_wave_1 = apply_constant_sinusoidal_envelope(t_1, sound_data_1)
            modulated_wave_2 = apply_constant_sinusoidal_envelope(t_2, sound_data_2)
            sounds_arrays.append([modulated_wave_1, modulated_wave_2])

        for i in all_intervals: 
            interval, interval_string = get_interval(i)
            freq_1 = tonal_centre
            freq_2 = tonal_centre*interval

            frequencies.append([freq_1, freq_2])
            interval_numerical_list.append(interval_string)
            interval_string_names.append(i)

            sound_1= generate_sound_data(tonal_centre)
            sound_2 = generate_sound_data(freq_2)
            
            sounds_arrays.append([sound_1, sound_2])

            if i in consonant_intervals: 
                sound_type.append("consonant")
            else:
                sound_type.append("dissonant")
            


        return frequencies, interval_numerical_list, interval_string_names, sound_type, sounds_arrays

                
            

        

            



            





