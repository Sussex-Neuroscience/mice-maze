'''Here you get to define/modify the experimental variables'''

from dataclasses import dataclass, field
from typing import List, Optional 

@dataclass

class ExperimentConfig: 

    

    # Audio settings 
    samplerate: int = 192000
    channel_id: int= 3

    # Arduino settings
    arduino_port: str = "COM4"
    arduino_baud: int = 11520
    use_microcontroller: bool = False # set to True if we are going to use an arduino (either for reward delivery or for photometry TTL)

    # Camera settings
    video_input: int = 0 #0 is usually the webcam, in case of multiple webcams and you want to pick the second, insert 1, otherwise if you want to reanalyse a video, just insert the full path
    record_video: bool = True
    draw_rois: bool = False # set to true if you want to redraw the rois
    pause_between_frames: bool = False

    #if testing/ need shorter or different times to check that the sounds and trials logic work. Check get_trial_lengths() below if you want to change trial lengths
    testing: bool = False
    longer_middle_silence: bool = False # set to true if you want the silent trial in the middle to be longer

    ### Experiment mode ###
    # Choose ONE of the following modes:
    # -  "simple_smooth" (only pure sine tones as stimuli, no controls), "simple_intervals" (only intervals as stimuli, no controls), 
    # -  "temporal_modulation" (stimuli with different amounts of temporal envelope modulation + one silent arm + 1 vocalisation control),
    # -  "complex_intervals" (similar to temporal_modulation, but with different intervals), 
    # - "sequences" (stimuli are sequences + controls), 
    # - "vocalisations" (all the stimuli are vocalisations)

    experiment_mode: str = "complex_intervals"

    # Only used if experiment_mode == "complex_intervals"
    # Options: "w1day2", "w1day3", "w1day4", "another_day"
    complex_interval_day: str = "w1_day3"


    # Trial Settings 
    rois_number:int  = 8 # here we put the number of ROIs. If you are changing the rois number from one experiment to the other, remember to set draw_rois = True
    entrance_rois: List[str]= field(default_factory= lambda:["entrance1", "entrance2"]) # cannot just put this as a list,this way it creates a new list for 
    

    # PATHS
    base_output_path: str = r"C:/Users/labadmin/Desktop/auditory_maze_experiments/maze_recordings"


    def get_trial_lengths(self) -> List[float]:

        if self.testing:
            return [0.1, 1, 0.2, 2, 0.2, 2, 0.2, 2, 0.2]
        elif self.longer_middle_silence:
            return [15, 15, 2, 15, 15, 15, 2, 15, 2]
        elif self.use_microcontroller:
            return [15, 10, 2, 10, 2, 10, 2, 10, 2]
        else:
            return [15, 15, 2, 15, 2, 15, 2, 15, 2]








