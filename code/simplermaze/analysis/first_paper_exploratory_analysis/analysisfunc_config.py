from dataclasses import dataclass, field
from typing import List, Optional
import os


# this will be the only analysis configuration file right now that we are handling the low likelihood DLC keypoints estimations

@dataclass

class Paths: 
    base_path:str = r"c:/Users/shahd/Box/Awake Project/Maze data/simplermaze/mouse 6357/"
 
    session_path:str = base_path+ r"2024-08-29_10_23_026357session3.7/"

    TRIAL_INFO_PATH:str = r"c:/Users/shahd/Box/Awake Project/Maze data/simplermaze/mouse 6357/2024-08-29_10_23_026357session3.7/mouse6357_session3.7_trial_info.csv" #session_path + r"mouse6357_session3.7_trial_info.csv" #r"trials_corrected_final_frames.csv"

    DLC_DATA_PATH:str = r"C:/Users/shahd/Box/Awake Project/Maze data/simplermaze/mouse 6357/deeplabcut/mouse6357/mouse6357-shahd-2025-09-08/videos/6357_2024-08-29_10_23_02s3.7DLC_Resnet50_mouse6357Sep8shuffle1_snapshot_200.csv" #base_path + r"deeplabcut/mouse6357/mouse6357-shahd-2025-09-08/videos/6357_2024-08-29_10_23_02s3.7DLC_Resnet50_mouse6357Sep8shuffle1_snapshot_200.csv"
    # for sesh 3.6 is "6357_2024-08-28_11_58_14s3.6DLC_Resnet50_mouse6357Sep8shuffle1_snapshot_200.csv"

    VIDEO_PATH:str = r"C:/Users/shahd/Box/Awake Project/Maze data/simplermaze/mouse 6357/2024-08-29_10_23_026357session3.7/6357_2024-08-29_10_23_02s3.7.mp4"#session_path + r"6357_2024-08-29_10_23_02s3.7.mp4" # for sesh 3.6"# r'6357_2024-08-28_11_58_14s3.6.mp4' # Replace with your full video path
    FPS:int = 30  # Adjust to your video's frame rate
    BODYPART:str = 'mid'  # Bodypart used for tracking
    DRAW_ROIS:bool = False  # Set to True to draw ROIs manually
    DRAW_BOUNDARIES:bool = False
    # GRID_SIZE = 20    # For Spatial Entropy
    LIKELIHOOD_THRESH:float = 0.5
    PX_PER_CM = 7.5
    

# def get_path_info(mouse, session, base_path)
