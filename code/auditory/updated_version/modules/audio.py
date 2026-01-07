# here we will handle all things (functions) related to the sound data generation

import os
import sounddevice as sd
import soundfile as sf
import pandas as pd
import numpy as np
from scipy.signal import resample_poly
from scipy.interpolate import resample_poly
from scipy.interpolate import interp1d
from typing import List, Dict, Optional, Union, Tuple

class Audio:
    def __init__(self, samplerate: int = 192000, device_id: int = 3, calibration_gain_path: Optional[str] = r"C:/Users/labuser/Documents/GitHub/mice-maze/code/auditory/speaker calibration scripts/frequency_response_speaker.csv"):
        pass