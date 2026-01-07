# figure out the logic of the session now that everything else is basically a function 

import cv2 as cv
import numpy as np
import pandas as pd
import time
import os
import sounddevice as sd

# our modules from the subdirectory `modules`
from config import ExperimentConfig
from modules.hardware import ArduinoController, Camera
from modules.vision import RoiMonitor
from modules.audio import Audio
from modules.experiments import ExperimentFactoryFarm
