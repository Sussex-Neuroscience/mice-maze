# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 14:34:38 2022

@author: seyij
"""

#get deeplabcut working

import deeplabcut as dlc
import cv2
from matplotlib import pyplot as plt
#%%entering information

task = "nort"
experimenter = "oj"
video_folder = "C:\\Users\\seyij\\oj_rotation2\\nort_vids"
videos =[video_folder+"1_trim.mp4", video_folder+"2_trim.mp4", video_folder+"3_trim.mp4", video_folder+"4_trim.mp4", video_folder+"5_trim.mp4"]

#%%
path_to_config = dlc.create_new_project(task, experimenter, videos, working_directory="C:\\Users\\seyij\\oj_rotation2", copy_videos = True, multianimal=False)

#%% load image
 
image = cv2.imread("C:\\Users\\seyij\\oj_rotation2\\nort_vids\\1_trim_moment.jpg")
# the image is 640w x 480h
#%% # the image is 640w x 480h
#left is h, right is w
crop = image[50:450, 150:480]

plt.imshow(crop)

#%% working within dlc_dml3 to confirm the environment works

import deeplabcut as dlc
import os.path

task = "pilot"
experimenter = "oj"
video_folder = r"C:\Users\seyij\Pictures\Camera Roll"
videos =[os.path.join(video_folder,"maze_pilot1.mp4"), os.path.join(video_folder,"maze_pilot2.mp4")]
#make the config
path_to_config = dlc.create_new_project(task, experimenter, videos, working_directory="C:\\Users\\seyij\\oj_rotation2", copy_videos = True, multianimal=False)

#%%
import deeplabcut as dlc

#get path to deeplabcut config file
path_to_config = r"C:\Users\seyij\oj_rotation2\pilot-oj-2022-03-03\config.yaml"
#check function details for downsampling function
help(dlc.DownSampleVideo)

#get path to target videos
vid1 = r"C:\Users\seyij\oj_rotation2\pilot-oj-2022-03-03\videos\maze_pilot1.mp4"
vid2 = r"C:\Users\seyij\oj_rotation2\pilot-oj-2022-03-03\videos\maze_pilot2.mp4"

#create suffix string for down sampled video nams
suff="_540p"
#downsample ideossto specified resolution
dlc.DownSampleVideo(vid1, width=960, height=540, outsuffix=suff)
dlc.DownSampleVideo(vid2, width=960, height=540, outsuffix=suff)

#%%
#paths to new videos
vid3 = r'C:\\Users\\seyij\\oj_rotation2\\pilot-oj-2022-03-03\\videos\\maze_pilot1_540p.mp4'
vid4 = r'C:\\Users\\seyij\\oj_rotation2\\pilot-oj-2022-03-03\\videos\\maze_pilot2_540p.mp4'
#add the new videos. Don't need to copy because they're already in the correct folder
dlc.add_new_videos(path_to_config, [vid3, vid4], copy_videos=False)

dlc.launch_dlc()
#%% had memory issues, will try again
import tensorflow as tf
import deeplabcut as dlc

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

#%% using mixed precision

import tensorflow as tf
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

#%%
import deeplabcut as dlc

config_path = r"C:\Users\seyij\oj_rotation2\mmaze-oj-2022-04-20\config.yaml"

dlc.train_network(config_path, shuffle=1, trainingsetindex=0,max_snapshots_to_keep=5, displayiters=100, saveiters=10000, maxiters=500000)
#%%

