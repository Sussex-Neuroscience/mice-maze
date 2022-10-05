# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 21:59:28 2022

@author: seyij
"""

# dlc_live dog test

import os
import shutil
from dlclive import benchmark_videos #had some future warnings but passed fine
import urllib.request
import tarfile

#%%


# make temporary directory in $HOME
print("changing to requested working directory")
# path to rotation directory
rt_dir = os.path.normpath(r"C:\Users\seyij\oj_rotation2")
# target folder within rotation 2 folder
t_dir = os.path.normpath(f"{rt_dir}\dlc_live")
#change working directory to specific folder selected 
os.chdir(t_dir)
# the above isn't really needed if your working directory is already full
#%% download dog test video from github:
url_link = "https://github.com/DeepLabCut/DeepLabCut-live/blob/master/check_install/dog_clip.avi?raw=True"
urllib.request.urlretrieve(url_link, "dog_clip.avi")
#not sure how below line helps but i'll leave it in for now
video_file = os.path.join(url_link, "dog_clip.avi")
#at this point the dog clip is downlowaded into the folder
video_file = "dog_clip.avi"
#%%
# download exported dog model from DeepLabCut Model Zoo
print("Downloading full_dog model from the DeepLabCut Model Zoo...")

#the url for the full dog model 
model_url = "http://deeplabcut.rowland.harvard.edu/models/DLC_Dog_resnet_50_iteration-0_shuffle-0.tar.gz"

#retrieve file from the url
urllib.request.urlretrieve(model_url, "full_doggo.tar.gz")

#open the tar file
file = tarfile.open("full_doggo.tar.gz")
#print names of the files within the tar 
print(file.getnames())
#looks like it's got all the neccessary files within it

# extract all files in tar to specified foldr
file.extractall(t_dir)
# it worked, extracting a deeplabcut folder with model files
#%% runthe demo

video_file = "dog_clip.avi"
# run benchmark videos
print("Running inference...")
model_dir = "DLC_Dog_resnet_50_iteration-0_shuffle-0"
print(video_file)
x = benchmark_videos(model_dir, video_file, display=True, resize=0.5, pcutoff=0.25)

# deleting temporary files
print("Done")




