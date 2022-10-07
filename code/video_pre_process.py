# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 11:50:42 2022

@author: seyij
"""

#script to preprocess videos and then make new DLC project.

import deeplabcut as dlc
import os
import cv2
from matplotlib import pyplot as plt
from moviepy.editor import VideoFileClip
#%% get list of video name

vid_direct = r"C:\Users\seyij\oj_rotation2\video_data"

vid_list = os.listdir(vid_direct)

#%% visualise frame from videos so i can crop appropriately

def ext_frame(vid_name, frame_index, out_name=False):
    """ Extract a frame or multiple frames from a video.
    
    Keyword arguments:
        vid_name -- name of input video
        
        frame_index -- frame or frame number to be extracted. Accepts integer for single frame. Accepts tuple or list with 2 numbers representing range of frames to be extracted.
        
        out_name -- name of image file output for single frame, or folder output for multiple frames (default False). 
    
    Returns:
        If extracting single frame with out_name=False, the function returns image matrix, with a out_name it writes an image file in the current working directory. If extracting multiple frames it writes a folder of images or writes many images to the cwd. 
    
    """
    if type(frame_index) == int:
        vid = cv2.VideoCapture(vid_name)
        print("Total amount of frames: " + str(vid.get(7)))
        vid.set(1, frame_index)
        ret, frame = vid.read()
        vid.release()
        if out_name == False:
            return frame
        else:
            cv2.imwrite(out_name, frame)
    #section for mutiple frames
    else:
        vid = cv2.VideoCapture(vid_name)
        a,b = frame_index
        if out_name == False:
            out_name = ""
        else:
            os.mkdir(out_name)
        #range cant iterate tuples, but it accepts multiple integers
        for number in range(a, b):
            vid.set(1, number)
            ret, frame = vid.read()
            cv2.imwrite((os.path.join(out_name, str(number), ".png"), frame))
        vid.release()

#get images from each video
img0 = ext_frame(os.path.join(vid_direct, vid_list[0]), 100)
img1 = ext_frame(os.path.join(vid_direct, vid_list[1]), 100)
img2 = ext_frame(os.path.join(vid_direct, vid_list[2]), 100)
img3 = ext_frame(os.path.join(vid_direct, vid_list[3]), 100)
img4 = ext_frame(os.path.join(vid_direct, vid_list[4]), 100)
img5 = ext_frame(os.path.join(vid_direct, vid_list[5]), 100)

#%% view the images and various crops

crop = img0[100:1000, 400:1600]

cv2.imshow("hello", crop)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%% now crop is decided, apply to videos

for file_name in vid_list:
    vid = os.path.join(vid_direct, file_name)
    
    dlc.CropVideo(vid, width=1600-400, height=1000-100, origin_x=400, origin_y=100, outsuffix="crop", outpath=vid_direct)

#%% remove audio from all videos

vid_direct = r"C:\Users\seyij\oj_rotation2\video_data"
vid_list = os.listdir(vid_direct)


for file_name in vid_list:
    #get full file name path
    vid = os.path.join(vid_direct, file_name)
    
    #define video
    videoclip = VideoFileClip(vid)
    #strip out sound
    new_clip = videoclip.without_audio()
    #output new file
    new_clip.write_videofile(filename=vid[:-4]+"_nos.mp4")

#%% delete all of the videos that still have sound
vid_direct = r"C:\Users\seyij\oj_rotation2\video_data"
vid_list = os.listdir(vid_direct)

for file_name in vid_list:
    #get full file name path
    vid = os.path.join(vid_direct, file_name)
    
    if file_name.endswith("nos.mp4") == False:
        os.remove(vid)

#%% rename the videos
vid_direct = r"C:\Users\seyij\oj_rotation2\video_data"
vid_list = os.listdir(vid_direct)

for file_name in vid_list:
    #get full file name path
    vid = os.path.join(vid_direct, file_name)
    
    os.rename(vid, vid[:-8]+".mp4")


#%% need to reduce resolution of videos to increase processing time

vid_direct = r"C:\Users\seyij\oj_rotation2\video_data"
vid_list = os.listdir(vid_direct)


#dlc.DownSampleVideo(vid, width=-1, height=450, outsuffix="down")

#the function can take the resolution
for file_name in vid_list:
    if file_name.endswith("down.mp4") == False:
        vid = os.path.join(vid_direct, file_name)
    #-1 makes it scale with the same aspect ratio while you change the other axis
        dlc.DownSampleVideo(vid, width=-1, height=600, outsuffix="down6")


#%% get all down sampled vids

vid_direct = r"C:\Users\seyij\oj_rotation2\video_data"
vid_list = os.listdir(vid_direct)

vid = os.path.join(vid_direct, vid_list[0])
dlc.DownSampleVideo(vid, width=-1, height=600, outsuffix="down6")

#%% useful way to get full path list without loops

task = "mmaze"
experimenter = "oj"
video_folder = r"C:\Users\seyij\oj_rotation2\video_data"
videos = os.listdir(video_folder)

#map makes iterators
full_video_paths = list(map(lambda vid_name: os.path.join(video_folder, vid_name), videos))

