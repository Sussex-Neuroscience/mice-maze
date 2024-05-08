#ampy --port /dev/ttyUSB0 --baud 115200 run /home/maze_/Desktop/codes_maze/maze_project/sensor.py &> sensor.csv

import numpy as np
import argparse
import cv2
import os
import sys
import time
import argparse
from random import *
import imutils
from easygui import *
from datetime import datetime
from os.path import exists
import matplotlib
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
import random
from threading import Thread

#Pannels #################################
exec(open("./servo_control_auto.py").read())
list_of_random = ["allzero", "allzero"]
list_trials=np.array(["RR","RL","LL","LR","allzero"])
index_trial=np.where(list_trials == list_of_random[-1])[0][0]
pannel_sort('allzero')

#Food #################################
exec(open("./servo_control_food.py").read())

#Hide prints #################################
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        
#Info #################################
text = "Enter the following details /n Don't put the mouse yet"
title = "Window Title GfG"
input_list = ["AnimalID", "Training sequence/habituation", "duration in minutes"]
output = multenterbox(text, title, input_list)
AnimalID=output[0]
training=output[1]
minutes=output[2]
today = datetime.today()
current_time = str(today.hour)+'_'+str(today.minute)+'_'+str(today.second)+'__'+str(today.day)+str(today.month)+str(today.year)
path = "/home/maze_/Desktop/data_maze_videos/"+AnimalID+"/"
isExist = os.path.exists(path)
if not isExist:
   os.makedirs(path)
   print("The new directory for " +AnimalID+ " created!")

#Sensor reading#################################
def convertMillis(s):
   seconds=(s)%60
   s = s - seconds
   minutes=(s/(60))%60
   s = s - minutes
   hours=(s/(60*60))%24
   return str(int(hours))+':'+str(int(minutes))+':'+str(int(seconds))
#Camera reading#################################
capture = cv2.VideoCapture(0)

#Saving video###################################

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videoWidth = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
videoHeight = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer_all = cv2.VideoWriter(path + AnimalID+'_'+current_time+'_'+training+'_all.mp4',fourcc, 30.0,(videoWidth,videoHeight))
writer_entry = cv2.VideoWriter(path + AnimalID+'_'+current_time+'_'+training+'_entry_only.mp4',fourcc, 30.0,(videoWidth,videoHeight))
writer_crop = cv2.VideoWriter(path + AnimalID+'_'+current_time+'_'+training+'_all_crop.mp4',fourcc, 30.0,(260,260))

#Initialize list####################

for i in range(1,100,1):
    (grabbed, frame) = capture.read() #video processing
frame = cv2.cvtColor(frame[120:380, 120:380], cv2.COLOR_BGR2GRAY)

list_frames = []
list_frames.append(frame.copy()[9:40, 139:167])
list_frames.append(frame.copy()[9:40, 209:240])
list_frames.append(frame.copy()[220:250, 130:165])
list_frames.append(frame.copy()[221:250, 205:240])


for i in range(0,4,1):
    exec('base_'+str(i)+' = cv2.mean(list_frames['+str(i)+'])[0]')

#Ploting pixel intensity#############################################

#Some other parameters####################
alpha = 0.4  #how transparent the pixel_ intensity rectangle is
counter = 0 
save_1= 0
save_2= 0

global timeout, time_start_label

message = "Attach the cage to the maze, start the sensor, recharge the food and press CONTINUE"
title = "Waiting ..."
ok_btn_txt = "CONTINUE"
output = msgbox(message, title, ok_btn_txt)
print("User pressed  : " + output)
global counter_visits
counter_visits = 0
pixel = 0
cmap = matplotlib.cm.get_cmap('Purples')
firstFrame = None
Reward=['No','No', 'No', 'No', 'No']
Rewarded_already='No'
Trial = True
food_allowances = np.array([1, 1, 1, 1, 1])
time.sleep(1.5)

inputFile = "/home/maze_/sensor.csv"
f1 = open(inputFile, "r")
last_line = f1.readlines()[-1].splitlines()[0].split(",")
f1.close()
last=last_line

time.sleep(1.5)
############################
pg.setConfigOptions(antialias=True)

pg.mkQApp("")

win = pg.GraphicsLayoutWidget(show=True)
win.resize(800  ,800)
win.setWindowTitle('Hit and Miss')
list_colors=[(255,0,0), (255,255,0), (0,255,0), (0,0,255), (255,255,255)]
list_colors_brush=[(255,0,0,200), (255,255,0,200), (0,255,0,200), (0,0,255,200),  (255,255,255,200)]

bins = 500
s=np.arange(-1,0,1/bins)
list_labels=["RR","RL","LL","LR"]

for i in range(0,2,1):
    exec('s'+str(i)+' = s.copy()')
    exec('p'+str(i)+' = win.addPlot(title="'+list_labels[i]+'")')
    exec('roi'+str(i)+'=p'+str(i)+'.plot(s,0*s'+str(i)+',pen=list_colors[i])')
    exec('p'+str(i)+'.enableAutoRange("y", True)')
    exec('p'+str(i)+'.setLabel("left", "pixel int.")')
    exec('p'+str(i)+'.setLabel("bottom", "time (s)")')

win.nextRow()
for i in range(2,4,1):
    exec('s'+str(i)+' = s.copy()')
    exec('p'+str(i)+' = win.addPlot(title="'+list_labels[i]+'")')
    exec('roi'+str(i)+'=p'+str(i)+'.plot(s,0*s'+str(i)+',pen=list_colors[i])')
    exec('p'+str(i)+'.enableAutoRange("y", True)')
    exec('p'+str(i)+'.setLabel("left", "pixel int.")')
    exec('p'+str(i)+'.setLabel("bottom", "time (s)")')


for i in range(0,4,1):
    exec('s_'+str(i)+' = s.copy()*0')

# import pyqtgraph.examples
# pyqtgraph.examples.run()

win.nextRow()
gonogo=win.addLayout(colspan=2)
performance=gonogo.addPlot(title="Performance history")
performance.enableAutoRange("x", True)
legend = pg.LegendItem((100,80), offset=(70,20))
legend.setParentItem(performance)
vector=np.zeros(500)
prfrmc=np.zeros(500)
for i in range(0,5,1):
    exec('vector_'+list_trials[i]+'=vector.copy()')
    exec('prfrmc_'+list_trials[i]+'=prfrmc.copy()')
    exec('perf_'+list_trials[i]+'=performance.plot(vector_'+list_trials[i]+',prfrmc_'+list_trials[i]+',fillLevel=0,pen=list_colors[i],brush='+str(list_colors_brush[i])+',name=list_trials[i])')
    exec('legend.addItem(perf_'+list_trials[i]+', list_trials[i])')
performance.setLabel("left", "ROI time spent")
performance.setLabel("bottom", "time (s)")


win.nextRow()
pixel_layout=win.addLayout(colspan=2)
pixel__=pixel_layout.addPlot(title="Pixel integration")
pixel__.enableAutoRange("x", True)
legend = pg.LegendItem((100,80), offset=(70,20))
legend.setParentItem(pixel__)
for i in range(0,4,1):
    exec('pixel_'+str(i)+'=pixel__.plot(np.zeros(bins),np.zeros(bins),pen=list_colors[i],name=list_labels[i])')
    exec('legend.addItem(pixel_'+str(i)+', list_labels[i])')
pixel__.setLabel("left", "ROI time spent")
pixel__.setLabel("bottom", "time (s)")

time_start = time.time()
timeout = time.time() + 60*1e10 #int(minutes)   # Timeout
time_start_label = "no"
time_base = np.arange(-1,0, 1/bins)


with HiddenPrints():
    while True: #camera
        if exists(inputFile): #sensor to read last line
            f1 = open(inputFile, "r")
            last_line = f1.readlines()[-1].splitlines()[0].split(",")
            f1.close()
            while last_line[-1]!="ready" :
                f1 = open(inputFile, "r")
                last_line = f1.readlines()[-1].splitlines()[0].split(",")
                f1.close()

        if last_line[0] == "cage": #measure the time spent inside the maze
            if last[0] == "maze":
                save_1 = int(last[2])/1000/60
                save_2 = int(last_line[2])/1000/60
                exec('vector_'+list_of_random[-1]+'=np.append(vector_'+list_of_random[-1]+',[save_1, save_1, save_2, save_2])')
                exec('vector_'+list_of_random[-1]+'=np.delete(vector_'+list_of_random[-1]+',[0,1,2,3])')
                if Trial == True:
                    exec('prfrmc_'+list_of_random[-1]+'=np.delete(prfrmc_'+list_of_random[-1]+',[0,1,2,3])')
                    exec('prfrmc_'+list_of_random[-1]+'=np.append(prfrmc_'+list_of_random[-1]+',[0,1,1,0])')
                if Trial == False:
                    exec('prfrmc_'+list_of_random[-1]+'=np.delete(prfrmc_'+list_of_random[-1]+',[0,1,2,3])')
                    exec('prfrmc_'+list_of_random[-1]+'=np.append(prfrmc_'+list_of_random[-1]+',[0,-1,-1,0])')
                counter_visits = counter_visits+1
                trial_choice=random.choice(list_trials[food_allowances < 8])
                list_of_random = np.append(list_of_random, trial_choice) #reset textures randomly
                T=Thread(target=pannel_sort,args=(list_of_random[-1],))
                T.start() #reset textures randomly
                index_trial=np.where(list_trials == list_of_random[-1])[0][0]
                Reward=['No','No', 'No', 'No', 'No'] # reset reward once animals comes off the maze 
                Rewarded_already='No' # reset reward once animals comes off the maze 
                Trial = False #reset trial outcome if False failure, if True success

        last = last_line.copy()
        
        (grabbed, frame) = capture.read() #video processing
        if grabbed == True:
            writer_all.write(frame)
            if last_line[0] == "maze": #if the mouse is in the maze
                writer_entry.write(frame)
                if time_start_label == "no" : # #only starts timeout if the animal is in the maze
                    timeout = time.time() + 60*int(minutes) 
                    time_start_label = "yes" #do not reestart this
            ###### to plot the transparent rectangles            
            overlay = frame.copy()[120:380, 120:380] 
            frame = frame.copy()[120:380, 120:380]
            writer_crop.write(frame)

            #saving reward region places for pixel intensity comparisons
            list_frames[0] = frame.copy()[9:40, 139:167]
            list_frames[1] = frame.copy()[9:40, 209:240]
            list_frames[2] = frame.copy()[220:250, 130:165]
            list_frames[3] = frame.copy()[221:250, 205:240]
            
            #time
            milli= time.time() - time_start
            time_base=np.delete(time_base,0)
            time_base=np.append(time_base,milli)

            gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)   
            if firstFrame is None:
                firstFrame = gray
                continue         
            exec('perf_'+list_of_random[-2]+'.setData(vector_'+list_of_random[-2]+',prfrmc_'+list_of_random[-2]+',fillLevel=0, listname=list_of_random[-2])')

            
            for i in range(0,4,1):
                exec('pixel = abs(base_'+str(i)+' - cv2.mean(list_frames['+str(i)+'])[0])')
                exec('s'+str(i)+'=np.delete(s'+str(i)+',0)')
                exec('s'+str(i)+'=np.append(s'+str(i)+', pixel)')
                exec('roi'+str(i)+'.setData(time_base,s'+str(i)+')')

                if pixel <= 15:
                    exec('s_'+str(i)+'=np.delete(s_'+str(i)+',0)')
                    exec('s_'+str(i)+'=np.append(s_'+str(i)+', s_'+str(i)+'[-1])')
                    exec('pixel_'+str(i)+'.setData(time_base,s_'+str(i)+',name=list_labels[i])')
                if pixel > 15:
                    exec('s_'+str(i)+'=np.delete(s_'+str(i)+',0)')
                    exec('s_'+str(i)+'=np.append(s_'+str(i)+', s_'+str(i)+'[-1]+(time_base[-1]-time_base[-2]))')
                    exec('pixel_'+str(i)+'.setData(time_base,s_'+str(i)+',pen=list_colors[i],listname=list_labels[i])')
                    if Reward[i]=='No':
                        Reward[i] ='Yes'
                        
            
            if Rewarded_already == 'No':
                if Reward[index_trial] == "Yes":   
                    if Reward.count('Yes')==1:
                        Trial = True
                        T_food=Thread(target=food_moving,args=(list_of_random[-1],food_allowances[index_trial],))
                        T_food.start()
                        #food_moving(list_of_random[-1],food_allowances[index_trial])
                        food_allowances[index_trial] = food_allowances[index_trial]+1
                        Rewarded_already = 'Yes'
            
                if Reward.count('Yes')>1:
                    Trial = False
            
            if list_of_random[-1] == 'allzero':
                if Reward.count('Yes')>0:
                    Trial = False

            counter=counter+1
        

            #saving reward region places for pixel intensity comparisons
            frameDelta = cv2.absdiff(firstFrame, gray)
            thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            cv2.imshow('thresh', thresh)
            # cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # cnts = imutils.grab_contours(cnts)
            # for c in cnts:
            #     if cv2.contourArea(c) < 350:
            #         continue
            #     (x, y, w, h) = cv2.boundingRect(c)
            #     cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

            #drawing rectangles
            max_color=max(s_0[-1],s_1[-1],s_2[-1],s_3[-1],0.001)
            cv2.rectangle(overlay, (139, 9), (167, 40), tuple(int(255*a) for a in cmap(s_0[-1]/max_color)[0:3]), -2)
            cv2.rectangle(overlay, (209, 9), (240, 40), tuple(int(255*a) for a in cmap(s_1[-1]/max_color)[0:3]), -2)
            cv2.rectangle(overlay, (130, 220), (165, 250), tuple(int(255*a) for a in cmap(s_2[-1]/max_color)[0:3]), -2)
            cv2.rectangle(overlay, (205, 221), (240, 250), tuple(int(255*a) for a in cmap(s_3[-1]/max_color)[0:3]), -2)
            

            frame_2 = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            frame_2 = cv2.resize(frame_2, (560, 560))

            cv2.putText(frame_2,'RR',(230, 66),cv2.FONT_HERSHEY_SIMPLEX, 1.2,list_colors[3],2,cv2.LINE_4)       
            cv2.putText(frame_2,'RL',(387, 66),cv2.FONT_HERSHEY_SIMPLEX, 1.2,(0,255,255),2,cv2.LINE_4)       
            cv2.putText(frame_2,'LL',(230, 530),cv2.FONT_HERSHEY_SIMPLEX, 1.2,list_colors[2],2,cv2.LINE_4)       
            cv2.putText(frame_2,'LR',(387, 530),cv2.FONT_HERSHEY_SIMPLEX, 1.2,list_colors[0],2,cv2.LINE_4)       
            cv2.putText(frame_2,convertMillis(milli),(20*2, 238*2),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255, 255 , 255),2,cv2.LINE_4)
            cv2.putText(frame_2,'task=' + list_of_random[-1],(20*2, 220*2),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255, 255 , 255),2,cv2.LINE_4)       
            cv2.putText(frame_2,str(Reward[0:-1]),(20*2, 202*2),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255, 255 , 255),2,cv2.LINE_4)
            cv2.putText(frame_2,str(food_allowances[0:-1]),(20*2, 180*2),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255, 255 , 255),2,cv2.LINE_4)       
            cv2.imshow('frame', frame_2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if (time.time() > timeout) and (last_line[0] == "cage"): #only finishes if the animal is in the cage and the time is out
            break
    capture.release()
    writer_all.release()
    writer_entry.release()
    writer_crop.release()
    cv2.destroyAllWindows()

print('counted visits is: ' + str(counter_visits))
os.system("cp " + inputFile + " " + path + AnimalID+"_"+current_time+"_"+training+"_entries.csv")
print((s_0[-1],s_1[-1],s_2[-1],s_3[-1]))

close_board()
close_board_food()
