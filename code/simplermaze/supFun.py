import pandas as pd
import cv2 as cv
import numpy as np
import time
import os
from tkinter import filedialog as fd
from tkinter import *
import csv


def collect_metadata(animal_ID, session_ID):
    ear_mark = input("Ear mark identifiers? (y/n): \n").lower()
    weight = input("Insert animal weight (g): \n").lower()
    birth_date = input("Insert animal birth date (dd/mm/yyyy): \n")
    gender = input("Insert animal assumed gender (m/f): \n").lower()

    data = {
    "animal ID": animal_ID,
    "session ID": session_ID,
    "ear mark identifier": ear_mark,
    "animal weight": weight,
    "animal birth date": birth_date,
    "animal gender": gender,
    }

    return data

def save_metadata_to_csv(data, new_dir_path, file_name):
    df = pd.DataFrame([data])  # Convert dict to dataframe
    csv_path = os.path.join(new_dir_path, file_name)
    df.to_csv(csv_path, index=False)
    print(f"Metadata saved to: {csv_path}")


def setup_directories(base_path, date_time, animal_ID, session_ID):
    new_directory = f"{date_time}{animal_ID}{session_ID}"
    new_dir_path = os.path.join(base_path, new_directory)
    ensure_directory_exists(new_dir_path)
    return new_dir_path

def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created: {path}")
    else:
        print(f"Directory already exists: {path}")

def get_user_inputs():
    animal_ID = input("Insert animal ID: \n").upper()
    session_ID = input("Insert Session ID: \n").lower()
    experiment_phase = int(input("Insert experiment phase (1-4): \n"))
    return animal_ID, session_ID, experiment_phase

def get_current_time_formatted():
    return time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime())

def get_metadata():
    anID = input("enter animal identification:")
    date = input("enter date:")
    sessionID = input("enter session identification")
    metatada = [anID, date, sessionID]
    #figure out what else users want to store
    return metadata

def write_text(text="string",window="noWindow"):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,500)
    fontScale              = 1
    fontColor              = (255,255,255)
    thickness              = 1
    lineType               = 2
    
    cv2.putText(window,text, 
    bottomLeftCornerOfText, 
    font, 
    fontScale,
    fontColor,
    thickness,
    lineType)
    
    return


def start_camera(videoInput=0):
    cap = cv.VideoCapture(videoInput)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    return cap

def record_video(cap, recordFile, frame_width, frame_height, fps):
    cc = cv.VideoWriter_fourcc(*'XVID')
    videoFileObject = cv.VideoWriter(recordFile, cc, fps, (frame_width, frame_height))
    return videoFileObject
    #if not videoFile.isOpened():
    #    print("Error: Failed to create VideoWriter object.")
    #    cap.release()
    #    exit()

    #while True:
    #    ret, frame = cap.read()
    #    if ret:
    #        videoFile.write(frame)
    #        cv.imshow("Frame", frame)
    #        if cv.waitKey(1) & 0xFF in [ord('q'), 27]:
    #            break
    #    else:
    #        print("Error: Failed to read frame from camera.")
    #        break

    #videoFile.release()


def grab_n_convert_frame(cameraHandle):
    #capture a frame
    ret, frame = cameraHandle.read()
    #print(ret)
    # Our operations on the frame come here
    #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray=frame
    return gray,ret 

def csv_to_dict(fileName="rois.csv"):
    output = dict()
    temp =  pd.read_csv(fileName,index_col=0)
    temp = temp.transpose()
    output = temp.to_dict()
    return output


def define_rois(videoInput = 0,
                roiNames = ["entrance1","entrance2",
                             "rew1","rew2","rew3","rew4"],
                outputName = "rois.csv"):
    

    cap = cv.VideoCapture(videoInput)
    #cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        #break
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # Display the resulting frame
    #cv.imshow('frame', gray)

    #release the capture
    cap.release()

    rois = {}
    for entry in roiNames:
        print("please select location of "+str(entry))
        rois[entry] = cv.selectROI('frame', gray)
        #print(rois[entry])
    
    df = pd.DataFrame(rois)
    #print(df)
    #print(roiNames)
    df.index = ["xstart","ystart","xlen","ylen"]
    df.to_csv(outputName,index=["xstart","ystart","xlen","ylen"])
    #when all done destroy windows
    cv.destroyAllWindows()

    return df

#def draw_on_video(frame=gray,roi=(0,0,0,0)):
    #cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (0, 0, 0), -1)
    

def grab_cut(frame,xstart,ystart,xlen,ylen):
    
    cut = frame[ystart:ystart+ylen,
                xstart:xstart+xlen]
    return cut

def create_trials(numTrials = 100, sessionStage = 2,nonRepeat=False):
    
    if sessionStage<1 or sessionStage>4:
        print("invalid session stage, defaulting to stage 1 - habituation")
        sessionStage=1

    gratingMap = pd.read_csv("grating_maps.csv")
    rewardSequences = pd.read_csv("reward_sequences.csv")

    stage = rewardSequences[rewardSequences.sessionID=="Stage "+str(sessionStage)]
    #lenRewLoc = len(stage.rewloc)

    trialsDistribution = dict()
    for index,location in enumerate(stage.rewloc):
        subTrials = int(np.floor(numTrials*list(stage.portprob)[index]))
        probRewardLocation = np.random.choice([True,False],numTrials,p=[list(stage.rewprob)[index],
                                                                1-list(stage.rewprob)[index]])
        
        #print(probRewardLocation)
        trialTuples = list()
        
            
        for i in range(subTrials):
            trialTuples.append((location,probRewardLocation[i],list(stage.wrongallowed)[index]))
        
        trialsDistribution[location] = trialTuples

    allTogether=list()

    for item in trialsDistribution.keys():
        allTogether+=(trialsDistribution[item])#this keeps the list flat
                                               #(as opposed to a list of lists)

    #now shuffle the list
    np.random.shuffle(allTogether)
    
    if nonRepeat==True:
        #rearrange the list so that two/three consecutive trials rewarding the same location never happens
        for i in range(4):
            for index in range(len(allTogether)-1):
                if allTogether[index][0]==allTogether[index+1][0]:
                    print("repeat coming up... fixing")
                    temp = allTogether[index+1]
                    allTogether.append(temp)
                    allTogether.pop(index+1)


    # create DataFrame using data
    trials = pd.DataFrame(allTogether, columns =['rewlocation', 'givereward', 'wrongallowed'])

    return trials

def choose_csv():
    root=Tk()
    # Show the file dialog and get the selected file name
    filename = fd.askopenfilename()
    # Print the file name to the console
    #print (filename)
    root.destroy()
    return filename

def empty_frame(rows=300,roi_names=["entrance1","entrance2","rewA","rewB","rewC","rewD"]):
    #create a dataframe that will contain only nans for all things to be measured.
    #during the session we will fill up this df with data
    

    columns = ["hit","miss","incorrect","rew_location","area_rewarded","time_to_reward",
           "trial_start_time","end_trial_time","mouse_enter_time","first_reward_area_visited"]+roi_names
    data = pd.DataFrame(None, index=range(rows), columns=columns)
    return data
    
def time_in_millis():
    millis=round(time.time() * 1000)
    return millis

def write_data(file_name="tests.csv",mode="a",data=["test","test","test"]):
    data_fh = open(file_name,mode)
    data_writer = csv.writer(data_fh, delimiter=',')
    data_writer.writerow(data)
    data_fh.close()