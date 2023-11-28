import pyfirmata2
import time
import numpy as np
import pandas as pd
from tkinter import filedialog as fd
from tkinter import *
import reward_pts_selection as roi_selection
import mouse_reward_detection
import entrance_pts_selection
from ast import literal_eval




def main():
    csv_file=choose_csv()
    add_custom_trial(csv_file)
    check_rois(csv_file)
    set_reward_rois(csv_file)
    set_entrance_points(csv_file)
    
    for i in range(set_index(csv_file)):
        set_gratings(csv_file, i)
        #start_recording()
            #entrance_pts()
        #set_reward(csv_file)
    #initialisation(csv_file)
    
    # start_trial()
        #start_recording()
            #entrance_pts()
        #set_reward(csv_file)
        # food_release()
    # end_trial()



'''
def initialisation(csv_file):
    initialising_board()
    check_rois(csv_file)
    set_gratings(csv_file)
'''

def initialising_board():
    global board
    PORT = pyfirmata2.Arduino.AUTODETECT
    board = pyfirmata2.Arduino(PORT)

    #If this fails you can also specify the serial port manually:
    #board = pyfirmata2.Arduino("/dev/ttyACM0")
    # On Linux: /dev/tty.usbserial-A6008rIF, /dev/ttyACM0,
    # On Windows: \\.\COM1, \\.\COM2
    # PORT = '/dev/ttyACM0'

    print("Setting up the connection to the board ...")

def choose_csv():
    root=Tk()
    # Show the file dialog and get the selected file name
    filename = fd.askopenfilename()
    # Print the file name to the console
    #print (filename)
    root.destroy()
    return filename

def add_custom_trial(csv_file):
        
        df= pd.read_csv(csv_file)
    # Ask the user for the type of trial or make a custom one
        custom = input("Do you want to add a custom trial? (y/n): ").upper()

        if custom == "Y":
            # Create a new row to append
            new_row = {'Trials': input("Enter Custom Trial name: "),
            'l': int(input("Enter angle for grating l: ")),
            'r': int(input("Enter angle for grating r: ")),
            'll': int(input("Enter angle for grating ll: ")),
            'lr': int(input("Enter angle for grating lr: ")),
            'rl': int(input("Enter angle for grating rl: ")),
            'rr': int(input("Enter angle for grating rr: "))}

            # Append the new row to the DataFrame
            df = df.append(new_row, ignore_index=True)
            # Update the CSV file
            df.to_csv(csv_file, index=False)

def set_index(csv_file):
    df = pd.read_csv(csv_file)
    num_rows = df.shape[0]
    return num_rows

def check_rois(csv_file):
    df = pd.read_csv(csv_file)

    # Define the ROI columns to check
    roi_columns = [f'ROI{i}' for i in range(1, 5)]

    # Check if any row lacks ROIs
    rows_missing_rois = df[roi_columns].isnull().any(axis=1)

    # Check if all specified ROI columns are empty for the entire DataFrame
    if df[roi_columns].isnull().all().all():
        # If all ROIs are missing in the DataFrame, get new ROI coordinates
        rois_pts = roi_selection()

        # Convert matrices to tuples
        rois_pts2 = [
            [(x, y) for x, y in matrix]
            for matrix in rois_pts
        ]

        # Update DataFrame with ROI coordinates for all rows
        for i, col in enumerate(roi_columns):
            df[col] = [rois_pts2[i]] * df.shape[0]

        # Update the CSV file
        df.to_csv(csv_file, index=False)
    elif any(rows_missing_rois):
        # If some rows have data but lack ROIs, fill with ROIs from the row above
        df.loc[rows_missing_rois, roi_columns] = df.shift(1).loc[rows_missing_rois, roi_columns].values

        # Update the CSV file
        df.to_csv(csv_file, index=False)

def set_gratings(csv_file, i):
    global servo_2, servo_3, servo_4, servo_5, servo_6, servo_7

    df = pd.read_csv(csv_file)
    #print(df)
    
    MOTOR_NAMES= df.columns[1:7].tolist()
    
    #select the trial based on the index that iterates through the trials in the csv file (look at main())
    trial = df.iloc[i, 0:7]
    trial_name = df.iloc[i, 0]
    angles_gratings = df.iloc[i, 1:7]
    angles_list = []
    for angle in angles_gratings:
        angles_list.append(angle)
        
    # Create a dictionary of the angles of teh specific trial 
    trial_angles = {MOTOR_NAMES[a]: angles_list[a] for a in range(len(MOTOR_NAMES))}
    
    # Setup the digital pins as servo. Servo is the motor, these numbers are the pins on the board
    servo_2 = board.get_pin('d:2:s')
    servo_3 = board.get_pin('d:3:s')
    servo_4 = board.get_pin('d:4:s')
    servo_5 = board.get_pin('d:5:s')
    servo_6 = board.get_pin('d:6:s')
    servo_7 = board.get_pin('d:7:s')


    # Set the duty cycle

    servo_2.write(trial_angles['l'])
    servo_3.write(trial_angles['r'])
    servo_4.write(trial_angles['ll'])
    servo_5.write(trial_angles['lr'])
    servo_6.write(trial_angles['rl'])
    servo_7.write(trial_angles['rr'])


def set_reward_rois(csv_file):
    df = pd.read_csv(csv_file)
    # Check if "reward" column is present
    if 'Reward ROI' not in df.columns:
        # If not present, add it after the other columns
        df['Reward ROI'] = [0] * df.shape[0]
        df['Reward ROI'] = df['Reward ROI'].astype(object) #if I don't do this, it doesn't allow me to copy the values from the ROIs

        df[['ROI1', 'ROI2', 'ROI3', 'ROI4']] = df[['ROI1', 'ROI2', 'ROI3', 'ROI4']].applymap(eval)

        df.at[0, 'Reward ROI'] = df.at[0, 'ROI1']
        df.at[1, 'Reward ROI'] = df.at[1, 'ROI2']
        df.at[2, 'Reward ROI'] = df.at[2, 'ROI3']
        df.at[3, 'Reward ROI'] = df.at[3, 'ROI4']
        
        #this makes sure that the values are not strings, but lists
        df['Reward ROI'] = df['Reward ROI'].apply(literal_eval)
        df.to_csv(csv_file, index=False)
    
    else:
        #this makes sure that the values are not strings, but lists
        df['Reward ROI'] = df['Reward ROI'].apply(literal_eval)

        if df['Reward ROI'].eq(0).any():
            # Set the 'Trials' column as the index
            df.set_index('Trials', inplace=True)
            # Create a dictionary to map the user's choice to the ROI column
            roi_dict = {"LL": "ROI1", "LR": "ROI2", "RL": "ROI3", "RR": "ROI4"}
            # Loop over the trials that have a 0 in the 'Reward ROI' column
            for trial in df[df['Reward ROI'].eq(0)].index:
                # Ask the user to select the location of the ROI
                set_custom_reward = input(f"Select location of ROI (LL, LR, RL, RR) for trial {trial}: ").upper()
                # Check if the user's choice is valid
                if set_custom_reward in roi_dict:
                    # Assign the value of the corresponding ROI column to the 'Reward ROI' column
                    df.loc[trial, 'Reward ROI'] = df.loc[trial, roi_dict[set_custom_reward]]
                else:
                    # Print an error message and skip the trial
                    print(f"No rewards for trial {trial}.") 
            
        
    
        df.to_csv(csv_file, index=False)
    
def set_entrance_points(csv_file):
    df= pd.read_csv(csv_file)    

###

def start_trial():
    # start_trial()
        #start_recording()
            #entrance_pts()
        #set_reward(csv_file)
    
    pass

def begin_recording():
    pass


###
def food_release():
    pass

###
def end_trial():
    pass



def save_recording():
    pass


def close_port():
    input("Press enter to exit")
    servo_2.write(0)
    servo_3.write(0)
    servo_4.write(0)
    servo_5.write(0)
    servo_6.write(0)
    servo_7.write(0)

    time.sleep(2)

    # Close the serial connection to the Arduino
    board.exit()



if __name__ == '__main__':
    main()