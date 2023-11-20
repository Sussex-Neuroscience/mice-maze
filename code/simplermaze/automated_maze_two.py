import pyfirmata2
import time
import numpy as np
import pandas as pd
from tkinter import filedialog as fd
from tkinter import *
import reward_pts_selection as roi_selection
import mouse_reward_detection
import entrance_pts_selection
import csv




def main():
    csv_file=choose_csv()
    initialisation(csv_file)
    
    # start_trial()
        #start_recording()
            #entrance_pts()
        #set_reward(csv_file)
        # food_release()
    # end_trial()




def initialisation(csv_file):
    initialising_board()
    check_rois(csv_file)
    set_gratings(csv_file)


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

def check_rois(csv_file):
    df = pd.read_csv(csv_file)
    # Display the first 5 rows of the DataFrame
    #print(df)
    
    #check if these columns are empty
    if df[['ROI1', 'ROI2', 'ROI3','ROI4']].isnull().all().all():
        #if they are empty, run code to select ROIs
        rois_pts=roi_selection.select_rois()
        
        #convert matrices to tuples
        rois_pts2= [
            [(x, y) for x, y in matrix]  
            for matrix in rois_pts
        ]
        
        print(rois_pts2)
        # Update DataFrame with ROI coordinates
        for i, col in enumerate(df.columns[-4:]):
            df[col] = [rois_pts2[i]]*df.shape[0]

        
        #update the csv file
        df.to_csv(csv_file, index=False)

        
        
    else:
        print("file already has rois")


def set_gratings():
    global servo_2
    global servo_3
    global servo_4
    global servo_5
    global servo_6
    global servo_7

    #### CHANGE FROM USER SELECTING TRIAL TO ITERATION THROUGH THE TRIALS ---> LINE 125 

    # This function creates and uses custom gratings for different trial types
    csv_file= choose_csv()
    # opens csv file,
    TRIALS={}
    try:
        
        f= open(csv_file, "r")
        reader = csv.reader(f)
        first_row= next(reader)
        MOTOR_NAMES= first_row[1:7]

        next(reader)
        # Loop through the rows
        for row in reader:
            # Get the first element as the key
            key = row[0]
            # Get the rest of the elements as the values
            values = row[1:]
            # Append the values to a list
            values_list = []
            for value in values:
                values_list.append(value)
            # Assign the list to the key in the dictionary
            TRIALS[key] = values_list
    except IOError as e:
        print(f"Error reading file {csv_file}: {e}")
        exit()

    # Ask the user for the type of trial or make a custom one
    trial_type = input("Insert type of trial or make a custom one: ").upper()

    if trial_type in TRIALS:
        # If the trial type exists, use the corresponding grating
        grating_angles = TRIALS[trial_type]

    else:
        f.close()
        # If the trial type does not exist, ask the user for the custom grating angles
        custom_trial_name = trial_type
        custom_grating_angles = []
        for motor in MOTOR_NAMES:
            angle = input(f"Insert angle for grating {motor}: ")
            # Validate the input as an integer
            if angle.isdigit():
                custom_grating_angles.append(int(angle))
            else:
                print(f"Invalid angle: {angle}")
                exit()

        # Update the trials dictionary with the new trial type and grating
        TRIALS[custom_trial_name] = custom_grating_angles

        #update the csv file with the custom trial
        try:
            with open(csv_file, "a", newline="") as file:
            # Create a csv writer object
                writer = csv.writer(file)
                # Write the custom trial name as the first element of the row
                writer.writerow([custom_trial_name] + custom_grating_angles)
        except IOError as e:
            print(f"Error writing file {csv_file}: {e}")
            exit()

        # Use the new grating as grating_angles
        grating_angles = TRIALS[custom_trial_name]


    # Create a dictionary of trial angles from the motor names and grating angles
    trial_angles = {MOTOR_NAMES[i]: grating_angles[i] for i in range(len(MOTOR_NAMES))}


   
    
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