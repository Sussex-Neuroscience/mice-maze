import pyfirmata2
import time
import numpy as np
import pandas as pd



def main():
    # initialisation()
    # start_trial()
    # food_release()
    # end_trial()
    set_gratings()

def initialisation():
    initialising_board()
    set_gratings()

def initialising_board():
    PORT = pyfirmata2.Arduino.AUTODETECT
    board = pyfirmata2.Arduino(PORT)

    #If this fails you can also specify the serial port manually:
    #board = pyfirmata2.Arduino("/dev/ttyACM0")
    # On Linux: /dev/tty.usbserial-A6008rIF, /dev/ttyACM0,
    # On Windows: \\.\COM1, \\.\COM2
    # PORT = '/dev/ttyACM0'

    print("Setting up the connection to the board ...")



def set_gratings():
    # This function creates and uses custom gratings for different trial types

    # motor names and default trials
    MOTOR_NAMES = ["l", "r", "ll", "lr", "rl", "rr"]
    TRIALS = {
        "LL": [0, 90, 0, 90, 90, 90],
        "LR": [0, 90, 90, 0, 90, 90],
        "RL": [90, 0, 90, 90, 0, 90],
        "RR": [90, 0, 90, 90, 90, 0],
        "control": [0, 0, 0, 0, 0, 0]
    }

    # Ask the user for the type of trial or make a custom one
    trial_type = input("Insert type of trial or make a custom one: ").upper()



    if trial_type in TRIALS:
        # If the trial type exists, use the corresponding grating
        grating_angles = TRIALS[trial_type]
            
    else:
        # If the trial type does not exist, ask the user for the custom grating angles
        custom_trial_name = trial_type
        custom_grating_angles = [int(input(f"Insert angle for grating {motor}: ")) for motor in MOTOR_NAMES]
        
        # Update the trials dictionary with the new trial type and grating
        TRIALS[custom_trial_name] = custom_grating_angles
        
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
    pass

def begin_recording():
    pass

def time_starts():
    pass

###
def food_release():
    pass

###
def end_trial():
    pass

def reset_gratings():
    pass

def save_recording():
    pass


def close_port():
    input("Press enter to exit")

    # Close the serial connection to the Arduino
    board.exit()



if __name__ == '__main__':
    main()