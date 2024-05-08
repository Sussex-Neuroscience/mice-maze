""" servo pyfirmata"""

import pyfirmata
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

#['L','R','LL','LR','RR','RL']
#[2,3,4,5,6,7]

#sequence of pins
digital_pins=[2,3,4,5,6,7]

#declare pins
for i in digital_pins:
    exec('global pin'+str(i))

#correspondence of pins and sequence decision the animal has to make
def moveable_grating(val_direction):
    if val_direction == 'L' : return 2
    if val_direction == 'R' : return 3
    if val_direction == 'LL' : return 4
    if val_direction == 'LR' : return 5
    if val_direction == 'RL' : return 6
    if val_direction == 'RR' : return 7

#used only for initial tuning of the moveable gratings
def move_servo(angle,servo):
    exec('pin'+str(moveable_grating(servo))+'.write(angle)')

#declare and initialise the arduino board
board=pyfirmata.Arduino("/dev/ttyACM0")
start_board = pyfirmata.util.Iterator(board)
start_board.start()
print("Communication with tunable walls started")

#define the pin to write in the board
for i in digital_pins:
    exec('pin'+str(i)+' = board.get_pin("d:'+ str(i) + ':s")')

#use the tuning csv file to interpolate angle values for the walls
def tuning(val_direction,angle):
    df = pd.read_csv('/home/maze_/Desktop/codes_maze/maze_project/angle_tuning.csv')
    x=np.array(df[df.direction == val_direction].angle).reshape((-1, 1))
    y=np.array(df[df.direction == val_direction].tuning)
    model = LinearRegression().fit(x, y)
    val=np.floor(model.predict(np.array([angle]).reshape((-1, 1)))[0])
    if val < 0: val = 0
    return val

#function to move servo with interpolated value
def move_servo_tuned(angle,val_direction):
    precise_angle=tuning(val_direction,angle)
    exec('pin'+str(moveable_grating(val_direction))+'.write('+str(precise_angle)+')')

# i=90
# move_servo_tuned(180-i,'R')
# move_servo_tuned(180-i,'RR')

# i=90
# move_servo_tuned(i,'L')
# move_servo_tuned(i,'LL')

def pannel_sort(z):
    if z=='allzero':
        angle=0
        time.sleep(.25)
        move_servo_tuned(angle,'L')
        time.sleep(.25)
        move_servo_tuned(angle,'LL')
        time.sleep(.25)
        move_servo_tuned(angle,'RL')
        time.sleep(.25)
        move_servo_tuned(180-angle,'R')
        time.sleep(.25)
        move_servo_tuned(180-angle,'RR')
        time.sleep(.25)
        move_servo_tuned(180-angle,'LR')
        time.sleep(.25)
    if z == 'RR':
        angle=0
        time.sleep(.25)
        move_servo_tuned(angle,'L')
        time.sleep(.25)
        move_servo_tuned(angle,'LL')
        time.sleep(.25)
        move_servo_tuned(angle,'RL')
        time.sleep(.25)
        move_servo_tuned(180-angle,'LR')
        time.sleep(.25)
        angle=90
        move_servo_tuned(180-angle,'R')
        time.sleep(.25)
        move_servo_tuned(180-angle,'RR')
        time.sleep(.25)
    if z == 'LL':
        angle=0
        time.sleep(.25)
        move_servo_tuned(angle,'RL')
        time.sleep(.25)
        move_servo_tuned(180-angle,'R')
        time.sleep(.25)
        move_servo_tuned(180-angle,'RR')
        time.sleep(.25)
        move_servo_tuned(180-angle,'LR')
        time.sleep(.25)
        angle=90
        move_servo_tuned(angle,'L')
        time.sleep(.25)
        move_servo_tuned(angle,'LL')
        time.sleep(.25)
    if z =='LR':
        angle=0
        time.sleep(.25)
        move_servo_tuned(angle,'LL')
        time.sleep(.25)
        move_servo_tuned(angle,'RL')
        time.sleep(.25)
        move_servo_tuned(180-angle,'R')
        time.sleep(.25)
        move_servo_tuned(180-angle,'RR')
        angle=90
        time.sleep(.25)
        move_servo_tuned(angle,'L')
        time.sleep(.25)
        move_servo_tuned(180-angle,'LR')
        time.sleep(.25)
    if z == 'RL':
        angle=0
        time.sleep(.25)
        move_servo_tuned(angle,'L')
        time.sleep(.25)
        move_servo_tuned(angle,'LL')
        time.sleep(.25)
        move_servo_tuned(180-angle,'RR')
        time.sleep(.25)
        move_servo_tuned(180-angle,'LR')
        time.sleep(.25)
        angle=90
        move_servo_tuned(180-angle,'R')
        time.sleep(.25)
        move_servo_tuned(angle,'RL')
        time.sleep(.25)
    return

def close_board():
    start_board.board.exit()
    print("Communication with tunable walls finished")
    return

#pannel_sort('allzero')
#pannel_sort('RR')
#pannel_sort('LL')
