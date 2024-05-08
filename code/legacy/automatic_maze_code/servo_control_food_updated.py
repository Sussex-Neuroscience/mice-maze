""" servo pyfirmata"""

import pyfirmata
import time 
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

#sequence of pins
food_pins=[2,4,7,8]

#declare pins
for i in food_pins:
    exec('global fpin'+str(i))

#correspondence of pins and sequence decision the animal has to make 
def move_food(reward_place):
    if reward_place == 'LL' : return 2
    if reward_place == 'LR' : return 4
    if reward_place == 'RL' : return 7
    if reward_place == 'RR' : return 8

#used only for initial tunning of the moveable gratings
def move_food_order(angle,servo): 
    exec('fpin'+str(move_food(servo))+'.write(angle)')

#declare and initialise the arduino board_food
board_food=pyfirmata.Arduino("/dev/Food")
start_board_food = pyfirmata.util.Iterator(board_food)
start_board_food.start()

print("Communication with food servo started")

#define the pin to write in the board_food
for i in food_pins:
    exec('fpin'+str(i)+' = board_food.get_pin("d:'+ str(i) + ':s")')

# for i in [9,10]:
#     exec('fpin'+str(i)+' = board_food.get_pin("d:'+ str(i) + ':p")')
# #use the tunning csv file to interpolate angle values for the walls 

def food_moving(reward_place,reward_order):
    df = pd.read_csv('/home/maze_/Desktop/codes_maze/maze_project/food_tunning.csv')
    if reward_order < 9:
        x=np.array(df[df.direction == reward_place].angle).reshape((-1, 1))[reward_order-1][0]
        x_next=np.array(df[df.direction == reward_place].angle).reshape((-1, 1))[reward_order][0]
        for i in np.arange(x,x_next,1):
            time.sleep(0.1)
            exec('fpin'+str(move_food(reward_place))+'.write('+str(i)+')')
    else:
        x=np.array(df[df.direction == reward_place].angle).reshape((-1, 1))[8][0]    
        for i in np.arange(x,0,-1):
            time.sleep(0.1)
            exec('fpin'+str(move_food(reward_place))+'.write('+str(i)+')')


def close_board_food():
    start_board_food.board.exit()
    print("Communication with food servo terminated")
    return

time.sleep(1.5)
move_food_order(0,'RR')
time.sleep(1.5)
move_food_order(0,'RL')
time.sleep(1.5)
move_food_order(0,'LL')
time.sleep(1.5)
move_food_order(0,'LR')
time.sleep(1.5)

#food_moving("RL",4)  #use this line for testing, let it commnetd when using the code with a mouse
def test_tray():
    for i in range(0,10):
        food_moving("RR",i)
        time.sleep(2.5)
        food_moving("RL",i)
        time.sleep(2.5)
        food_moving("LL",i)
        time.sleep(2.5)
        food_moving("LR",i)
        time.sleep(2.5)

# move_food_order(0,'RR')
# time.sleep(1)
# move_food_order(25,'RR')
# time.sleep(1)
# move_food_order(40,'RL')
# time.sleep(1)
# move_food_order(60,'RR')
# time.sleep(1)
# move_food_order(80,'RR')
# time.sleep(1)
# move_food_order(110,'RR')
# time.sleep(1)
# move_food_order(130,'RR')
# time.sleep(1)
# move_food_order(150,'RR')
# time.sleep(1)
# move_food_order(170,'RR')
# time.sleep(1)
# move_food_order(190,'RR')
# time.sleep(1)
# time.sleep(1)
# move_food_order(0,'RL')
# time.sleep(1)
# for i in range(0,24,2):
#      time.sleep(0.1)
#      move_food_order(i,'RL')
# time.sleep(1)
# for i in range(24,40,2):
#     time.sleep(0.1)
#     move_food_order(i,'RL')
# time.sleep(1)
# for i in range(40,58,2):
#     time.sleep(0.1)
#     move_food_order(i,'RL')
# time.sleep(1)
# for i in range(58,73,2):
#     time.sleep(0.1)
#     move_food_order(i,'RL')
# time.sleep(1)
# for i in range(73,96,2):
#     time.sleep(0.1)
#     move_food_order(i,'RL')
# time.sleep(1)
# for i in range(96,126,2):
#     time.sleep(0.1)
#     move_food_order(i,'RL')
# time.sleep(1)
# for i in range(126,144,2):
#     time.sleep(0.1)
#     move_food_order(i,'RL')
# time.sleep(1)
# for i in range(144,164,2):
#     time.sleep(0.1)
#     move_food_order(i,'RL')    
# time.sleep(1)
   

# for i in range(164,0,-2):
#     time.sleep(0.1)
#     move_food_order(i,'RL')