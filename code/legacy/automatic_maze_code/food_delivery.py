""" servo pyfirmata"""

import pyfirmata
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

#sequence of pins
food_pins=[5,6,9,10]

#declare pins
for i in food_pins:
    exec('global fpin'+str(i))

#correspondence of pins and sequence decision the animal has to make
def move_food(reward_place):
    if reward_place == 'LL' : return 5
    if reward_place == 'LR' : return 6
    if reward_place == 'RL' : return 9
    if reward_place == 'RR' : return 10

#used only for initial tuning of the moveable gratings
def move_food_order(angle,servo):
    exec('fpin'+str(move_food(servo))+'.write(angle)')

#declare and initialise the arduino board_food
board_food=pyfirmata.Arduino("/dev/ttyACM1")
start_board_food = pyfirmata.util.Iterator(board_food)
start_board_food.start()

print("Communication with food servo started")

#define the pin to write in the board_food
for i in food_pins:
    exec('fpin'+str(i)+' = board_food.get_pin("d:'+ str(i) + ':s")')

# for i in [9,10]:
#     exec('fpin'+str(i)+' = board_food.get_pin("d:'+ str(i) + ':p")')
# #use the tuning csv file to interpolate angle values for the walls 

def food_moving(reward_place,reward_order):
    df = pd.read_csv('/home/maze_/Desktop/codes_maze/maze_project/food_tuning.csv')
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

move_food_order(0,'RR')
move_food_order(0,'RL')
move_food_order(0,'LL')
move_food_order(0,'LR')
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


# move_food_order(0,'LR')

# for i in range(0,23,2):
#      time.sleep(0.1)
#      move_food_order(i,'LR')
# time.sleep(1)
# for i in range(23,43,2):
#     time.sleep(0.1)
#     move_food_order(i,'LR')
# time.sleep(1)
# for i in range(43,63,2):
#     time.sleep(0.1)
#     move_food_order(i,'LR')
# time.sleep(1)
# for i in range(63,83,2):
#     time.sleep(0.1)
#     move_food_order(i,'LR')
# time.sleep(1)
# for i in range(83,106,2):
#     time.sleep(0.1)
#     move_food_order(i,'LR')
# time.sleep(1)
# for i in range(106,130,2):
#     time.sleep(0.1)
#     move_food_order(i,'LR')
# time.sleep(1)
# for i in range(130,150,2):
#     time.sleep(0.1)
#     move_food_order(i,'LR')
# time.sleep(1)
# for i in range(150,175,2):
#     time.sleep(0.1)
#     move_food_order(i,'LR')
# time.sleep(1)


# for i in range(180,0,-2):
#     time.sleep(0.1)
#     move_food_order(i,'LR')
