

"""
first version of the micropython implementation of the maze. there are 2 pairs of gratings, 
where the animals can get information for 2 decision points. This leads to 4 potential reward sites.

possible combinations for reward are:
RR (two right turns) Reward Location 1
RL (one right one left) Reward Location 2
LR (one left one right) Reward Location 3
LL (two left turns) Reward Location 4

intertrial interval starts
the trial combination is selected (RR,RL, LR, LL)
all gratings move to tilted position
pause 100ms
all gratings move to the specific combination for that trial
system waits for animal to move in the maze (counting time)
once animal is in, system sets that as time point zero.
from now on, every time the animal passes through the gratings light gate, it gets time stamped
if animal passes the correct reward light gate, then reward is given, otherwise nothing happens, and the animal has to move out of the maze to start another trial. 

"""

import machine
import time



#each grating can have 3 positions (vertical, horizontal and tilted). there are two pairs of grating on the maze


def reward:
    pass

