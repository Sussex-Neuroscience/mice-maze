#rshell --port /dev/ttyUSB0 --baud 115200 repl
#ampy --port /dev/ttyUSB0 --baud 115200 run sensor.py
from machine import Pin
import sys 
import time
IR1 = Pin(22,Pin.IN)
IR2 = Pin(23,Pin.IN)

def convertMillis(milli):
   ms = milli % 1000
   milli = milli-ms
   seconds=(milli/1000)%60
   milli = milli - seconds*1000
   minutes=(milli/(1000*60))%60
   milli = milli - minutes*1000
   hours=(milli/(1000*60*60))%24
   return str(int(hours))+':'+str(int(minutes))+':'+str(int(seconds))+':'+str(ms)

maze=[[0,0],[1,0],[1,1]]
cage=[[0,0],[0,1],[1,1]]
run=[1,1]
run_old=[1,1]
mod=0
seq_change=[[1,1],[1,1],[1,1]]
reset=[[1,1],[1,1],[1,1]] 
tick=time.ticks_ms()
print("position, timestamps, tick, ready")
print("cage,0:0:0:0,0,ready")

maze_visits = 0
cage_visits = 0

last="cage"
while True:
    run[0]=IR1.value()
    run[1]=IR2.value()
    if (run) != (run_old):
        if run == [0,0]:
            mod=0
        seq_change[mod%3]=run.copy()
        mod=mod+1
    if seq_change == cage:
        if last == "maze":
                print("cage," + convertMillis(time.ticks_diff(time.ticks_ms(),tick)) + "," + str(time.ticks_diff(time.ticks_ms(),tick))+",ready")
                seq_change = reset.copy()
                last = "cage"
    if seq_change == maze:
        if last == "cage":
            print("maze," + convertMillis(time.ticks_diff(time.ticks_ms(),tick)) + "," + str(time.ticks_diff(time.ticks_ms(),tick)) +",ready")
            seq_change = reset.copy()
            last = "maze"
    run_old = run.copy()
    time.sleep_ms(15)

print("done")