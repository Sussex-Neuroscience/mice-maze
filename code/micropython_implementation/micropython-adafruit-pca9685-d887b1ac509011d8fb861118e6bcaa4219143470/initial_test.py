from machine import Pin
import machine
import servo
import pca9685
import time
import random

random.seed(10)

# pins 
scl=Pin(25)
sda=Pin(26)
 
  # i2c = I2C(1,scl=Pin(26),sda=Pin(25))
i2c=machine.SoftI2C(scl=scl,sda=sda)
  # addresses = i2c.scan()
  # print(addresses)
address = i2c.scan()
print(address)
address = address[0]

time.sleep_ms(10)
  #pca= pca9685.PCA9685(i2c=i2c,address=address)
  #pca.freq=50

# gratings and angles
grating1LIndex=0
grating2LIndex=3

allGratingsIndex = [grating1LIndex,
                    grating2LIndex]

restAngle = 45
horizontalAngle = 90
verticalAngle = -180

# list of trials
listTrials = ("LL", "LR")

    #pca.pwm()
    #pca.duty(i,100)
ser=servo.Servos(i2c=i2c)

  #ser.position(3,duty=4096)

# noting number of trials
numTrials = 10
for trial in range(numTrials):
    print(trial+1)
    
# gratings move to rest position at start of trial
    for grating in allGratingsIndex:
        ser.position(grating,degrees=restAngle)
        print("motor: "+str(grating))
#     ser.position(grating2LIndex,degrees=restAngle)
    time.sleep_ms(500) 

    #randomly select trial and then turn gratings to corresponding position
    gratingDirection1 = (random.choice(listTrials))
        
        gratingTrials(gratingDirection=gratingDirection1)

#detect animal in maze
    #detect which target is crossed (if sensor light is crossed)
        #determine if correct (if animal went to reward location)
            #correct = reward (food dispensed in reward location)
            #incorrect = no reward (no food dispensed)
    
    
    
    
    
    
    
    
    
    
    
    
    
         
#      for i in range(90,0,-10):
#          ser.position(0,degrees=i)
#          ser.position(3,degrees=i)
#          #ser.release(3)
#          time.sleep_ms(100)
     #time.sleep_ms(200)
#ser.release(3) 



# 
# for ind in range(10):
#      print(ind)
#      for i in range(10,100,10):
#          ser.position(grating1LIndex,degrees=i)
#          ser.position(3,degrees=i)
#          #ser.release(3)
#          time.sleep_ms(100)
#          
#      for i in range(90,0,-10):
#          ser.position(0,degrees=i)
#          ser.position(3,degrees=i)
#          #ser.release(3)
#          time.sleep_ms(100)
#      #time.sleep_ms(200)
# ser.release(3) 
 # ser=servo.Servos(i2c=i2c)


def gratingTrials(gratingDirection):
#     for grating in range(allGratingsIndex):
    #LL trial position
    if gratingDirection == "LL"
        for grating in range(allGratingsIndex):
            print(grating)
            ser.position(grating1LIndex,degrees=verticalAngle)
            ser.position(grating2LIndex,degrees=verticalAngle)
            #ser.release(3)
            time.sleep_ms(500)
    
    #LR trial position
    if gratingDirection == "LR" 
        for grating in range(allGratingsIndex):
            print(grating)
            ser.position(grating1LIndex,degrees=verticalAngle)
            ser.position(grating2LIndex,degrees=horizontalAngle)
            #ser.release(3)
            time.sleep_ms(500)