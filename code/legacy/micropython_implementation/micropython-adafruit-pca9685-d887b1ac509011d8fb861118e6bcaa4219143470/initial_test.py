from machine import Pin
import machine
import servo
import pca9685
import time
import random

random.seed(10)

# pins
# pins for the servo control board
sclPin=Pin(25)
sdaPin=Pin(26)


# pins for the light detectors
tunnel1Pin=Pin(15,Pin.IN)
tunnel2Pin=Pin(2,Pin.IN)
LLrewardPin=Pin(16,Pin.IN)
LRrewardPin=Pin(17,Pin.IN)
RLrewardPin=Pin(5,Pin.IN)
RRrewardPin=Pin(18,Pin.IN)
grating1Pin=Pin(19,Pin.IN)
grating2Pin=Pin(21,Pin.IN)
 
 # i2c = I2C(1,scl=Pin(26),sda=Pin(25))
i2c=machine.SoftI2C(scl=sclPin,sda=sdaPin)
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
    
        
#two light detectors in tunnel - either represent values 0 or 1 (0 when empty, 1 when crossed)
    #entry into maze = 00, 10, 11, 01
    #exit out of maze = 01, 11, 10, 00

# light detector in tunnel senses animal entering maze
# recording whether tunnel 1 pin value is 0 or 1
#animal in flag recording whether animal is inside or outside maze
tunnel1Detector = tunnel1Pin.value()
animalInFlag = 0
while tunnel1Detector > 0:
    tunnel1Detector = tunnel1Pin.value()
animalInFlag = 1

    if animalInFlag = 1: # move gratings, start trial and record animal
        # gratings move to rest position at start of trial
        for grating in allGratingsIndex:
            ser.position(grating,degrees=restAngle)
            print("motor: "+str(grating))
    #     ser.position(grating2LIndex,degrees=restAngle)
        time.sleep_ms(500) 
        #randomly select trial and then turn gratings to corresponding position
        gratingDirection1 = (random.choice(listTrials))
            gratingTrials(gratingDirection=gratingDirection1)
            
       #detecting if animal is in LL area    
        LLDetector = LLrewardPin.value()
        animalInLL = 0
        while LLDetector > 0:
            LLDetector = LLrewardPin.value()
        animalInLL = 1
        while LLDetector == 0:
            LLDetector = LLrewardPin.value()
        animalInLL = 0
        
        #detecting if animal is in LR area    
        LRDetector = LRrewardPin.value()
        animalInLR = 0
        while LRDetector > 0:
            LRDetector = LRrewardPin.value()
        animalInLR = 1
        while LRDetector == 0:
            LRDetector = LRrewardPin.value()
        animalInLR = 0
        
        #detecting if animal is in RL area    
        RLDetector = RLrewardPin.value()
        animalInRL = 0
        while RLDetector > 0:
            RLDetector = RLrewardPin.value()
        animalInRL = 1
        while RLDetector == 0:
            RLDetector = RLrewardPin.value()
        animalInRL = 0
        
        #detecting if animal is in RR area    
        RRDetector = RRrewardPin.value()
        animalInRR = 0
        while RRDetector > 0:
            RRDetector = RRrewardPin.value()
        animalInRR = 1
        while RRDetector == 0:
            RRDetector = RRrewardPin.value()
        animalInRR = 0
            
while tunnel1Detector == 0:
    tunnel1Detector = tunnel1Pin.value()
animalInFlag = 0

def detect_animal(pin=RLrewardPin):
    detector = pin.value()
    animalIn = 0
    while detector > 0:
        detector = pin.value()
    animalIn = 1
    while detector == 0:
        detector = pin.value()
    animalIn = 0
    return animalIn

RLanimalIn = detect_animal(pin=RLrewardPin)

    #elif animalInFlag = 0: #stop trial, end recording, wait for animal to enter maze again
        #tunnel1Detector = tunnel1Pin.value()
            
        
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