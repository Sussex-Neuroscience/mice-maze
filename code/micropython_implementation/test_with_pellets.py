from machine import Pin
import machine
import servo
import pca9685
import time



detector_food1=Pin(15,Pin.IN)

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

motorReward1=0




restAngle = 45
horizontalAngle = 90
verticalAngle = -180

    #pca.pwm()
    #pca.duty(i,100)
ser=servo.Servos(i2c=i2c)

#ser.position(3,duty=4096)

numTrials = 10
#Gratings move to rest at start of trial
for trial in range(1,numTrials+1):
    print("trial "+str(trial))
    giveReward=True
    
    while giveReward:
        ser.position(motorReward1,degrees=45)
        time1 = time.ticks_ms()
        time2 = time.ticks_ms()
        while time2-time1<200:
            #print("here")
            notDropped = detector_food1.value()
            print(notDropped)
            time2 = time.ticks_ms()
            #print(time2-time1)
            if notDropped:
                giveReward=False
                
        ser.position(motorReward1,degrees=60)
        time1 = time.ticks_ms()
        time2 = time.ticks_ms()
        while time2-time1<200:
            notDropped = detector_food1.value()
            time2 = time.ticks_ms()
            if notDropped:
                giveReward=False
        
        time.sleep_ms(500) 