from machine import Pin
import machine
import servo
import pca9685
import time


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


    #pca.pwm()
    #pca.duty(i,100)
ser=servo.Servos(i2c=i2c)
#ser.position(3,duty=4096)

for ind in range(10):
     print(ind)
     for i in range(10,100,10):
         ser.position(3,degrees=i)
         #ser.release(3)
         time.sleep_ms(100)
         
     for i in range(90,0,-10):
         ser.position(3,degrees=i)
         #ser.release(3)
         time.sleep_ms(100)
     #time.sleep_ms(200)
ser.release(3) 
 # ser=servo.Servos(i2c=i2c)
