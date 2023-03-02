from machine import Pin
import machine
import servo
import pca9685
import time


scl=Pin(25)
sda=Pin(26)
 

# i2c = I2C(1,scl=Pin(26),sda=Pin(25))
i2c=machine.SoftI2C(scl=scl,sda=sda,freq=100000)
# addresses = i2c.scan()
# print(addresses)
address = i2c.scan()
print(address)
address = address[0]

time.sleep_ms(10)
pca= pca9685.PCA9685(i2c=i2c,address=address)
pca.freq=50

for i in range(16):
    #pca.pwm()
    pca.duty(i,100)
# ser=servo.Servos(i2c=i2c,address=address)
# for ind in range(10):
#     print(ind)
#     for i in range(0,120,5):
#         ser.position(0,degrees=i)
#         time.sleep_ms(5)
#     for i in range(120,0,-5):
#         ser.position(0,degrees=i)
#         time.sleep_ms(5)
#     time.sleep_ms(200)

# ser=servo.Servos(i2c=i2c)
