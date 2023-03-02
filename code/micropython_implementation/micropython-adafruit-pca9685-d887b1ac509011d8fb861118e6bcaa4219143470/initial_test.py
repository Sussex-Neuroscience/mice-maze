from machine import Pin
import machine
import servo

scl=Pin(15)
sda=Pin(2)
 

# i2c = I2C(1,scl=Pin(26),sda=Pin(25))
i2c=machine.SoftI2C(scl=scl,sda=sda,freq=400000)
# addresses = i2c.scan()
# print(addresses)

ser=servo.Servos(i2c=i2c,address=0x40)
# ser=servo.Servos(i2c=i2c)
for i in range(15):
    ser.position(index=i,degrees=90)