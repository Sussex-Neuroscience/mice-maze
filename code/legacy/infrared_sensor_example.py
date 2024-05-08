from machine import Pin
import time


IR1 = Pin(5,Pin.IN)
IR2 = Pin(18,Pin.IN)


for i in range(2000):
    temp1=IR1.value()
    temp2=IR2.value()
    print("IR1",temp1)
    print("IR2",temp2)
    time.sleep_ms(50)
    