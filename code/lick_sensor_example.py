from machine import Pin
from machine import ADC
import time

lick1 = ADC(Pin(25))
lick1.atten(ADC.ATTN_11DB) 
valve1 = Pin(18,Pin.OUT)

for i in range(2000):
    temp = lick1.read()
    print(temp)
    if temp>0:
        valve1.on()
        time.sleep_ms(200)
        valve1.off()
    
    time.sleep_ms(10)
    