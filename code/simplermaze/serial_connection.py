import serial 
import logging

from typing import Optional, Union

import supFun as sf  

class Serial: 
    def __init__(self, port: Optional[str] = "COM5", baud_rate: int = 115200):
        self.logger = logging.getLogger("simple_maze.Serial")
        self.logger.info("created an instance of Serial")
        self.enabled = sf.config.serial_enabled
        self.port = port 
        self.baud_rate = baud_rate
        self.logger.info("attempting to connect to serial port %s with baud rate of %d", self.port, self.baud_rate)

        if self.enabled:  
            self.serial = serial.Serial(self.port, self.baud_rate)
            self.logger.info("connected successfully ")
        else: 
            self.logger.info("serial connection disabled")
    
    def wait_for_ready(self):
        if not self.enabled: 
            self.logger.info("attempted to call `wait_for_ready` while serial communication disabled")
            return 

        while self.serial.in_waiting > 0: 
            self.serial.readline()
        
        self.flush()

    def write(self, message: Union[str, bytes]): 
        if not self.enabled:
            self.logger.info("attempted to call `write` while serial communication disabled")
            return 

        if isinstance(message, str): 
            message = message.encode('utf-8')

        self.serial.write(message)
        self.flush()

    def flush(self):
        if not self.enabled: 
            self.logger.info("attempted to call `flush` while serial communication disabled")
            return 

        self.serial.flush()
    
    def set_motor_to_neutral(self, motor): 
        if not self.enabled:
            self.logger.info("attempted to move motors while serial communication disabled")
            return 

        self.logger.info("attempting to move motor %s to neutral position", motor)
        message = f"grt{motor} 45\n"
        self.write(message)

        