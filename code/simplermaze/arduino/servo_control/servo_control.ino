#include <SerialCommand.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// called this way, it uses the default address 0x40
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();
// you can also call it with a different address you want
//Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x41);
// you can also call it with a different address and I2C interface
//Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40, Wire);

// Depending on your servo make, the pulse width min and max may vary, you 
// want these to be as small/large as possible without hitting the hard stop
// for max range. You'll have to tweak them as necessary to match the servos you
// have!
#define SERVOMIN  150 // This is the 'minimum' pulse length count (out of 4096)
#define SERVOMAX  600 // This is the 'maximum' pulse length count (out of 4096)
#define USMIN  600 // This is the rounded 'minimum' microsecond length based on the minimum pulse of 150
#define USMAX  2400 // This is the rounded 'maximum' microsecond length based on the maximum pulse of 600
#define SERVO_FREQ 50 // Analog servos run at ~50 Hz updates

// our servo # counter
uint8_t servoNum = 0;

#define IRsensor1 8
#define IRsensor2 9
#define IRsensor3 10
#define IRsensor4 11

//int servoNum = 0;
int IRsensor = 0;
int degree = 0;
int IRvalue = 0;
int time1 = 0;
int time2 = 0;
int pulseLen = 0;
int pelletDropped = 0;
int rewPulseLen1 = map(60, 0, 180, SERVOMIN, SERVOMAX);
int rewPulseLen2 = map(15, 0, 180, SERVOMIN, SERVOMAX);

SerialCommand sCmd;     // The demo SerialCommand object


void setup() {
  pinMode(IRsensor1,INPUT);
  pinMode(IRsensor2,INPUT);
  pinMode(IRsensor3,INPUT);
  pinMode(IRsensor4,INPUT);

  Serial.begin(115200);

  
  sCmd.addCommand("grating",  grating); // turn servos holding gratings (grating 0 45) // turn servo 0 to position 45 degrees
  sCmd.addCommand("reward", reward); // activate reward routine (reward 10) // start reward routine on servo 10

  // generic functions for development/understanding the parsing library
  sCmd.addCommand("HELLO", sayHello);        // Echos the string argument back
  sCmd.addCommand("P",     processCommand);  // Converts two arguments to integers and echos them back
  sCmd.addDefaultHandler(unrecognized);      // Handler for command that isn't matched  (says "What?")


  //servo library initialization
  pwm.begin();
  /*
   * In theory the internal oscillator (clock) is 25MHz but it really isn't
   * that precise. You can 'calibrate' this by tweaking this number until
   * you get the PWM update frequency you're expecting!
   * The int.osc. for the PCA9685 chip is a range between about 23-27MHz and
   * is used for calculating things like writeMicroseconds()
   * Analog servos run at ~50 Hz updates, It is importaint to use an
   * oscilloscope in setting the int.osc frequency for the I2C PCA9685 chip.
   * 1) Attach the oscilloscope to one of the PWM signal pins and ground on
   *    the I2C PCA9685 chip you are setting the value for.
   * 2) Adjust setOscillatorFrequency() until the PWM update frequency is the
   *    expected value (50Hz for most ESCs)
   * Setting the value here is specific to each individual I2C PCA9685 chip and
   * affects the calculations for the PWM update frequency. 
   * Failure to correctly set the int.osc value will cause unexpected PWM results
   */
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(SERVO_FREQ);  // Analog servos run at ~50 Hz updates

  delay(10);

}// setup

void loop() {
  sCmd.readSerial();     // We don't do much, just process serial commands
}

///servo callback functions

void grating(){
  //int aNumber;
  char *arg;
  arg = sCmd.next();
  if (arg != NULL) {
     servoNum = atoi(arg);
  }//if

  arg = sCmd.next();
  if (arg != NULL) {
     degree = atoi(arg);
  }//if

     pulseLen = map(degree, 0, 180, SERVOMIN, SERVOMAX);
     pwm.setPWM(servoNum, 0, pulseLen);
     
}//end grating


void reward(){
  int aNumber;
  char *arg;
  arg = sCmd.next();
  if (arg != NULL) {
     aNumber = atoi(arg);
      if (aNumber==8){
        IRsensor=IRsensor1;
      }
      if (aNumber==9){
        IRsensor=IRsensor2;
      }
      if (aNumber==10){
        IRsensor=IRsensor3;
      }
      if (aNumber==11){
        IRsensor=IRsensor4;
      }
     pelletDropped = 0;
     while (pelletDropped==0){
      pwm.setPWM(servoNum, 0, rewPulseLen1);
      time1=millis();
      time2=millis();
      while (time2-time1<100){
        IRvalue = digitalRead(IRsensor);
        if(IRvalue==1){
          pelletDropped=1;
          break;
        }//if
        time2=millis();
      }// while
      if (pelletDropped==0){
        pwm.setPWM(servoNum, 0, rewPulseLen2);

      time1=millis();
      time2=millis();
      while (time2-time1<100){
        IRvalue = digitalRead(IRsensor);
        if(IRvalue==1){
          pelletDropped=1;
          break;
        }//if
        time2=millis();
      }//while


     }// if
     
     
  }//if
  }
}//end Reward

/////////////////////////////////////////



void sayHello() {

  char *arg;
  arg = sCmd.next();    // Get the next argument from the SerialCommand object buffer
  if (arg != NULL) {    // As long as it existed, take it
    Serial.print("Hello ");
    Serial.println(arg);
  }
  else {
    Serial.println("Hello, whoever you are");
  }
}


void processCommand() {
  int aNumber;
  char *arg;

  Serial.println("We're in processCommand");
  arg = sCmd.next();
  if (arg != NULL) {
    aNumber = atoi(arg);    // Converts a char string to an integer
    Serial.print("First argument was: ");
    Serial.println(aNumber);
  }
  else {
    Serial.println("No arguments");
  }

  arg = sCmd.next();
  if (arg != NULL) {
    aNumber = atol(arg);
    Serial.print("Second argument was: ");
    Serial.println(aNumber);
  }
  else {
    Serial.println("No second argument");
  }
}

// This gets set as the default handler, and gets called when no other command matches.
void unrecognized(const char *command) {
  Serial.println("What?");
}


/*
void S90(){
  int aNumber;
  char *arg;
  arg = sCmd.next();
  if (arg != NULL) {
     aNumber = atoi(arg);

     //pulseLen = map(90, 0, 180, SERVOMIN, SERVOMAX);
     //pwm.setPWM(servoNum, 0, pulseLen);
     
     
  }//if
     Serial.print("servo90");
     Serial.println(aNumber);
}//end S90
*/