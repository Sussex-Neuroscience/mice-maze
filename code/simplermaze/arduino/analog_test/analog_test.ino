

//reward location A
#define IRsensorA A0
//2

//reward location B
#define IRsensorB A1//3//15

//reward location C
#define IRsensorC A3//4//17

//reward location D
#define IRsensorD A2//5//16




void setup() {
  // put your setup code here, to run once:
  //pinMode(IRsensorA,INPUT);
  //pinMode(IRsensorB,INPUT);
  //pinMode(IRsensorC,INPUT);
  //pinMode(IRsensorD,INPUT);
  Serial.begin(115200);
}

void loop() {
  Serial.print("A: ");
  Serial.println(analogRead(IRsensorA));
  Serial.print("B: ");
  Serial.println(analogRead(IRsensorB));
  Serial.print("C: ");
  Serial.println(analogRead(IRsensorC));
  Serial.print("D: ");
  Serial.println(analogRead(IRsensorD));
  delay(500);
  // put your main code here, to run repeatedly:

}
