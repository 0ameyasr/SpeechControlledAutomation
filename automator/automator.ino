//Include the INMP441 sampler for the driver 
#include <driver/i2s.h>

//LED pin setup at different pins for different colours
const int LED_BUILTIN = 2;
const int RED_LED = 18;
const int GREEN_LED = 4;
const int ORANGE_LED = 23;
const int YELLOW_LED = 21;

//ESP32 will ignore all your commands if true
bool ignore = true;

//Setup function
void setup() {
  Serial.begin(9600);
  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(RED_LED,OUTPUT);
  pinMode(GREEN_LED,OUTPUT);
  pinMode(ORANGE_LED,OUTPUT);
  pinMode(YELLOW_LED,OUTPUT);
}

//Perpetual Loop
void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readString();
    command.trim();
    Serial.print("Received command: ");
    Serial.println(command);
    if (command == "on") {
      ignore = false;
      digitalWrite(LED_BUILTIN, HIGH);
      Serial.println("LED turned on");
    }
    else if (command == "zero" && !ignore) {
      digitalWrite(RED_LED, HIGH);
      Serial.println("RED LED turned on");
      delay(750);
      digitalWrite(RED_LED, LOW);
    }
    else if (command == "one" && !ignore) {
      digitalWrite(GREEN_LED, HIGH);
      Serial.println("GREEN LED turned on");
      delay(750);
      digitalWrite(GREEN_LED, LOW);
    }
    else if (command == "two" && !ignore) {
      digitalWrite(ORANGE_LED, HIGH);
      Serial.println("ORANGE LED turned on");
      delay(750);
      digitalWrite(ORANGE_LED, LOW);
    }
    else if (command == "three" && !ignore) {
      digitalWrite(YELLOW_LED, HIGH);
      Serial.println("YELLOW LED turned on");
      delay(750);
      digitalWrite(YELLOW_LED, LOW);
    }
    else if (command == "off") {
      ignore = true;
      digitalWrite(LED_BUILTIN, LOW);
      Serial.println("LED turned off");
    }
    else {
      Serial.println("Command ignored. Ensure that you have issued a command.");
    }
  }
}
