/* ParallaxServoControl
 * Controls the Parallax 360 Feedback Servo using a P-controller
 * by mriveralee - 7/2/2019
 * Based on datasheet for Parallax 360 Servo Motor
 * See here: https://www.parallax.com/sites/default/files/downloads/900-00360-Feedback-360-HS-Servo-v1.2.pdf
 *
 */
#include <Arduino.h>
#include <Servo.h>

class ParallaxServo{
public:
  ParallaxServo(int servo_pin, int feedback_pin):SERVO_PIN(servo_pin),SERVO_FEEDBACK_PIN(feedback_pin){
    SERVO.attach(SERVO_PIN);
    pinMode(SERVO_FEEDBACK_PIN, INPUT);
  }
  void update(){
    // Run pulseWidth measuring to figure out the current angle of the servo
    unsigned long tHigh = pulseIn(SERVO_FEEDBACK_PIN, HIGH);
    unsigned long tLow = pulseIn(SERVO_FEEDBACK_PIN, LOW);
    unsigned long  tCycle = tHigh + tLow;
    // Check if our cycle time was appropriate
    if (!(tCycle > 1000 && tCycle < 1200)) {
      // Invalid cycle time, so try pulse measuring again
      // Serial.println("Invalid cycle time");
      return;
    }
    // Calculate the duty cycle of the pulse
    float dutyCycle = (DUTY_SCALE) * ((float) tHigh / tCycle);
    float maxUnitsForCircle = UNITS_IN_FULL_CIRCLE - 1;

    // Calculate exact angle of servo
    currentAngle = maxUnitsForCircle - ((dutyCycle - DUTY_CYCLE_MIN) * UNITS_IN_FULL_CIRCLE) / ((DUTY_CYCLE_MAX - DUTY_CYCLE_MIN) + 1);

    // Clip current angle if we're somehow above or below range
    if (currentAngle < 0) {
      currentAngle = 0;
    } else if (currentAngle > maxUnitsForCircle) {
      currentAngle = maxUnitsForCircle;
    }

    // Handle quadrant wrap q1 -> q4 and q4 -> q1, to count turns
    if ((currentAngle < ANGLE_Q2_MIN) && (prevAngle > ANGLE_Q3_MAX)) {
      turns += 1;
    } else if ((prevAngle < ANGLE_Q2_MIN) && (currentAngle > ANGLE_Q3_MAX)) {
      turns -= 1;
    }

    // Save previous position
    prevAngle = currentAngle;
    errorAngle = targetAngle - currentAngle;
    integral += errorAngle*Ki;
    if(integral>integral_max)
      integral = integral_max;
    else if(integral<integral_min)
        integral = integral_min;
    // Simple P Controller
    float outputSpeed = errorAngle * Kp + (errorAngle-errorAngle_prev) * Kd + integral;
    errorAngle_prev = errorAngle;

    if (outputSpeed > output_max) {
      outputSpeed = output_max;
    } else if (outputSpeed < output_min) {
      outputSpeed = output_min;
    }
    // char str[200];
    // sprintf(str,"outputSpeed %f", outputSpeed);
    // Serial.println(str);

    outputSpeed = HOLD_STILL_PULSE_SPEED_US + outputSpeed;
    SERVO.writeMicroseconds(outputSpeed);
  }

  void SetPoint(int setpoint){
      targetAngle = setpoint;
  }
  int GetAngle(){
    return currentAngle;
  }
private:
  Servo SERVO;                        // The Parallax Servo
  byte SERVO_PIN = 2;                // The servo pin
  byte SERVO_FEEDBACK_PIN = 5;        // The feedback pin (from servo)

  float DUTY_SCALE = 1000;
  unsigned long DUTY_CYCLE_MIN = 29;  // 2.9% * DUTY_SCALE
  unsigned long DUTY_CYCLE_MAX = 971; // 9.71% * DUTY_SCALE
  float UNITS_IN_FULL_CIRCLE = 360;   // Because 360 degrees are in a circle

  // Tune the vars below for controlling how fast the servo gets to the right place and stays there
  int ERROR_ANGLE_OFFSET_US = 23;
  float integral = 0, integral_max = 10, integral_min = -10;
  float Kp = 1, Ki = 0.01, Kd = 0;

  int output_max = 200;    // Going Counter-clockwise a bit - can be smaller ( < -40)
  int output_min = -200;     // Going Clockwise  a bit - can be bigger (> 40)
  int HOLD_STILL_PULSE_SPEED_US = 1500;   // HOLD_STILL is for keeping the servo in place (no movement, don't change)

  // Angles for different quadrants around the unit circle (for counting number of turns)
  int ANGLE_Q2_MIN = 90;
  int ANGLE_Q3_MAX = 270;

  // Holds the current state
  int targetAngle = 0;                  // The angle we want the servo to go to
  int currentAngle = 0;                   // The angle the servo is at
  int prevAngle = 0;                      // The last angle the servo had
  int errorAngle = 0, errorAngle_prev = 0;                     // How off we are from the target angle
  int turns = 0;                          // How many times we've gone around the circle
};
