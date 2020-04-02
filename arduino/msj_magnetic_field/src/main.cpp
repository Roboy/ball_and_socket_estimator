#include <ParallaxServo.hpp>
#define USE_USBCON
#include "ros.h"
#include "std_msgs/Int16.h"

ParallaxServo motor[] = {ParallaxServo(3,6),ParallaxServo(2,5)};
Servo servo;

ros::NodeHandle nh;

void motor0_CB( const std_msgs::Int16& msg){
  if(msg.data>=90 && msg.data<=270)
    motor[0].SetPoint(msg.data);
}

void motor1_CB( const std_msgs::Int16& msg){
  if(msg.data>=90 && msg.data<=270)
    motor[1].SetPoint(msg.data);
}

void motor2_CB( const std_msgs::Int16& msg){
  if(msg.data>=20 && msg.data<=160)
    servo.write(msg.data);
}

ros::Subscriber<std_msgs::Int16> sub[] = {ros::Subscriber<std_msgs::Int16>("motor_command/motor0", &motor0_CB ),
                                          ros::Subscriber<std_msgs::Int16>("motor_command/motor1", &motor1_CB ),
                                          ros::Subscriber<std_msgs::Int16>("motor_command/motor2", &motor2_CB )
                                        };

std_msgs::Int16 status_msg;
ros::Publisher p[] = {ros::Publisher("motor_status/motor0", &status_msg),ros::Publisher("motor_status/motor1", &status_msg),ros::Publisher("motor_status/motor2", &status_msg)};

void setup() {
  servo.attach(4);
  nh.initNode();
  for(int i=0;i<3;i++){
    nh.advertise(p[i]);
    nh.subscribe(sub[i]);
  }
  motor[0].SetPoint(180);
  motor[1].SetPoint(180);
  servo.write(90);
}

int counter;

void loop() {
  for(int i=0;i<2;i++){
    motor[i].update();
  }
  // delay(20);
  // delay(1000);
  if(counter++%10==0){
    for(int i=0;i<3;i++){
      if(i<2)
        status_msg.data = motor[i].GetAngle();
      else
        status_msg.data = servo.read();
      p[i].publish(&status_msg);
    }
  }
  nh.spinOnce();
}
