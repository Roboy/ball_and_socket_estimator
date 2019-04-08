#!/usr/bin/env python
import rospy
import std_msgs
import time
import random
rospy.init_node('msj_platform_grid_map')

sphere_axis0_max = 0.4
sphere_axis1_max = 0.4
sphere_axis2_max = 1.5

sphere0 = rospy.Publisher('/sphere_axis0/sphere_axis0/target', std_msgs.msg.Float32 , queue_size=1)
sphere1 = rospy.Publisher('/sphere_axis1/sphere_axis1/target', std_msgs.msg.Float32 , queue_size=1)
sphere2 = rospy.Publisher('/sphere_axis2/sphere_axis2/target', std_msgs.msg.Float32 , queue_size=1)

roll = 0
pitch = 0
yaw = 0

roll_dir = False
pitch_dir = False
yaw_dir = False

while not rospy.is_shutdown():
    msg = std_msgs.msg.Float32()
    msg.data = roll
    sphere0.publish(msg)
    msg.data = pitch
    sphere1.publish(msg)
    msg.data = yaw
    sphere2.publish(msg)
    if roll_dir:
        roll = roll + random.uniform(0.002, 0.01)
    else:
        roll = roll - random.uniform(0.002, 0.01)

    if pitch_dir:
        pitch = pitch + random.uniform(0.02, 0.01)
    else:
        pitch = pitch - random.uniform(0.02, 0.01)

    if yaw_dir:
        yaw = yaw + random.uniform(0.0002, 0.001)
    else:
        yaw = yaw - random.uniform(0.0002, 0.001)

    if abs(roll)>=sphere_axis0_max:
        roll_dir = not roll_dir
    if abs(pitch)>=sphere_axis1_max:
        pitch_dir = not pitch_dir
    if abs(yaw)>=sphere_axis2_max:
        yaw_dir = not yaw_dir

    time.sleep(0.1)