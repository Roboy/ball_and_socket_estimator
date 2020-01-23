#!/usr/bin/env python
import rospy
import sensor_msgs.msg
import time
import random
rospy.init_node('grid_map')

sphere_axis0_max = 0.5
sphere_axis1_max = 0.5
sphere_axis2_max = 0.6

joint_targets = rospy.Publisher("/joint_targets", sensor_msgs.msg.JointState, queue_size=1)

roll = 0
pitch = 0
yaw = 0

roll_dir = False
pitch_dir = False
yaw_dir = False

joint_targets_msg = sensor_msgs.msg.JointState()
joint_targets_msg.name = ["head_axis0", "head_axis1", "head_axis2"]
joint_targets_msg.velocity = [0,0,0]
joint_targets_msg.effort = [0,0,0]

while not rospy.is_shutdown():
    joint_targets_msg.position = [pitch,roll,yaw]
    joint_targets.publish(joint_targets_msg)

    if pitch_dir:
        pitch = pitch + random.uniform(0.005, 0.01)
    else:
        pitch = pitch - random.uniform(0.005, 0.01)

    if yaw_dir:
        yaw = yaw + random.uniform(0.005, 0.01)
    else:
        yaw = yaw - random.uniform(0.005, 0.01)

    if abs(roll)>=sphere_axis0_max:
        if roll_dir:
            roll = roll - random.uniform(0.015, 0.03)
        else:
            roll = roll + random.uniform(0.015, 0.05)
        roll_dir = not roll_dir
    if abs(pitch)>=sphere_axis1_max:
        if pitch_dir:
            pitch = pitch - random.uniform(0.015, 0.03)
        else:
            pitch = pitch + random.uniform(0.015, 0.03)
        pitch_dir = not pitch_dir
        if roll_dir:
            roll = roll + random.uniform(0.005, 0.01)
        else:
            roll = roll - random.uniform(0.005, 0.01)
    if abs(yaw)>=sphere_axis2_max:
        if yaw_dir:
            yaw = yaw - random.uniform(0.015, 0.03)
        else:
            yaw = yaw + random.uniform(0.015, 0.03)
        yaw_dir = not yaw_dir

    time.sleep(0.2)