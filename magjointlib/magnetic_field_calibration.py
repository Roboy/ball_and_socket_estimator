#!/usr/bin/python3
import magjoint
import sys,time
import numpy as np
from math import *
import argparse
from pyquaternion import Quaternion
import rospy
from std_msgs.msg import Float32
from roboy_middleware_msgs.msg import MagneticSensor

parser = argparse.ArgumentParser()
parser.add_argument("config",help="path to magjoint config file, eg config/single_magnet.yaml")
parser.add_argument("-g",help="generate magnetic field samples",action="store_true")
parser.add_argument("-s",help="steps at which the magnetic field shall be sampled",type=float,default=1.0)
parser.add_argument("-scale",help="scale the magnetic field in cloud visualization",type=float,default=1.0)
parser.add_argument("-m",help="model name to load, eg models/two_magnets.npz",default='models/three_magnets.npz')
parser.add_argument("-v",help="visualize only",action="store_true")
parser.add_argument("-r",help="radius on which to sample the magnetic field in mm",type=float,default=23.5)
args = parser.parse_args()
print(args)

ball = magjoint.BallJoint(args.config)

magnets = ball.gen_magnets()
if args.v:
    ball.plotMagnets(magnets)
    sys.exit()

rospy.init_node('magnetic_field_calibration',anonymous=True)

motor_target = rospy.Publisher('motor_target', Float32, queue_size=1)

motor_target.publish(0)
rospy.sleep(1)

sensor_values = []
positions = []
for i in range(0,2200):
    motor_target.publish(i/10)
    # rospy.sleep(0.02)
    sensor = rospy.wait_for_message("/roboy/middleware/MagneticSensor", MagneticSensor, timeout=None)
    motor_position = rospy.wait_for_message("/motor_position", Float32, timeout=None)
    quat = Quaternion(axis=[0, 1, 0], angle=-motor_position.data/180.0*pi)
    for s,a,x,y,z in zip(ball.config['sensor_pos'],ball.config['sensor_angle'],sensor.x,sensor.y,sensor.z):
        v = quat.rotate(s)
        positions.append(np.array(v))
        sensor_quat = Quaternion(axis=[0, 0, 1], angle=-a[2]/180.0*pi)
        sensor_values.append(quat.rotate(sensor_quat.rotate(np.array([x,y,z]))))
motor_target.publish(0)
ball.visualizeCloud(sensor_values,positions,args.scale)
