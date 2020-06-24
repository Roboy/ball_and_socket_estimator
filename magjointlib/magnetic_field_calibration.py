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
from tqdm import tqdm
import random

parser = argparse.ArgumentParser()
parser.add_argument("config",help="path to magjoint config file, eg config/single_magnet.yaml")
parser.add_argument("-g",help="generate magnetic field samples",action="store_true")
parser.add_argument("-s",help="steps at which the magnetic field shall be sampled",type=float,default=1.0)
parser.add_argument("-scale",help="scale the magnetic field in cloud visualization",type=float,default=1.0)
parser.add_argument("-m",help="model name to load, eg data/three_magnets.npz",default='data/three_magnets.npz')
parser.add_argument("-v",help="visualize only",action="store_true")
parser.add_argument("-select", nargs='+', help="select which sensors", type=int,
                        default=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25])
args = parser.parse_args()

ball = magjoint.BallJoint(args.config)
print(args)

magnets = ball.gen_magnets()
if args.v:
    ball.plotMagnets(magnets)
    sys.exit()

rospy.init_node('magnetic_field_calibration',anonymous=True)


motor_target = rospy.Publisher('motor_target', Float32, queue_size=1)

if args.g:
    motor_target.publish(0)
    rospy.sleep(1)

    values = {'motor_position':[], 'sensor_values':[]}
    pbar = tqdm(total=3600)
    for i in range(0,3600):
        motor_target.publish(i/10)
        sensor = rospy.wait_for_message("/roboy/middleware/MagneticSensor", MagneticSensor, timeout=None)
        motor_position = rospy.wait_for_message("/motor_position", Float32, timeout=None)
        values['motor_position'].append(motor_position.data)
        v = []
        for x,y,z in zip(sensor.x,sensor.y,sensor.z):
            v.append(np.array([x,y,z]))
        values['sensor_values'].append(v)
        pbar.update(1)
    pbar.close()
    motor_target.publish(0)

    print('saving data to '+args.m)
    np.savez_compressed(args.m,values=values)
else:
    print('loading model from '+args.m)
    values = np.load(args.m)['values']
    # print(values[()]['motor_position'])
    sensor_values = []
    positions = []
    pbar = tqdm(total=3600)
    colors = [random.sample(range(0, 255),len(args.select)),random.sample(range(0, 255),len(args.select)),random.sample(range(0, 255),len(args.select))]
    color = []
    # sensor_select = [0,1,2,3,4,14,15,16,17,18]
    for i in range(0,3600):
        motor_pos = values[()]['motor_position'][i]
        quat = Quaternion(axis=[0, 1, 0], degrees=motor_pos)
        j = 0
        for select in args.select:
            pos = ball.config['sensor_pos'][select]
            sensor_pos_new = quat.rotate(pos)
            positions.append(np.array(sensor_pos_new))
            angle = ball.config['sensor_angle'][select][2]
            sensor_quat = Quaternion(axis=[0, 0, 1], degrees=-angle)
            sv = sensor_quat.rotate(values[()]['sensor_values'][i][select])
            if j<14:
                # sensor_quat = Quaternion(axis=[1,0, 0], degrees=90)
                # sv = sensor_quat.rotate(sv)
                sv = np.array([sv[0],-sv[1],-sv[2]])
            sensor_values.append(sv)
            color.append([colors[0][j],colors[1][j],colors[2][j]])
            j+=1
        pbar.update(1)
    pbar.close()
    ball.visualizeCloudColor(sensor_values,positions,args.scale,color)
