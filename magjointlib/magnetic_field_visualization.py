#!/usr/bin/python3
import magjoint
import sys,time
import numpy as np
from math import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("config",help="path to magjoint config file, eg config/single_magnet.yaml")
parser.add_argument("-g",help="generate magnetic field samples",action="store_true")
parser.add_argument("-s",help="steps at which the magnetic field shall be sampled",type=float,default=1.0)
parser.add_argument("-scale",help="scale the magnetic field in cloud visualization",type=float,default=1.0)
parser.add_argument("-m",help="model name to load, eg models/two_magnets.npz",default='models/three_magnets.npz')
parser.add_argument("-v",help="visualize",action="store_true")
parser.add_argument("-t",help="sensor arrangement as for the magnetic field calibration",action="store_true")
parser.add_argument("-r",help="radius on which to sample the magnetic field in mm",type=float,default=22)
args = parser.parse_args()
print(args)

ball = magjoint.BallJoint(args.config)

magnets = ball.gen_magnets()
if args.v:
    ball.plotMagnets(magnets)

if args.g:
    positions,pos_offsets,angles,angle_offsets = [],[],[],[]
    if not args.t:
        x_angles = np.arange(0,360,args.s)
        y_angles = np.arange(0,360,args.s)
        for theta,i in zip(x_angles,range(0,len(x_angles))):
            for phi,j in zip(y_angles,range(0,len(y_angles))):
                positions.append([args.r * sin(theta * pi / 180) * cos(phi * pi / 180),
                                 args.r * sin(theta * pi / 180) * sin(phi * pi / 180),
                                  args.r * cos(theta * pi / 180),])
                pos_offsets.append([0,0,0])
                angles.append([0,0,90])
                angle_offsets.append([0,0,0])
    else:
        y_angles = np.arange(0, 360, args.s)
        x_angles = np.arange(0.645774*180/pi, 2.949599*180/pi, 5.5)
        for theta, i in zip(x_angles, range(0, len(x_angles))):
            for phi, j in zip(y_angles, range(0, len(y_angles))):
                positions.append([args.r * sin(theta * pi / 180) * cos(phi * pi / 180),
                                  args.r * sin(theta * pi / 180) * sin(phi * pi / 180),
                                  args.r * cos(theta * pi / 180)])
                pos_offsets.append([0, 0, 0])
                angles.append([0, 0, 90])
                angle_offsets.append([0, 0, 0])
    number_of_sensors = len(positions)
    print('number_of_sensors %d'%number_of_sensors)
    print('scale %f'%args.scale)
    start = time.time()
    sensors = ball.gen_sensors_all(positions,pos_offsets,angles,angle_offsets)
    print('sensors generated')
    print('generating sensor data')
    sensor_values = []
    for sens in sensors:
        val = sens.getB(magnets)
        sensor_values.append(val)
    ball.visualizeCloud(sensor_values,positions,args.scale)
else:
    print('loading model from '+args.m)
    tex = np.load(args.m)['tex']
    print('the loaded texture has the shape:')
    print(tex.shape)
    positions = []
    sensor_values = []
    x_angles = np.arange(0,360,360/tex.shape[0])
    y_angles = np.arange(0,360,360/tex.shape[1])
    for theta,i in zip(x_angles,range(0,len(x_angles))):
        for phi,j in zip(y_angles,range(0,len(y_angles))):
            positions.append([args.r*sin(theta*pi/180)*cos(phi*pi/180),args.r*sin(theta*pi/180)*sin(phi*pi/180),args.r*cos(theta*pi/180)])
            sensor_values.append(tex[i,j,0:3])

    ball.visualizeCloud(sensor_values,positions,args.scale)
