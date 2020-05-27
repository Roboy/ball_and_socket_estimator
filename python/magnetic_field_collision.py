import rospy
from roboy_middleware_msgs.msg import MagneticSensor
import numpy as np
import matplotlib.pyplot as plt
from magpylib.source.magnet import Box,Cylinder
from magpylib import Collection, displaySystem, Sensor
from scipy.optimize import fsolve, least_squares
import matplotlib.animation as manimation
import random, math
import sensor_msgs.msg, std_msgs
from yaml import load,dump,Loader,Dumper
import sys
from os import path
import pandas
from pandas import DataFrame
from pandas import concat
from tqdm import tqdm

if len(sys.argv) < 2:
    print("\nUSAGE: python3 magnetic_field_collision.py body_part, e.g. \n python3 magnetic_field_collision.py head \n")
    sys.exit()

body_part = sys.argv[1]

dataset = pandas.read_csv('/home/letrend/workspace/roboy3/'+body_part+'_data0.log', delim_whitespace=True)
dataset = dataset.values[0:len(dataset),0:]

sensor_values = []
sensor_values.append(dataset[:,0:3])
sensor_values.append(dataset[:,3:6])
sensor_values.append(dataset[:,6:9])
sensor_values.append(dataset[:,9:12])
pos = dataset[:,12:]
print('first sensor values')
print(sensor_values[0][0])
print(sensor_values[1][0])
print(sensor_values[2][0])
print(sensor_values[3][0])
print('first ball position')
print(pos[0])

number_of_samples = len(pos)
print('number of samples %d'%number_of_samples)
position_difference_sensitivity = 5/180*math.pi
magnetic_field_difference_sensitivity = 10

comparisons = 0
magnetic_field_difference = []
position_difference = []

for i in range(0,number_of_samples):
    for j in range(i+1,number_of_samples):
        mag_diff = 0
        for k in range(0,4):
            mag_diff += np.linalg.norm(sensor_values[k][i]-sensor_values[k][j])
        magnetic_field_difference.append(mag_diff)
        pos_diff = np.linalg.norm(pos[i]-pos[j])
        position_difference.append(pos_diff)
        if pos_diff>position_difference_sensitivity and mag_diff<magnetic_field_difference_sensitivity:
            for k in range(0,4):
                print(k)
                print(sensor_values[k][i])
                print(sensor_values[k][j])
            print('positions in degree')
            print(pos[i]*180/math.pi)
            print(pos[j]*180/math.pi)
            print('mag_diff')
            print(mag_diff)
            print('pos_diff in degree')
            print(pos_diff*180/math.pi)
            print('oh oh')

        comparisons += 1


print('comparisons: %d'%comparisons)
# print(magnetic_field_difference)
# print(position_difference)