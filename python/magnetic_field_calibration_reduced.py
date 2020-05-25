import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from magpylib.source.magnet import Box,Cylinder
from magpylib import Collection, displaySystem, Sensor
from scipy.optimize import fsolve, least_squares
import matplotlib.animation as manimation
import random
import math, os, time, sys
import geometry_msgs
from multiprocessing import Pool, freeze_support
from pathlib import Path
from yaml import load,dump,Loader,Dumper

if len(sys.argv) < 3:
    print("\nUSAGE: python3 magnetic_field_calibration.py balljoint_config_yaml sensor_log_file, e.g. \n python3 magnetic_field_calibration.py test.yaml sensor_log.yaml\n")
    sys.exit()

balljoint_config = load(open(sys.argv[1], 'r'), Loader=Loader)
print(balljoint_config)
sensor_log = load(open(sys.argv[2], 'r'), Loader=Loader)
print(sensor_log['position'])
print(sensor_log['sensor_values'])

def gen_sensors(offset,angles):
    sensor_pos = [[-22.7,7.7,0],[-14.7,-19.4,0],[14.7,-19.4,0],[22.7,7.7,0]]
    sensors = []
    i = 0
    for pos in sensor_pos:
        s = Sensor(pos=(pos[0]+offset[i][0],pos[1]+offset[i][1],pos[2]+offset[i][2]),angle=90,axis=(0,0,1))
        s.rotate(angle=angles[i][0],axis=(1,0,0))
        s.rotate(angle=angles[i][1],axis=(0,1,0))
        s.rotate(angle=angles[i][2],axis=(0,0,1))
        sensors.append(s)
        i = i+1
    return sensors

def gen_magnets(fieldstrength,offset,angles):
    magnets = []
    field = [(0,fieldstrength[0],0),(0,0,fieldstrength[1]),(0,0,fieldstrength[2])]
    for i in range(0,3):
        magnet = Box(mag=field[i],dim=(10,10,10),\
        pos=(13*math.sin(i*(360/3)/180.0*math.pi)+offset[i][0],13*math.cos(i*(360/3)/180.0*math.pi)+offset[i][1],offset[i][2]),\
        angle=i*(360/3))
        magnet.rotate(angle=angles[i][0],axis=(1,0,0))
        magnet.rotate(angle=angles[i][1],axis=(0,1,0))
        magnet.rotate(angle=angles[i][2],axis=(0,0,1))
        magnets.append(magnet)
    return magnets

def func(x):
    fieldstrength = balljoint_config['field_strength']
    magnet_offsets = [[0,0,x[0]],[0,0,x[1]],[0,0,x[2]]]
    magnet_angles = [[x[3],x[4],x[5]],[x[6],x[7],x[8]],[x[9],x[10],x[11]]]
    sensor_offsets = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
    sensor_angles = [[0,0,x[12]],[0,0,x[13]],[0,0,x[14]],[0,0,x[15]]]
    b_error = 0
    j =0
    for pos in sensor_log['position']:
        sensors = gen_sensors(sensor_offsets,sensor_angles)
        c = Collection(gen_magnets(fieldstrength,magnet_offsets,magnet_angles))
        c.rotate(angle=pos[0],axis=(1,0,0), anchor=(0,0,0))
        c.rotate(angle=pos[1],axis=(0,1,0), anchor=(0,0,0))
        c.rotate(angle=pos[2],axis=(0,0,1), anchor=(0,0,0))
        # print("%f %f %f"%(pos[0],pos[1],pos[2]))
        i = 0
        for sens in sensors:
            b_error = b_error + np.linalg.norm(sens.getB(c)-sensor_log['sensor_values'][j][i])
            i=i+1
        j =j +1
    # print(b_error)
    return [b_error]

res = least_squares(func, \
[0,0,0,\
0,0,0,0,0,0,0,0,0,\
0,0,0,0\
],\
bounds = \
((-5,-5,-5,\
-5,-5,-5,-5,-5,-5,-5,-5,-5,\
-5,-5,-5,-5\
), \
(5,5,5,\
5,5,5,5,5,5,5,5,5,\
5,5,5,5\
)), ftol=1e-8, xtol=1e-8,verbose=2,max_nfev=2000)

magnet_offsets = [[0,0,res.x[0]],[0,0,res.x[1]],[0,0,res.x[2]]]
magnet_angles = [[res.x[3],res.x[4],res.x[5]],[res.x[6],res.x[7],res.x[8]],[res.x[9],res.x[10],res.x[11]]]
sensor_offsets = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
sensor_angles = [[0,0,res.x[12]],[0,0,res.x[13]],[0,0,res.x[14]],[0,0,res.x[15]]]
print("b_field without calibration:")
c = Collection(gen_magnets(balljoint_config['field_strength'],[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]]))
sensors = gen_sensors([[0,0,0],[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0],[0,0,0]])
for sens in sensors:
    print(sens.getB(c))
print("b_field_error without calibration: %f\n"%func([\
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\
])[0])
print("b_field with calibration:")
c = Collection(gen_magnets(balljoint_config['field_strength'],magnet_offsets,magnet_angles))
sensors = gen_sensors(sensor_offsets,sensor_angles)
i = 0
for sens in sensors:
    print(sens.getB(c))
    i = i+1
print("b_field_error with calibration: %f\n"%func(res.x)[0])
print("target b_field:")
i = 0
for sens in sensors:
    print(sensor_log['sensor_values'][0][i])
    i = i+1
print("\noptimization results: \n\
fieldstrength %.3f %.3f %.3f\n\
magnet_offsets:\n \
%.3f %.3f %.3f\n \
%.3f %.3f %.3f\n \
%.3f %.3f %.3f\n \
magnet_angles:\n \
%.3f %.3f %.3f\n \
%.3f %.3f %.3f\n \
%.3f %.3f %.3f\n \
sensor_offsets:\n \
%.3f %.3f %.3f\n \
%.3f %.3f %.3f\n \
%.3f %.3f %.3f\n \
%.3f %.3f %.3f\n \
sensor_angles:\n \
%.3f %.3f %.3f\n \
%.3f %.3f %.3f\n \
%.3f %.3f %.3f\n \
%.3f %.3f %.3f\n \
"%\
(balljoint_config['field_strength'][0],balljoint_config['field_strength'][1],balljoint_config['field_strength'][2],\
0,0,res.x[0],0,0,res.x[1],0,0,res.x[2],\
res.x[3],res.x[4],res.x[5],res.x[6],res.x[7],res.x[8],res.x[9],res.x[10],res.x[11],\
0,0,0,0,0,0,0,0,0,0,0,0,\
0,0,res.x[12],\
0,0,res.x[13],\
0,0,res.x[14],\
0,0,res.x[15]\
))

input("Press Enter to write optimization values to ball_config file %s"%sys.argv[1])
balljoint_config['magnet_offsets'][0] = (0,0,res.x[0])
balljoint_config['magnet_offsets'][1] = (0,0,res.x[1])
balljoint_config['magnet_offsets'][2] = (0,0,res.x[2])
balljoint_config['magnet_angles'][0] = (res.x[3],res.x[4],res.x[5])
balljoint_config['magnet_angles'][1] = (res.x[6],res.x[7],res.x[8])
balljoint_config['magnet_angles'][2] = (res.x[9],res.x[10],res.x[11])
balljoint_config['sensor_angles'][0] = (0,0,res.x[12])
balljoint_config['sensor_angles'][1] = (0,0,res.x[13])
balljoint_config['sensor_angles'][2] = (0,0,res.x[14])
balljoint_config['sensor_angles'][3] = (0,0,res.x[15])

print(balljoint_config)
with open(sys.argv[1], 'w') as file:
    documents = dump(balljoint_config, file)
