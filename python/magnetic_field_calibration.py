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
print(dump(balljoint_config, Dumper=Dumper))
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
        pos=(13*math.sin(i*(360/3)/180.0*math.pi)+offset[i][0],13*math.cos(i*(360/3)/180.0*math.pi+offset[i][1]),offset[i][2]),\
        angle=i*(360/3))
        magnet.rotate(angle=angles[i][0],axis=(1,0,0))
        magnet.rotate(angle=angles[i][1],axis=(0,1,0))
        magnet.rotate(angle=angles[i][2],axis=(0,0,1))
        magnets.append(magnet)
    return magnets

def func(x):
    fieldstrength = x[0:3]
    magnet_offsets = [[x[3],x[4],x[5]],[x[6],x[7],x[8]],[x[9],x[10],x[11]]]
    magnet_angles = [[x[12],x[13],x[14]],[x[15],x[16],x[17]],[x[18],x[19],x[20]]]
    sensor_offsets = [[x[21],x[22],x[23]],[x[24],x[25],x[26]],[x[27],x[28],x[29]],[x[30],x[31],x[32]]]
    sensor_angles = [[x[33],x[34],x[35]],[x[36],x[37],x[38]],[x[39],x[40],x[41]],[x[42],x[43],x[44]]]
    b_error = 0
    j =0
    for pos in sensor_log['position']:
        sensors = gen_sensors(sensor_offsets,sensor_angles)
        c = Collection(gen_magnets(x,magnet_offsets,magnet_angles))
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
    return [b_error,b_error,b_error]

res = least_squares(func, \
[1300,1300,1300,\
0,0,0,0,0,0,0,0,0,\
0,0,0,0,0,0,0,0,0,\
0,0,0,0,0,0,0,0,0,0,0,0,\
0,0,0,0,0,0,0,0,0,0,0,0\
],\
bounds = \
((1000,1000,1000,\
-1,-1,-1,-1,-1,-1,-1,-1,-1,\
-5,-5,-5,-5,-5,-5,-5,-5,-5,\
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,\
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1\
), \
(2000,2000,2000,\
1,1,1,1,1,1,1,1,1,\
5,5,5,5,5,5,5,5,5,\
1,1,1,1,1,1,1,1,1,1,1,1,\
1,1,1,1,1,1,1,1,1,1,1,1\
)), ftol=1e-9, xtol=1e-9,verbose=2)

magnet_offsets = [[res.x[3],res.x[4],res.x[5]],[res.x[6],res.x[7],res.x[8]],[res.x[9],res.x[10],res.x[11]]]
magnet_angles = [[res.x[12],res.x[13],res.x[14]],[res.x[15],res.x[16],res.x[17]],[res.x[18],res.x[19],res.x[20]]]
sensor_offsets = [[res.x[21],res.x[22],res.x[23]],[res.x[24],res.x[25],res.x[26]],[res.x[27],res.x[28],res.x[29]],[res.x[30],res.x[31],res.x[32]]]
sensor_angles = [[res.x[33],res.x[34],res.x[35],res.x[36]],[res.x[37],res.x[38],res.x[39],res.x[40]],[res.x[41],res.x[42],res.x[43],res.x[44]]]
print("b_field without calibration:")
c = Collection(gen_magnets(balljoint_config['field_strength'],[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]]))
sensors = gen_sensors([[0,0,0],[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0],[0,0,0]])
for sens in sensors:
    print(sens.getB(c))
print("b_field_error without calibration: %f\n"%func([balljoint_config['field_strength'][0],balljoint_config['field_strength'][1],balljoint_config['field_strength'][2],\
0,0,0,0,0,0,0,0,0,\
0,0,0,0,0,0,0,0,0,\
0,0,0,0,0,0,0,0,0,0,0,0,\
0,0,0,0,0,0,0,0,0,0,0,0,\
])[0])
print("b_field with calibration:")
c = Collection(gen_magnets(res.x[0:3],magnet_offsets,magnet_angles))
sensors = gen_sensors(sensor_offsets,sensor_angles)
i = 0
for sens in sensors:
    print(sens.getB(c))
    i = i+1
print("b_field_error with calibration: %f\n"%func(res.x)[0])
print("\noptimization results: \n\
fieldstrength %.3f %.3f %.3f\n\
magnet offsets:\n \
%.3f %.3f %.3f\n \
%.3f %.3f %.3f\n \
%.3f %.3f %.3f\n \
magnet angles:\n \
%.3f %.3f %.3f\n \
%.3f %.3f %.3f\n \
%.3f %.3f %.3f\n \
sensor offsets:\n \
%.3f %.3f %.3f\n \
%.3f %.3f %.3f\n \
%.3f %.3f %.3f\n \
%.3f %.3f %.3f\n \
sensor angles:\n \
%.3f %.3f %.3f\n \
%.3f %.3f %.3f\n \
%.3f %.3f %.3f\n \
%.3f %.3f %.3f\n \
"%\
(res.x[0],res.x[1],res.x[2],\
res.x[3],res.x[4],res.x[5],res.x[6],res.x[7],res.x[8],res.x[9],res.x[10],res.x[11],\
res.x[12],res.x[13],res.x[14],res.x[15],res.x[16],res.x[17],res.x[18],res.x[19],res.x[20],\
res.x[21],res.x[22],res.x[23],res.x[24],res.x[25],res.x[26],res.x[27],res.x[28],res.x[29],res.x[30],res.x[31],res.x[32],\
res.x[33],res.x[34],res.x[35],res.x[36],res.x[37],res.x[38],res.x[39],res.x[40],res.x[41],res.x[42],res.x[43],res.x[44],\
))
