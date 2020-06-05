#!/usr/bin/python3
import magjoint
import sys,math,time
import numpy as np
from scipy.interpolate import NearestNDInterpolator

if len(sys.argv) < 5:
    print("\nUSAGE: ./magnetic_field_visualization.py ball_joint_config x_step y_step plot_magnet_arrangement scale, e.g. \n python3 magnetic_field_visualization.py two_magnets.yaml 10 10 1 0.1\n")
    sys.exit()

balljoint_config = sys.argv[1]
x_step = int(sys.argv[2])
y_step = int(sys.argv[3])
plot_magnet_arrangement = sys.argv[4]=='1'
scale = float(sys.argv[5])

ball = magjoint.BallJoint(balljoint_config)

magnets = ball.gen_magnets()
if plot_magnet_arrangement:
    ball.plotMagnets(magnets)

grid_positions,positions,pos_offsets,angles,angle_offsets = [],[],[],[],[]
for i in np.arange(-math.pi+math.pi/180*x_step,math.pi-math.pi/180*x_step,math.pi/180*x_step):
    for j in np.arange(-math.pi,math.pi,math.pi/180*y_step):
        grid_positions.append([i,j])
        positions.append([22*math.sin(i)*math.cos(j),22*math.sin(i)*math.sin(j),22*math.cos(i)])
        pos_offsets.append([0,0,0])
        angles.append([0,0,90])
        angle_offsets.append([0,0,0])
number_of_sensors = len(positions)
print('number_of_sensors %d'%number_of_sensors)
print('scale %f'%scale)
start = time.time()
sensors = ball.gen_sensors_all(positions,pos_offsets,angles,angle_offsets)

sensor_values = []
for sens,pos in zip(sensors,positions):
    val = sens.getB(magnets)
    sensor_values.append(val)

my_interpolating_function = NearestNDInterpolator(positions, sensor_values)
print(my_interpolating_function((positions[0][0],positions[0][1],positions[0][2])))
print(sensor_values[0])
print('starting interpolation')
positions = []
sensor_values = []
for i in np.arange(-math.pi+math.pi/180*1,math.pi-math.pi/180*1,math.pi/180*1):
    for j in np.arange(-math.pi,math.pi,math.pi/180*1):
        pos = [22*math.sin(i)*math.cos(j),22*math.sin(i)*math.sin(j),22*math.cos(i)]
        positions.append(pos)
        sensor_values.append(my_interpolating_function((pos[0],pos[1],pos[2])))
print('starting interpolation')
ball.visualizeCloud(sensor_values,positions,scale)
