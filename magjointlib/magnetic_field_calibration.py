#!/usr/bin/python3
import magjoint
import sys,time
import numpy as np
from math import pi,atan2,sqrt,fabs
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
parser.add_argument("-p",help="predict",action="store_true")
parser.add_argument("-select", nargs='+', help="select which sensors", type=int,
                        default=[1,14,2,15,3,16,4,17,5,18,6,19,7,20,8,21,9,22,10,23,11,24,12,25,13])
args = parser.parse_args()

ball = magjoint.BallJoint(args.config)
print(args)

magnets = ball.gen_magnets()
if args.v:
    ball.plotMagnets(magnets)
    sys.exit()

rospy.init_node('magnetic_field_calibration',anonymous=True)


motor_target = rospy.Publisher('motor_target', Float32, queue_size=1)

if args.g: #record data
    motor_target.publish(0)
    rospy.sleep(1)

    values = {'motor_position':[], 'sensor_values':[]}
    pbar = tqdm(total=360)
    for i in range(0,360):
        motor_target.publish(i)
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
else: # load recorded data
    if not args.p:
        print('loading model from '+args.m)
        values = np.load(args.m)['values']
        # print(values[()]['motor_position'])
        sensor_values = []
        sensor_positions = []
        pbar = tqdm(total=360)
        colors = [random.sample(range(0, 255),len(args.select)),random.sample(range(0, 255),len(args.select)),random.sample(range(0, 255),len(args.select))]
        color = []
        # sensor_select = [0,1,2,3,4,14,15,16,17,18]
        for i in range(0,360):
            motor_pos = values[()]['motor_position'][i]
            quat = Quaternion(axis=[0, 1, 0], degrees=motor_pos)
            j = 0
            sensor_values_ = []
            positions = []
            for select in args.select:
                pos = ball.config['sensor_pos'][select]
                sensor_pos_new = quat.rotate(pos)
                positions.append(np.array(sensor_pos_new))
                angle = ball.config['sensor_angle'][select][2]
                sensor_quat = Quaternion(axis=[0, 0, 1], degrees=-angle)
                sv = sensor_quat.rotate(values[()]['sensor_values'][i][select])
                if select>=14: # the sensor values on the opposite pcb side need to inverted
                    sv = np.array([sv[0],-sv[1],-sv[2]])
                sensor_values_.append(sv)
                color.append([colors[0][j],colors[1][j],colors[2][j]])
                j+=1
            sensor_positions.append(positions)
            sensor_values.append(sensor_values_)
            pbar.update(1)
        print('saving sensor sensor_positions to '+'models/sensor_positions.npz')
        print('saving sensor sensor_values to '+'models/sensor_values.npz')
        np.savez_compressed('models/sensor_position.npz',values=sensor_positions)
        np.savez_compressed('models/sensor_values.npz',values=sensor_values)
        pbar.close()
        ball.visualizeCloudColor(sensor_values,sensor_positions,args.scale,color)
    else:
        sensor_positions = np.load('models/sensor_position.npz')['values']
        sensor_values = np.load('models/sensor_values.npz')['values']
        number_of_sensors = len(args.select)
        number_of_samples = len(sensor_positions)
        # calculate spherical coordinates
        phi = [np.zeros(number_of_samples)]*number_of_sensors
        theta = [np.zeros(number_of_samples)]*number_of_sensors

        sensor_values_temp = np.zeros((number_of_sensors,number_of_samples,3))
        sensor_positions_temp = np.zeros((number_of_sensors,number_of_samples,3))
        colors = [random.sample(range(0, 255), len(args.select)), random.sample(range(0, 255), len(args.select)),
                  random.sample(range(0, 255), len(args.select))]
        color = []

        for j in range(0,number_of_sensors):
            for i in range(0,number_of_samples):
                phi[j][i] = atan2(sensor_positions[i][j][2], sensor_positions[i][j][0])
                theta[j][i] = atan2(sqrt(sensor_positions[i][j][0]**2 + sensor_positions[i][j][2]**2),sensor_positions[i][j][1])
            indices = sorted(range(number_of_samples), key=lambda k: phi[j][k])
            phi[j] = phi[j][indices]
            theta[j] = theta[j][indices]
            for i in range(0,number_of_samples):
                sensor_values_temp[j][i] = sensor_values[indices[i]][j]
                sensor_positions_temp[j][i] = sensor_positions[indices[i]][j]

        sensor_values = sensor_values_temp
        sensor_positions = sensor_positions_temp
        # positions = []
        # values = []
        # for (pos,value) in zip(sensor_positions,sensor_values):
        #     for (p,v) in zip(pos,value):
        #         positions.append(p)
        #         values.append(v)
        #         color.append([255,255,255])
        # ball.visualizeCloudColor2(values, positions, args.scale, color)

        phi_steps = 360

        phi_indices = [np.zeros(phi_steps,dtype=np.int)]*number_of_sensors
        phi_min = [np.zeros(phi_steps)]*number_of_sensors
        for j in range(0,number_of_sensors):
            for deg in range(0,phi_steps):
                phi_min = abs(phi[j]-[((deg-180.0) / 180.0 * pi)]*number_of_samples)
                index_min = min(range(len(phi_min)), key=phi_min.__getitem__)
                phi_indices[j][deg] = index_min
                # print(index_min)

        class PoseEstimator:
            sensor_values = []
            grid = []
            theta_min = 0
            theta_range = pi
            theta_steps = 0
            phi_steps = 0
            def __init__(self, theta_steps, phi_steps, phi_indices, theta_min, theta_range, sensor_values):
                self.sensor_values = sensor_values
                self.theta_steps = theta_steps
                self.phi_steps = phi_steps
                self.grid = [np.zeros(3)]*theta_steps*phi_steps
                self.theta_min = theta_min
                self.theta_range = theta_range
                for j in range(0,theta_steps):
                  for i in range(0,phi_steps):
                    self.grid[j*phi_steps+i] = sensor_values[j][phi_indices[j][i]]
            def interpolate(self,pos):
                phi = atan2(pos[2], pos[0])
                theta = atan2(sqrt(pos[0] ** 2 + pos[2] ** 2),pos[1])
                phi_normalized = (phi + pi) / (pi * 2.0)
                theta_normalized = 1-((theta - self.theta_min) / self.theta_range) #because of the order of theta from high to low
                gx = theta_normalized * self.theta_steps
                gxi = int(gx)
                tx = gx - gxi
                gy = phi_normalized * self.phi_steps
                gyi = int(gy)
                ty = gy - gyi
                if(gxi>=self.theta_steps):
                    gxi = self.theta_steps-1
                if (gyi >= self.phi_steps):
                    gyi = self.phi_steps - 1
                c000 = self.grid[gxi*self.phi_steps+gyi]
                return c000

        estimator = PoseEstimator(number_of_sensors,phi_steps,phi_indices,theta[-1][0],(theta[0][0]-theta[-1][0]),sensor_values)
        positions = []
        values = []
        color = []
        for j in range(number_of_sensors):
            for i in range(number_of_samples):

                # pos = np.array([random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1)])
                # pos = pos/np.linalg.norm(pos)
                value = estimator.interpolate(sensor_positions[j][i])
                positions.append(sensor_positions[j][i])
                values.append(np.array([value[0],value[1],value[2]]))
                color.append([255,255,255])
                positions.append(sensor_positions[j][i])
                values.append(sensor_values[j][i])
                color.append([0, 0, 255])
        ball.visualizeCloudColor2(values, positions, args.scale, color)
