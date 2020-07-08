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
from scipy.optimize import fsolve, least_squares
from scipy.spatial.transform import Rotation as R
import std_msgs.msg, sensor_msgs.msg

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
    pbar = tqdm(total=len(range(-10,3700)))
    for i in range(-10,3700):
        motor_target.publish(i/10)
        rospy.sleep(0.01)
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
        pbar = tqdm(total=len(values[()]['motor_position']))
        colors = [random.sample(range(0, 255),len(args.select)),random.sample(range(0, 255),len(args.select)),random.sample(range(0, 255),len(args.select))]
        number_of_samples = int(len(values[()]['motor_position']))
        number_of_sensors = len(args.select)
        sensor_positions = np.zeros((number_of_samples,number_of_sensors,3))
        sensor_values = np.zeros((number_of_samples, number_of_sensors, 3))

        for i in range(number_of_samples):
            motor_pos = values[()]['motor_position'][i]

            j = 0
            for select in args.select:
                # print(motor_pos)
                quat = Quaternion(axis=[1, 0, 0], degrees=motor_pos)
                sensor_positions[i][j] = np.array(ball.config['sensor_pos'][select])
                sensor_positions[i][j] = quat.rotate(sensor_positions[i][j])
                angle = ball.config['sensor_angle'][select][2]
                sensor_quat = Quaternion(axis=[0, 0, 1], degrees=-angle)
                # sv = np.array([0,0,0])
                sv = values[()]['sensor_values'][i][select]
                sv = sensor_quat.rotate(sv)
                # quat2 = Quaternion(axis=[1, 0, 0], degrees=motor_pos)
                # sv = quat2.rotate(sv)
                if select>=14: # the sensor values on the opposite pcb side need to inverted
                    quat2 = Quaternion(axis=[1, 0, 0], degrees=200)
                    sv = quat2.rotate(sv)
                    # sv = np.array([sv[0],-sv[1],-sv[2]])
                #     quat2 = Quaternion(axis=[0, 1, 0], degrees=motor_pos+180)
                #     sv = quat2.rotate(sv)
                # else:

                sensor_values[i][j] = sv
                j+=1
            pbar.update(1)
        print('saving sensor sensor_positions to '+'models/sensor_positions.npz')
        print('saving sensor sensor_values to '+'models/sensor_values.npz')
        np.savez_compressed('models/sensor_position.npz',values=sensor_positions)
        np.savez_compressed('models/sensor_values.npz',values=sensor_values)
        pbar.close()
        ball.visualizeCloudColor(sensor_values,sensor_positions,args.scale,colors)
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
                phi[j][i] = atan2(sensor_positions[i][j][2], sensor_positions[i][j][1])
                theta[j][i] = atan2(sqrt(sensor_positions[i][j][1]**2 + sensor_positions[i][j][2]**2),sensor_positions[i][j][0])
            indices = sorted(range(number_of_samples), key=lambda k: phi[j][k])
            phi[j] = phi[j][indices]
            theta[j] = theta[j][indices]
            for i in range(0,number_of_samples):
                sensor_values_temp[j][i] = sensor_values[indices[i]][j]
                sensor_positions_temp[j][i] = sensor_positions[indices[i]][j]

        sensor_values = sensor_values_temp
        sensor_positions = sensor_positions_temp

        phi_steps = 360

        phi_indices = [np.zeros(phi_steps,dtype=np.int)]*number_of_sensors
        phi_min = [np.zeros(phi_steps)]*number_of_sensors
        for j in range(0,number_of_sensors):
            for deg in range(0,phi_steps):
                phi_min = abs(phi[j]-[((deg-180.0) / 180.0 * pi)]*number_of_samples)
                index_min = min(range(len(phi_min)), key=phi_min.__getitem__)
                phi_indices[j][deg] = index_min
                # print(index_min)

        # positions = []
        # values = []
        # for j in range(number_of_sensors):
        #     for deg in range(phi_steps):
        #         positions.append(sensor_positions[j][phi_indices[j][deg]])
        #         values.append(sensor_values[j][phi_indices[j][deg]])
        #         color.append([255, 255, 255])
        # ball.visualizeCloudColor2(values, positions, args.scale, color)


        class PoseEstimator:
            balljoint = None
            sensor_values = []
            grid = []
            theta_min = 0
            theta_range = pi
            theta_steps = 0
            phi_steps = 0
            number_of_sensors = 0
            normalize_magnetic_strength = False
            b_target = None
            pos_estimate_prev = [0,0,0]
            body_part = 'head'
            # sensor_select = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
            # sensor_select = [1,14,2,15,3,16,4,17,5,18,6,19,7,20,8,21,9,22,10,23,11,24,12,25,13]
            sensor_select = [0,1,2,3]
            def __init__(self, balljoint, theta_steps, phi_steps, phi_indices, theta_min, theta_range, sensor_values):
                self.balljoint = balljoint
                self.number_of_sensors = balljoint.number_of_sensors
                self.sensor_values = sensor_values
                self.theta_steps = theta_steps
                self.phi_steps = phi_steps
                self.grid = [np.zeros(3)]*theta_steps*phi_steps
                self.theta_min = theta_min
                self.theta_range = theta_range
                self.b_target = [np.zeros(3)]*self.number_of_sensors
                print("phi steps %d, theta steps %d, theta range %f-%f"%(phi_steps,theta_steps,theta_min,theta_min+theta_range))
                for j in range(0,theta_steps):
                  for i in range(0,phi_steps):
                    self.grid[j*phi_steps+i] = sensor_values[j][phi_indices[j][i]]
                self.joint_state = rospy.Publisher('/external_joint_states', sensor_msgs.msg.JointState, queue_size=1)
                print('init done, listening for roboy/middleware/MagneticSensor ... ')
                while not rospy.is_shutdown():
                    msg = rospy.wait_for_message("roboy/middleware/MagneticSensor", MagneticSensor)
                    self.magneticsCallback(msg)
            def interpolate(self,pos):
                phi = atan2(pos[2], pos[1])
                theta = atan2(sqrt(pos[2] ** 2 + pos[1] ** 2),pos[0])
                phi_normalized = (phi + pi) / (pi * 2.0)
                theta_normalized = ((theta - self.theta_min) / self.theta_range)
                gx = theta_normalized * self.theta_steps
                gxi = int(gx)
                tx = gx - gxi
                gy = phi_normalized * self.phi_steps
                gyi = int(gy)
                ty = gy - gyi
                if(gxi>=self.theta_steps-1):
                    gxi = self.theta_steps-2
                if (gyi >= self.phi_steps-1):
                    gyi = self.phi_steps-2
                c000 = self.grid[gxi*self.phi_steps+gyi]
                c100 = self.grid[(gxi+1)*self.phi_steps+gyi]
                c010 = self.grid[gxi*self.phi_steps+gyi+1]
                c110 = self.grid[(gxi+1)*self.phi_steps+gyi+1]
                return (1 - tx) * (1 - ty) * c000 +\
                        tx * (1 - ty) * c100 +\
                        (1 - tx) * ty * c010 +\
                        tx * ty * c110
                return c000
            def minimizeFunc(self, x):
                values = []
                for select in self.sensor_select:
                    pos = np.array(self.balljoint.config['sensor_pos'][select])
                    # pos/=np.linalg.norm(pos)
                    r = R.from_euler('xzy', x, degrees=True)
                    values.append(self.interpolate(r.apply(pos)))
                b_error = 0
                for out, target in zip(values, self.b_target):
                    b_error += np.linalg.norm(out - target)
                return [b_error]
            def magneticsCallback(self, data):
                if (data.id != self.balljoint.config['id']):
                    return

                for select,i in zip(self.sensor_select,range(len(data.x))):
                    angle = self.balljoint.config['sensor_angle'][select][2]
                    sensor_quat = Quaternion(axis=[0, 0, 1], degrees=-angle)
                    val = np.array((data.x[select], data.y[select], data.z[select]))
                    sv = sensor_quat.rotate(val)
                    # if select >= 14:  # the sensor values on the opposite pcb side need to inverted
                    #     quat2 = Quaternion(axis=[1, 0, 0], degrees=200)
                    #     sv = quat2.rotate(sv)
                    self.b_target[i] = sv

                # print(self.b_target)
                res = least_squares(self.minimizeFunc, self.pos_estimate_prev, bounds=((-360, -360, -360), (360, 360, 360)),
                                    ftol=1e-15, gtol=1e-15, xtol=1e-15, verbose=0, diff_step=0.01)  # ,max_nfev=20
                b_field_error = res.cost
                rospy.loginfo_throttle(1, "result %.3f %.3f %.3f b-field error %.3f" % (
                res.x[0], res.x[1], res.x[2], res.cost))
                msg = sensor_msgs.msg.JointState()
                msg.header = std_msgs.msg.Header()
                msg.header.stamp = rospy.Time.now()
                msg.name = [self.body_part + '_axis0', self.body_part + '_axis1', self.body_part + '_axis2']
                msg.velocity = [0, 0, 0]
                msg.effort = [0, 0, 0]
                euler = [res.x[0] / 180 * pi, res.x[1] / 180 * pi, res.x[2] / 180 * pi]
                msg.position = [euler[0], euler[1], euler[2]]
                self.joint_state.publish(msg)
                # if b_field_error < 20000:
                self.pos_estimate_prev = res.x
                # else:
                #     rospy.logwarn_throttle(1, 'b field error too big, resetting joint position...')
                #     self.pos_estimate_prev = [0, 0, 0]

        estimator = PoseEstimator(ball,number_of_sensors,phi_steps,phi_indices,theta[0][0],(theta[-1][0]-theta[0][0]),sensor_values)
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
                color.append([100, 0, 255])
        for i in range(100000):
            pos = np.array([random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1)])
            pos = pos / np.linalg.norm(pos)*22
            value = estimator.interpolate(pos)
            positions.append(pos)
            values.append(np.array([value[0], value[1], value[2]]))
            color.append([80, 30, 255])
        ball.visualizeCloudColor2(values, positions, args.scale, color)
