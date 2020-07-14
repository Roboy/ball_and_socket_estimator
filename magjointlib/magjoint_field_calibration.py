#!/usr/bin/python3
import magjoint
import sys,time
import numpy as np
import math
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
from bisect import bisect_left
import pycuda.driver as drv
from pycuda.compiler import SourceModule

parser = argparse.ArgumentParser()
parser.add_argument("config",help="path to magjoint config file, eg config/single_magnet.yaml")
parser.add_argument("-g",help="generate magnetic field samples",action="store_true")
parser.add_argument("-s",help="steps at which the magnetic field shall be sampled",type=float,default=1.0)
parser.add_argument("-scale",help="scale the magnetic field in cloud visualization",type=float,default=0.05)
parser.add_argument("-model",help="model name to save, eg models/three_magnets.npz",default='models/three_magnets_v0.2.npz')
parser.add_argument("-tex",help="use texture",action="store_true")
parser.add_argument("-v",help="visualize only",action="store_true")
parser.add_argument("-p",help="predict",action="store_true")
parser.add_argument("-select", nargs='+', help="select which sensors", type=int,
                        default=[0,1,14,2,15,3,16,4,17,5,18,6,19,7,20,8,21,9,22,10,23,11,24,12,25,13])
args = parser.parse_args()

ball = magjoint.BallJoint(args.config)
print(args)

magnets = ball.gen_magnets()
if args.v:
    ball.plotMagnets(magnets)
    sys.exit()

sensor_positions = np.load('models/sensor_position.npz')['values']
sensor_values = np.load('models/sensor_values.npz')['values']

sensor_positions_selection,sensor_values_selection,phi_selection,theta_selection = ball.filterRecordedData(args.select,360,sensor_positions,sensor_values)

number_of_sensors = len(args.select)
number_of_samples = len(sensor_positions_selection[0])

if args.tex:
    print('saving texture '+args.model)
    tex = np.zeros((number_of_sensors,number_of_samples,4),dtype=np.float32)
    for i in range(number_of_sensors):
        for j in range(number_of_samples):
            tex[i, j, 0] = sensor_values_selection[i][j][0]
            tex[i, j, 1] = sensor_values_selection[i][j][1]
            tex[i, j, 2] = sensor_values_selection[i][j][2]

    np.savez_compressed(args.model,tex=tex)

# color = []
# positions = []
# values = []
# for j in range(number_of_sensors):
#     for (pos,val) in zip(sensor_positions_selection[j],sensor_values_selection[j]):
#         positions.append(pos)
#         values.append(val)
#         color.append([255, 255, 255])
# ball.visualizeCloudColor2(values, positions, args.scale, color)


class PoseEstimator:
    balljoint = None
    sensor_positions = None
    sensor_values = None
    theta = []
    phi = None
    phi_steps = 0
    theta_steps = 0
    number_of_sensors = 0
    normalize_magnetic_strength = False
    b_target = None
    pos_estimate_prev = [0,0,0]
    body_part = 'head'
    # sensor_select = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
    # sensor_select = [1,14,2,15,3,16,4,17,5,18,6,19,7,20,8,21,9,22,10,23,11,24,12,25,13]
    sensor_select = [4]
    def __init__(self, balljoint,phi,theta,sensor_positions,sensor_values):
        self.balljoint = balljoint
        self.number_of_sensors = balljoint.number_of_sensors
        self.sensor_positions = sensor_positions
        self.sensor_values = sensor_values
        self.phi = phi
        self.phi_steps = len(phi[0])
        self.theta_steps = len(phi)
        for j in range(len(theta_selection)):
            self.theta.append(theta[j][0])
        self.b_target = [np.zeros(3)]*self.number_of_sensors
        rospy.init_node('magnetic_field_calibration', anonymous=True)
        self.joint_state = rospy.Publisher('/external_joint_states', sensor_msgs.msg.JointState, queue_size=1)
        # print('init done, listening for roboy/middleware/MagneticSensor ... ')
        # while not rospy.is_shutdown():
        #     msg = rospy.wait_for_message("roboy/middleware/MagneticSensor", MagneticSensor)
        #     self.magneticsCallback(msg)

    def take_closest(self, myList, myNumber):
        """
        Assumes myList is sorted. Returns closest value to myNumber.

        If two numbers are equally close, return the smallest number.
        """
        pos = bisect_left(myList, myNumber)
        if pos == 0:
            return myList[0],pos
        if pos == len(myList):
            return myList[-1],pos-1
        return myList[pos], pos
        # before = myList[pos - 1]
        # after = myList[pos]
        # if after - myNumber < myNumber - before:
        #     return after,pos
        # else:
        #     return before,pos
    def interpolate(self,pos):
        phi = math.atan2(pos[0], pos[1])
        theta = math.pi-math.atan2(math.sqrt(pos[0] ** 2 + pos[1] ** 2),pos[2])
        theta_selected,theta_index = self.take_closest(self.theta,theta)
        phi_selected,phi_index = self.take_closest(self.phi[theta_index],phi)
        # tx = theta - theta_selected
        # ty = phi - phi_selected
        #
        # if phi_index<self.phi_steps-1 and theta_index<self.theta_steps-1 and phi_index>0 and theta_index>0:
        #     c00 = self.sensor_values[theta_index][phi_index]
        #     if tx>0:
        #         c10 = self.sensor_values[theta_index + 1][phi_index]
        #     else:
        #         c10 = self.sensor_values[theta_index - 1][phi_index]
        #     if ty>0:
        #         c01 = self.sensor_values[theta_index][phi_index - 1]
        #     else:
        #         c01 = self.sensor_values[theta_index][phi_index - 1]
        #     if tx>0 and ty>0:
        #         c11 = self.sensor_values[theta_index + 1][phi_index+1]
        #     elif tx>0 and ty<0:
        #         c11 = self.sensor_values[theta_index + 1][phi_index - 1]
        #     elif tx<0 and ty>0:
        #         c11 = self.sensor_values[theta_index - 1][phi_index + 1]
        #     else:
        #         c11 = self.sensor_values[theta_index - 1][phi_index - 1]
        #     return (1 - tx) * (1 - ty) * c00 + \
        #             tx * (1 - ty) * c10 +\
        #             (1 - tx) * ty * c01 +\
        #             tx * ty * c11
        # else:
        return self.sensor_values[theta_index][phi_index]

        # c000 = self.sensor_values[theta_index][phi_index]
        # return (1 - tx) * (1 - ty) * c000 +\
        #         tx * (1 - ty) * c100 +\
        #         (1 - tx) * ty * c010 +\
        #         tx * ty * c110
        # return c000
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
                            ftol=1e-15, gtol=1e-15, xtol=1e-15, verbose=0, diff_step=0.1)  # ,max_nfev=20
        b_field_error = res.cost
        rospy.loginfo_throttle(1, "result %.3f %.3f %.3f b-field error %.3f" % (
        res.x[0], res.x[1], res.x[2], res.cost))
        msg = sensor_msgs.msg.JointState()
        msg.header = std_msgs.msg.Header()
        msg.header.stamp = rospy.Time.now()
        msg.name = [self.body_part + '_axis0', self.body_part + '_axis1', self.body_part + '_axis2']
        msg.velocity = [0, 0, 0]
        msg.effort = [0, 0, 0]
        euler = [res.x[0] / 180 * math.pi, res.x[1] / 180 * math.pi, res.x[2] / 180 * math.pi]
        msg.position = [euler[0], euler[1], euler[2]]
        self.joint_state.publish(msg)
        # if b_field_error < 20000:
        self.pos_estimate_prev = res.x
        # else:
        #     rospy.logwarn_throttle(1, 'b field error too big, resetting joint position...')
        #     self.pos_estimate_prev = [0, 0, 0]

estimator = PoseEstimator(ball,phi_selection,theta_selection,sensor_positions_selection,sensor_values_selection)
positions = []
values = []
color = []
for j in range(number_of_sensors):
    for i in range(number_of_samples):

        # value_interpolated = estimator.interpolate(sensor_positions_selection[j][i])
        # positions.append(sensor_positions_selection[j][i])
        # values.append(np.array([value_interpolated[0],value_interpolated[1],value_interpolated[2]]))
        # color.append([255,255,255])
        positions.append(sensor_positions_selection[j][i])
        value_measured = sensor_values_selection[j][i]
        values.append(value_measured)
        color.append([255, 255, 0])

        # phi = (i-180) * math.pi / 180
        # theta = (j * 11) * math.pi / 180
        # positions.append([
        #     22 * math.cos(theta),
        #     22 * math.sin(theta) * math.cos(phi),
        #     22 * math.sin(theta) * math.sin(phi)])
        # value_texture = tex[j][i][0:3]
        # values.append(np.array(value_texture))
        # # values.append(np.zeros(3))
        # color.append([255, 255, 255])
# for i in range(100000):
#     pos = np.array([random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1)])
#     pos = pos / np.linalg.norm(pos)*22
#     value = estimator.interpolate(pos)
#     positions.append(pos)
#     values.append(np.array([value[0], value[1], value[2]]))
#     color.append([80, 90, 0])
ball.visualizeCloudColor2(values, positions, args.scale, color)
