#!/usr/bin/python3
import magjoint
import sys,math,time
import numpy as np
import argparse
from math import sqrt,atan2,pi
import random
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import pycuda.autoinit
from pycuda import gpuarray, tools
import pcl
import pcl.pcl_visualization
from scipy.spatial.transform import Rotation as R
import rospy
from roboy_middleware_msgs.msg import MagneticSensor
import std_msgs.msg, sensor_msgs.msg
from scipy.optimize import fsolve, least_squares
from pyquaternion import Quaternion
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("config",help="path to magjoint config file, eg config/single_magnet.yaml")
parser.add_argument("-v",help="visualize",action="store_true")
args = parser.parse_args()
print(args)

ball = magjoint.BallJoint(args.config)

magnets = ball.gen_magnets()
if args.v: #visualize
    ball.plotMagnets(magnets)

class PoseEstimator:
    balljoint = None
    b_target = None
    normalize_magnetic_strength = True
    pos_estimate_prev = [0,0,0]
    joint_state = None
    selection = [0,1,2,3]
    body_part = 'head'

    def __init__(self, balljoint):
        self.balljoint = balljoint
        self.b_target = [np.zeros(3)]*balljoint.number_of_sensors
        rospy.init_node('BallJointPoseestimator', anonymous=True)
        self.joint_state = rospy.Publisher('/external_joint_states', sensor_msgs.msg.JointState, queue_size=1)

    def minFunc(self,x):
        global b_target
        magnets = self.balljoint.gen_magnets()
        magnets.rotate(x[0],(1,0,0), anchor=(0,0,0))
        magnets.rotate(x[1],(0,1,0), anchor=(0,0,0))
        magnets.rotate(x[2],(0,0,1), anchor=(0,0,0))
        b_error = 0
        i = 0
        for sens in self.balljoint.sensors:
            val = sens.getB(magnets)
            if self.normalize_magnetic_strength:
                val /= np.linalg.norm(val)
            b_error = b_error + np.linalg.norm(val-self.b_target[i])
            i=i+1
        # print(b_error)
        return [b_error]

    def magneticsCallback(self,data):
        if(data.id != self.balljoint.config['id']):
            return
        for select in self.selection:
            val = np.array((data.x[select], data.y[select], data.z[select]))
            angle = self.balljoint.config['sensor_angle'][select][2]
            sensor_quat = Quaternion(axis=[0, 0, 1], degrees=-angle)
            # sensor_quat = Quaternion(axis=[0, 0, 1], degrees=self.angles[select])
            val = sensor_quat.rotate(val)
            if self.normalize_magnetic_strength:
                val /= np.linalg.norm(val)
            self.b_target[select] = val
        # print(b_target)
        res = least_squares(self.minFunc, self.pos_estimate_prev, bounds = ((-50,-50,-80), (50, 50, 80)),ftol=1e-3, xtol=1e-3)#,max_nfev=20
        b_field_error = res.cost
        rospy.loginfo_throttle(1,"result %.3f %.3f %.3f b-field error %.3f"%(res.x[0],res.x[1],res.x[2],res.cost))
        # print("result %.3f %.3f %.3f b-field error %.3f\nb_target %.3f %.3f %.3f\t%.3f %.3f %.3f\t%.3f %.3f %.3f\t%.3f %.3f %.3f"%(res.x[0],res.x[1],res.x[2],res.cost,b_target[0][0],b_target[0][1],b_target[0][2],b_target[1][0],b_target[1][1],b_target[1][2],b_target[2][0],b_target[2][1],b_target[2][2],b_target[3][0],b_target[3][1],b_target[3][2]))
        msg = sensor_msgs.msg.JointState()
        msg.header = std_msgs.msg.Header()
        msg.header.stamp = rospy.Time.now()
        msg.name = [self.body_part+'_axis0', self.body_part+'_axis1', self.body_part+'_axis2']
        msg.velocity = [0,0,0]
        msg.effort = [0,0,0]
        euler = [res.x[0]/180*math.pi,res.x[1]/180*math.pi,res.x[2]/180*math.pi]
        msg.position = [euler[0], euler[1], euler[2]]
        self.joint_state.publish(msg)
        self.pos_estimate_prev = res.x

estimator = PoseEstimator(ball)

while not rospy.is_shutdown():
    msg = rospy.wait_for_message("roboy/middleware/MagneticSensor", MagneticSensor)
    estimator.magneticsCallback(msg)
