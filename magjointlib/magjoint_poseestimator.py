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
parser.add_argument("-g",help="generate magnetic field samples",action="store_true")
parser.add_argument("-s",help="steps at which the magnetic field shall be sampled",type=float,default=1.0)
parser.add_argument("model",help="model name to save/load, eg models/two_magnets.npz")
parser.add_argument("-v",help="visualize",action="store_true")
parser.add_argument("-scale",help="scale the magnetic field in cloud visualization",type=float,default=0.05)
parser.add_argument("-r",help="radius on which to sample the magnetic field in mm",type=float,default=22)
args = parser.parse_args()
print(args)

ball = magjoint.BallJoint(args.config)

magnets = ball.gen_magnets()
if args.v: #visualize
    ball.plotMagnets(magnets)

if args.g: #generate magnetic field
    # we sample the magnetic field on a sphere around the ball joint center
    x_angles = np.arange(0,360,args.s)
    y_angles = np.arange(0,360,args.s)
    width,height = len(x_angles),len(y_angles)
    texture_shape = (width,height)
    # this texture is filled with the magnetic field data and saved as the model.
    print('the texture has the shape:')
    print(texture_shape)
    tex = np.zeros((width,height,4),dtype=np.float32)
    print('generating magnetic field data')
    start = time.time()
    k = 0
    for theta,i in zip(x_angles,range(0,width)):
        for phi,j in zip(y_angles,range(0,height)):
            pos = [[args.r*sin(theta*pi/180)*cos(phi*pi/180),args.r*sin(theta*pi/180)*sin(phi*pi/180),args.r*cos(theta*pi/180)]]
            sensor = ball.gen_sensors_custom(pos,[[0,0,90]])
            val = sensor[0].getB(magnets)
            tex[i,j,0] = val[0]
            tex[i,j,1] = val[1]
            tex[i,j,2] = val[2]
            k+=1
    print('done generating magnetic data')
    end = time.time()
    print('took: %f s or %f min'%(end - start,(end - start)/60))
    print('saving model to '+args.model)
    np.savez_compressed(args.model,tex=tex)
else: # loading texture
    print('loading model from '+args.model)
    tex = np.load(args.model)['tex']
    print('the loaded texture has the shape:')
    print(tex.shape)
    # print(tex)

class PoseEstimator:
    balljoint = None
    tex = None
    texref = None
    mod = SourceModule("""
    texture<float4, 2> tex;
    #define PI 3.141592654f
    __global__ void MagneticFieldInterpolateKernel(
        int32_t number_of_samples,
        float3 *pos,
        float3 *data
        )
    {
        unsigned int index = blockIdx.x * blockDim.x + threadIdx.x + blockIdx.y * blockDim.y + threadIdx.y;

        if( index > number_of_samples )
            return;

        float phi = atan2(pos[index].x,pos[index].y)*180.0f/PI;
        float theta = 180.0f-atan2(sqrtf(powf(pos[index].x,2.0)+powf(pos[index].y,2.0)),pos[index].z)*180.0f/PI;
        
        float phi_normalized = (phi+180.0f)/360.0f;
        float theta_normalized = (theta)/143.0f;

        float4 texval = tex2D(tex, phi_normalized,theta_normalized);
        data[index].x = texval.x;
        data[index].y = texval.y;
        data[index].z = texval.z;
    }
    """)
    interpol = None
    number_of_sensors = 0
    sensor_pos = []
    output = None
    input = None
    bdim = None
    gdim = None
    joint_state = None
    b_target = None
    normalize_magnetic_strength = False
    pos_estimate_prev = [0,0,0]
    body_part = 'head'
    # selection = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
    # selection = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,14,15,16,17,18,19,20,21,22,23,24,25]
    selection = [0, 1, 2, 3]
    def __init__(self,balljoint,texture):
        self.balljoint = balljoint
        self.tex = texture
        self.interpol = self.mod.get_function("MagneticFieldInterpolateKernel")
        self.texref = self.mod.get_texref('tex')
        drv.bind_array_to_texref(
                        drv.make_multichannel_2d_array(self.tex, order="C"),
                        self.texref
                        )
        self.texref.set_flags(drv.TRSF_NORMALIZED_COORDINATES)
        self.texref.set_filter_mode(drv.filter_mode.LINEAR)
        self.texref.set_address_mode(0,drv.address_mode.WRAP)
        self.texref.set_address_mode(1,drv.address_mode.CLAMP)
        self.sensor_pos = balljoint.config['sensor_pos']
        self.number_of_sensors = len(self.sensor_pos)
        self.input = np.zeros((self.number_of_sensors,3), dtype=np.float32,order='C')
        self.output = np.zeros((self.number_of_sensors,3), dtype=np.float32,order='C')
        self.b_target = np.zeros((self.number_of_sensors,3), dtype=np.float32,order='C')

        self.bdim = (16, 16, 1)
        dx, mx = divmod(self.number_of_sensors, self.bdim[0])
        dy, my = divmod(self.number_of_sensors, self.bdim[1])
        self.gdim = ( int((dx + (mx>0))), int((dy + (my>0))))
        rospy.init_node('BallJointPoseestimator',anonymous=True)
        self.joint_state = rospy.Publisher('/external_joint_states', sensor_msgs.msg.JointState , queue_size=1)
    def minimizeFunc(self,x):
        r = R.from_euler('xyz', x, degrees=True)
        for (select,i) in zip(self.selection,range(len(self.selection))):
             pos = self.sensor_pos[select]
             self.input[i] = r.apply(pos)
        self.interpol(np.int32(len(self.selection)),drv.In(self.input),drv.Out(self.output),texrefs=[self.texref],block=self.bdim,grid=self.gdim)
        b_error = 0
        for i in range(len(self.selection)):
            out = self.output[i]
            target = r.apply(self.b_target[i])
            b_error += np.linalg.norm(out-target)
        return [b_error]
    def interpolate(self,x):
        r = R.from_euler('xyz', x, degrees=True)
        for pos,i in zip(self.sensor_pos,range(self.number_of_sensors)):
             self.input[i] = r.apply(pos)
        self.interpol(np.int32(self.number_of_sensors), drv.In(self.input), drv.Out(self.output), texrefs=[self.texref],
                      block=self.bdim, grid=self.gdim)
        output_rot = np.zeros((self.number_of_sensors,3))
        for i in range(self.number_of_sensors):
            output_rot[i] = r.inv().apply(self.output[i])
        return self.input, self.output,output_rot
    def interpolatePosition(self,pos):
        self.input[0] = pos
        self.interpol(np.int32(1), drv.In(self.input), drv.Out(self.output), texrefs=[self.texref],
                      block=self.bdim, grid=self.gdim)
        return self.input, self.output
    def magneticsCallback(self,data):
        if(data.id != self.balljoint.config['id']):
            return

        for select in self.selection:
            val = np.array((data.x[select], data.y[select], data.z[select]))
            angle = self.balljoint.config['sensor_angle'][select][2]
            sensor_quat = Quaternion(axis=[0, 0, 1], degrees=-angle)
            # sensor_quat = Quaternion(axis=[0, 0, 1], degrees=self.angles[select])
            val = sensor_quat.rotate(val)
            if select >= 14:  # the sensor values on the opposite pcb side need to inverted
                quat2 = Quaternion(axis=[1, 0, 0], degrees=12)
                val = quat2.rotate(val)
            self.b_target[select] = val
        # print(b_target)
        res = least_squares(self.minimizeFunc, self.pos_estimate_prev, bounds = ((-360,-360,-360), (360, 360, 360)),
                            ftol=1e-15, xtol=1e-15, gtol=1e-15, verbose=0,diff_step=0.01)#,max_nfev=20
        b_field_error = res.cost
        rospy.loginfo_throttle(1,"result %.3f %.3f %.3f b-field error %.3f"%(res.x[0],res.x[1],res.x[2],res.cost))
        # print(self.b_target)
        # print(self.output)
        msg = sensor_msgs.msg.JointState()
        msg.header = std_msgs.msg.Header()
        msg.header.stamp = rospy.Time.now()
        msg.name = [self.body_part+'_axis0', self.body_part+'_axis1', self.body_part+'_axis2']
        msg.velocity = [0,0,0]
        msg.effort = [0,0,0]
        euler = [res.x[0]/180*math.pi,res.x[1]/180*math.pi,res.x[2]/180*math.pi]
        msg.position = [euler[0], euler[1], euler[2]]
        self.joint_state.publish(msg)
        # if b_field_error<2000:
        self.pos_estimate_prev = res.x
        # else:
        #     rospy.logwarn_throttle(1,'b field error too big, resetting joint position...')
        #     self.pos_estimate_prev = [0,0,0]

estimator = PoseEstimator(ball,tex)

# generate_training_data = True
# if generate_training_data:
#     body_part = 'head'
#     record = open("/home/letrend/workspace/roboy3/"+body_part+"_data0.log","w")
#     record.write("mx0 my0 mz0 mx1 my1 mz1 mx2 my2 mz3 mx3 my3 mz3 roll pitch yaw\n")
#
# positions = []
# values = []
# color = []
#
# number_of_samples = 10000
# pbar = tqdm(total=number_of_samples)
# selection = [0,1,2,3]
# for i in range(number_of_samples):
#     pose = np.array([0,0, random.uniform(-180, 180)])
#     # pose = np.array([random.uniform(-5,5),random.uniform(-5,5),random.uniform(-180,180)])
#     # pose = np.array([random.uniform(-90,90),random.uniform(-90,90),random.uniform(-180,180)])
#     pos,value,value_rot = estimator.interpolate(pose)
#     for i in range(ball.number_of_sensors):
#         positions.append(np.array([pos[i][0],pos[i][1],pos[i][2]]))
#         values.append(np.array([value[i][0],value[i][1],value[i][2]]))
#         color.append([80, 90, 0])
#     if generate_training_data:
#         record.write( \
#         str(value_rot[selection[0]][0]) + " " + str(value_rot[selection[0]][1]) + " " + str(value_rot[selection[0]][2]) + " " + \
#         str(value_rot[selection[1]][0]) + " " + str(value_rot[selection[1]][1]) + " " + str(value_rot[selection[1]][2]) + " " + \
#         str(value_rot[selection[2]][0]) + " " + str(value_rot[selection[2]][1]) + " " + str(value_rot[selection[2]][2]) + " " + \
#         str(value_rot[selection[3]][0]) + " " + str(value_rot[selection[3]][1]) + " " + str(value_rot[selection[3]][2]) + " " + \
#         str(pose[0] / 180.0 * math.pi) + " " + str(pose[1] / 180.0 * math.pi) + " " + str(
#             pose[2] / 180.0 * math.pi) + "\n")
#     pbar.update(1)
# if generate_training_data:
#     record.close()
#     print('data saved to /home/letrend/workspace/roboy3/' + body_part + '_data0.log')
#
# for j in range(tex.shape[0]):
#     for i in range(tex.shape[1]):
#         phi = (i - 180)
#         theta = 180-(j * 5.5)
#         # theta_normalized = (theta) / 143.0
#         # phi_normalized = (phi + 180) / 360.0
#         pos = [22 * math.sin(theta * math.pi / 180) * math.sin(phi * math.pi / 180),
#                22 * math.sin(theta * math.pi / 180) * math.cos(phi * math.pi / 180),
#                22 * math.cos(theta * math.pi / 180)]
#         positions.append(pos)
#         values.append(np.array(tex[j][i][0:3]))
#         color.append([255,255,255])
# #
#         p,value = estimator.interpolatePosition(pos)
#         positions.append(pos)
#         values.append(np.array([value[0][0],value[0][1],value[0][2]]))
#         color.append([0, 255, 255])
#
# ball.visualizeCloudColor2(values, positions, args.scale, color)

while not rospy.is_shutdown():
    msg = rospy.wait_for_message("roboy/middleware/MagneticSensor", MagneticSensor)
    estimator.magneticsCallback(msg)
