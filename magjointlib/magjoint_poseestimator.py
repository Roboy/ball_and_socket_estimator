#!/usr/bin/python3
import magjoint
import sys,math,time
import numpy as np
import argparse
from math import *
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

parser = argparse.ArgumentParser()
parser.add_argument("config",help="path to magjoint config file, eg config/single_magnet.yaml")
parser.add_argument("-g",help="generate magnetic field samples",action="store_true")
parser.add_argument("-s",help="steps at which the magnetic field shall be sampled",type=float,default=10.0)
parser.add_argument("model",help="model name to save/load, eg models/two_magnets.npz")
parser.add_argument("-v",help="visualize",action="store_true")
parser.add_argument("-r",help="radius on which to sample the magnetic field in mm",type=float,default=23.5)
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
            sensor = ball.gen_sensors_custom(pos,[[0,0,0]])
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

        float theta = atan2(pos[index].y,pos[index].x)/(2*PI);
        float phi = atan2(sqrtf(powf(pos[index].x,2.0)+powf(pos[index].y,2.0)),pos[index].z)/(2*PI);

        float4 texval = tex2D(tex, theta, phi);
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
        self.texref.set_address_mode(1,drv.address_mode.WRAP)
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
        while not rospy.is_shutdown():
            msg = rospy.wait_for_message("roboy/middleware/MagneticSensor", MagneticSensor)
            self.magneticsCallback(msg)
    def minimizeFunc(self,x):
        for pos,i in zip(self.sensor_pos,range(self.number_of_sensors)):
             r = R.from_euler('xyz', x,degrees=True)
             self.input[i] = r.apply(pos)
        self.interpol(np.int32(self.number_of_sensors),drv.In(self.input),drv.Out(self.output),texrefs=[self.texref],block=self.bdim,grid=self.gdim)
        b_error = 0
        for out,target in zip(self.output,self.b_target):
            b_error += np.linalg.norm(out-target)
        return [b_error]
    def magneticsCallback(self,data):
        if(data.id != self.balljoint.config['id']):
            return

        for i in range(0,4):
            val = np.array((data.x[i], data.y[i], data.z[i]))
            if self.normalize_magnetic_strength:
                val /= np.linalg.norm(val)
            self.b_target[i] = val
        # print(b_target)
        res = least_squares(self.minimizeFunc, self.pos_estimate_prev, bounds = ((-90,-90,-90), (90, 90, 90)),ftol=1e-8, xtol=1e-8,verbose=0,diff_step=0.001)#,max_nfev=20
        b_field_error = res.cost
        rospy.loginfo_throttle(1,"result %.3f %.3f %.3f b-field error %.3f"%(res.x[0],res.x[1],res.x[2],res.cost))
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

PoseEstimator(ball,tex)
