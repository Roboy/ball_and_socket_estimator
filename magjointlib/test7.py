#!/usr/bin/python3
from __future__ import division
import numpy as np
import pycuda.driver as drv
from pycuda.compiler import SourceModule

import pycuda.autoinit
import numpy.testing
from pycuda import gpuarray, tools
from math import *

import pcl
import pcl.pcl_visualization

from scipy.spatial.transform import Rotation as R

import magjoint, sys

if len(sys.argv) < 5:
    print("\nUSAGE: ./magnetic_collision_cuda.py ball_joint_config x_step y_step visualize_only, e.g. \n python3 magnetic_collision_cuda.py two_magnets.yaml 10 10 1\n")
    sys.exit()

balljoint_config = sys.argv[1]
x_step = int(sys.argv[2])
y_step = int(sys.argv[3])
visualize_only = sys.argv[4]=='1'

ball = magjoint.BallJoint(balljoint_config)

magnets = ball.gen_magnets()
if visualize_only:
    ball.plotMagnets(magnets)
    sys.exit()

mod = SourceModule("""
texture<float4, 2> tex;

__global__ void MagneticFieldInterpolateKernel(
    int32_t number_of_samples,
    float *x_angle,
    float *y_angle,
    float3* data
    )
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int index = x+y;

    if( index > number_of_samples )
        return;

    float4 texval = tex2D(tex, float(x_angle[index]), float(y_angle[index]));
    data[index].x = texval.x;
    data[index].y = texval.y;
    data[index].z = texval.z;
}
""")

x_step = 1
y_step = 1

x_angles = np.arange(0,180,x_step)
y_angles = np.arange(0,360,y_step)

width,height = len(x_angles),len(y_angles)
texture_shape = (width,height)
tex = np.zeros((width,height,4),dtype=np.float32)

x_angle_queries = np.zeros(width*height,dtype=np.float32)
y_angle_queries = np.zeros(width*height,dtype=np.float32)

r = R.from_euler('zyx', [0,0,0], degrees=True)
for pos in ball.config['sensor_pos']:
    v = r.apply(pos)
    phi = atan2(v[1],v[0])
    theta = atan2(sqrt(v[0]**2+v[1]**2),v[2])
    print(r.apply(pos))
    print((phi,theta))

radius = 22

k = 0
for theta,i in zip(x_angles,range(0,width)):
    for phi,j in zip(y_angles,range(0,height)):
        pos = [[radius*sin(theta*pi/180)*cos(phi*pi/180),radius*sin(theta*pi/180)*sin(phi*pi/180),radius*cos(theta*pi/180)]]
        # val = pos[0]
        sensor = ball.gen_sensors_custom(pos,[[0,0,0]])
        val = sensor[0].getB(magnets)
        tex[i,j,0] = val[0]
        tex[i,j,1] = val[1]
        tex[i,j,2] = val[2]
        # print(val)
        x_angle_queries[k] = theta/180.0
        y_angle_queries[k] = phi/180.0
        k+=1

print(texture_shape)

interpol = mod.get_function("MagneticFieldInterpolateKernel")
texref = mod.get_texref('tex')

drv.bind_array_to_texref(
                drv.make_multichannel_2d_array(tex, order="C"),
                texref
                )
texref.set_flags(drv.TRSF_NORMALIZED_COORDINATES)
texref.set_filter_mode(drv.filter_mode.LINEAR)
texref.set_address_mode(0,drv.address_mode.WRAP)
texref.set_address_mode(1,drv.address_mode.WRAP)

# number_of_queries = 100
# x_angle_queries = np.random.rand(number_of_queries)
# y_angle_queries = np.random.rand(number_of_queries)
# x_angle_queries = x_angles
# y_angle_queries = y_angles
# x_angle_queries = np.float32(np.arange(0,1,1/number_of_queries))
# y_angle_queries = np.float32(np.arange(0,1,1/number_of_queries))
# x_angle_queries = np.zeros(number_of_samples*number_of_samples,dtype=np.float32)
# y_angle_queries = np.zeros(number_of_samples*number_of_samples,dtype=np.float32)
# k = 0
# for i in range(number_of_samples):
#     for j in range(number_of_samples):
#         x_angle_queries[k] = (i/number_of_samples)+np.random.rand()*0.1
#         y_angle_queries[k] = (j/number_of_samples)+np.random.rand()*0.1
#         k+=1
number_of_queries = len(x_angle_queries)
# x_angle_queries, y_angle_queries = np.meshgrid(x_angles, y_angles, sparse=True)
x_angles_gpu = gpuarray.to_gpu(x_angle_queries)
y_angles_gpu = gpuarray.to_gpu(y_angle_queries)
print((x_angles_gpu,y_angles_gpu))

output = np.zeros(number_of_queries*3, dtype=np.float32,order='C')

bdim = (16, 16, 1)
dx, mx = divmod(number_of_queries, bdim[0])
dy, my = divmod(number_of_queries, bdim[1])
gdim = ( int((dx + (mx>0))), int((dy + (my>0))))

interpol(np.int32(number_of_queries),x_angles_gpu,y_angles_gpu,drv.Out(output),texrefs=[texref],block=bdim,grid=gdim)
out = output.reshape(number_of_queries,3)
# i = 0
# for o in out:
#     print(o)
#     print(tex[i,i,:])
#     i+=1
print(out.shape)

# cloud = pcl.PointCloud_PointXYZRGB()
# points = np.zeros((number_of_queries+width*height, 4), dtype=np.float32)
# k = 0
# for o in out:
#     points[k][0] = o[0]
#     points[k][1] = o[1]
#     points[k][2] = o[2]
#     points[k][3] = 255 << 16 | 255 << 8 | 255
#     k = k+1
# for i in range(width):
#     for j in range(height):
#         points[k][0] = tex[i,j,0]
#         points[k][1] = tex[i,j,1]
#         points[k][2] = tex[i,j,2]
#         points[k][3] = 0 << 16 | 255 << 8 | 255
#         k = k+1
#
#
# cloud.from_array(points)
#
# visual = pcl.pcl_visualization.CloudViewing()
# visual.ShowColorCloud(cloud)
#
# v = True
# while v:
#     v = not(visual.WasStopped())

x_angle_queries = np.zeros(10,dtype=np.float32)
y_angle_queries = np.zeros(10,dtype=np.float32)
truth = []
for i in range(10):
    theta = np.random.uniform(0,pi)
    phi = np.random.uniform(0,pi)
    print((theta,phi))
    x_angle_queries[i]=(theta/180)
    y_angle_queries[i]=(phi/180)
    pos = [[radius*sin(theta*pi/180)*cos(phi*pi/180),radius*sin(theta*pi/180)*sin(phi*pi/180),radius*cos(theta*pi/180)]]
    # val = pos[0]
    sensor = ball.gen_sensors_custom(pos,[[0,0,0]])
    val = sensor[0].getB(magnets)
    truth.append(val)

number_of_queries = len(x_angle_queries)
# x_angle_queries, y_angle_queries = np.meshgrid(x_angles, y_angles, sparse=True)
x_angles_gpu = gpuarray.to_gpu(x_angle_queries)
y_angles_gpu = gpuarray.to_gpu(y_angle_queries)
print((x_angles_gpu,y_angles_gpu))

output = np.zeros(number_of_queries*3, dtype=np.float32,order='C')

bdim = (16, 16, 1)
dx, mx = divmod(number_of_queries, bdim[0])
dy, my = divmod(number_of_queries, bdim[1])
gdim = ( int((dx + (mx>0))), int((dy + (my>0))))

interpol(np.int32(number_of_queries),x_angles_gpu,y_angles_gpu,drv.Out(output),texrefs=[texref],block=bdim,grid=gdim)
out = output.reshape(number_of_queries,3)
for v,t in zip(out,truth):
    print("-------")
    print(v)
    print(t)
