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

x_angles = np.float32(np.arange(0,1,0.01))
y_angles = np.float32(np.arange(0,1,0.01))
number_of_samples = len(x_angles)
print(number_of_samples)
# tex = np.float32(np.random.rand(number_of_samples, number_of_samples,4))
# tex = np.float32(np.arange(number_of_samples*number_of_samples*4).reshape(number_of_samples,number_of_samples,4))
tex = np.zeros((number_of_samples,number_of_samples,4),dtype=np.float32)
for i in range(number_of_samples):
    for j in range(number_of_samples):
        tex[i,j,0] = sin(x_angles[i]*pi)*cos(y_angles[j]*pi)
        tex[i,j,1] = sin(x_angles[i]*pi)*sin(y_angles[j]*pi)
        tex[i,j,2] = cos(x_angles[i]*pi)

# print(tex)

interpol = mod.get_function("MagneticFieldInterpolateKernel")
texref = mod.get_texref('tex')

drv.bind_array_to_texref(
                drv.make_multichannel_2d_array(tex, order="C"),
                texref
                )
texref.set_flags(drv.TRSF_NORMALIZED_COORDINATES)
texref.set_filter_mode(drv.filter_mode.LINEAR)

# number_of_queries = 100
# x_angle_queries = np.random.rand(number_of_queries)
# y_angle_queries = np.random.rand(number_of_queries)
# x_angle_queries = x_angles
# y_angle_queries = y_angles+0.1
# x_angle_queries = np.float32(np.arange(0,1,1/number_of_queries))
# y_angle_queries = np.float32(np.arange(0,1,1/number_of_queries))
x_angle_queries = np.zeros(number_of_samples*number_of_samples,dtype=np.float32)
y_angle_queries = np.zeros(number_of_samples*number_of_samples,dtype=np.float32)
k = 0
for i in range(number_of_samples):
    for j in range(number_of_samples):
        x_angle_queries[k] = (i/number_of_samples)+np.random.rand()*0.1
        y_angle_queries[k] = (j/number_of_samples)+np.random.rand()*0.1
        k+=1
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
for o in out:
    print(o)
#     print(tex[i,i,:])
#     i+=1
print(out.shape)

cloud = pcl.PointCloud_PointXYZRGB()
points = np.zeros((number_of_queries+number_of_samples*number_of_samples, 4), dtype=np.float32)
k = 0
for o in out:
    points[k][0] = o[0]
    points[k][1] = o[1]
    points[k][2] = o[2]
    points[k][3] = 255 << 16 | 255 << 8 | 255
    k = k+1
for i in range(number_of_samples):
    for j in range(number_of_samples):
        points[k][0] = tex[i,j,0]
        points[k][1] = tex[i,j,1]
        points[k][2] = tex[i,j,2]
        points[k][3] = 0 << 16 | 255 << 8 | 255
        k = k+1


cloud.from_array(points)

visual = pcl.pcl_visualization.CloudViewing()
visual.ShowColorCloud(cloud)

v = True
while v:
    v = not(visual.WasStopped())
