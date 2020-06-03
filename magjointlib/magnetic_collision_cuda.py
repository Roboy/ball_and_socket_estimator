#!/usr/bin/python3
import magjoint
import sys, time
import numpy as np

if len(sys.argv) < 5:
    print("\nUSAGE: ./magnetic_collision_cuda.py ball_joint_config x_step y_step z_step visualize_only, e.g. \n python3 magnetic_collision_cuda.py two_magnets.yaml 10 10 10 1\n")
    sys.exit()

balljoint_config = sys.argv[1]
x_step = int(sys.argv[2])
y_step = int(sys.argv[3])
z_step = int(sys.argv[4])
visualize_only = sys.argv[5]=='1'

ball = magjoint.BallJoint(balljoint_config)

magnets = ball.gen_magnets()
if visualize_only:
    ball.plotMagnets(magnets)
    sys.exit()

grid_positions = []
for i in np.arange(-80,80,x_step):
    for j in np.arange(-80,80,y_step):
        for k in np.arange(-80,80,z_step):
            grid_positions.append([i,j,k])

sensor_values,pos = ball.generateMagneticDataGrid(grid_positions)
start = time.time()
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray, tools

mod = SourceModule("""
  __global__ void distance(long number_of_samples, float3 *p1, float3 *p2, float3 *p3, float3 *p4, float *d)
  {
    // for upper triangle indices check this reference: https://math.stackexchange.com/questions/646117/how-to-find-a-function-mapping-matrix-indices
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;
    if(i>=number_of_samples || j>=number_of_samples || j<=i)
        return;
    d[(2*i*number_of_samples-i*i+2*j-3*i-2)/2] = sqrtf(powf(p1[i].x-p1[j].x,2.0) + powf(p1[i].y-p1[j].y,2.0) + powf(p1[i].z-p1[j].z,2.0)) +
                               sqrtf(powf(p2[i].x-p2[j].x,2.0) + powf(p2[i].y-p2[j].y,2.0) + powf(p2[i].z-p2[j].z,2.0)) +
                               sqrtf(powf(p3[i].x-p3[j].x,2.0) + powf(p3[i].y-p3[j].y,2.0) + powf(p3[i].z-p3[j].z,2.0)) +
                               sqrtf(powf(p4[i].x-p4[j].x,2.0) + powf(p4[i].y-p4[j].y,2.0) + powf(p4[i].z-p4[j].z,2.0));
  };
  """)

distance = mod.get_function("distance")

number_of_samples = len(sensor_values)
p1 = np.zeros((number_of_samples,3),dtype=np.float32,order='C')
p2 = np.zeros((number_of_samples,3),dtype=np.float32,order='C')
p3 = np.zeros((number_of_samples,3),dtype=np.float32,order='C')
p4 = np.zeros((number_of_samples,3),dtype=np.float32,order='C')
i = 0
for val in sensor_values:
    p1[i] = val[0]
    p2[i] = val[1]
    p3[i] = val[2]
    p4[i] = val[3]
    i = i+1
# print(p1)
p1_gpu = gpuarray.to_gpu(p1)
p2_gpu = gpuarray.to_gpu(p2)
p3_gpu = gpuarray.to_gpu(p3)
p4_gpu = gpuarray.to_gpu(p4)
comparisons = np.int32((number_of_samples-1)*(number_of_samples/2))
if comparisons<0:
    print('ohoh too many values')
    sys.exit()
else:
    print('calculating %d comparisons'%comparisons)

out_gpu = gpuarray.empty(comparisons, np.float32)
number_of_samples = np.int32(number_of_samples)

bdim = (16, 16, 1)
dx, mx = divmod(number_of_samples, bdim[0])
dy, my = divmod(number_of_samples, bdim[1])
gdim = ( int((dx + (mx>0))), int((dy + (my>0))))
print(bdim)
print(gdim)
distance(number_of_samples, p1_gpu, p2_gpu, p3_gpu, p4_gpu, out_gpu, block=bdim, grid=gdim)
end = time.time()
print('took: %d s or %f min'%(end - start,(end - start)/60))
out = out_gpu.get()#np.reshape(out_gpu.get(),(number_of_samples,number_of_samples))
collisions = 0
mag_diffs = []
colliders = []
a = (out<1.44*x_step);
collisions = len(out[a])
print(out[a])
print('number of collisions %d'%collisions)
# for val,i in zip(out,range(0,number_of_samples)):
#     for v,j in zip(val,range(0,number_of_samples)):
#         if j>i:
#             if v<1.44*x_step:
#                 colliders.append([i,j])
#                 collisions+=1

# print('there are %d collisions'%(collisions))
