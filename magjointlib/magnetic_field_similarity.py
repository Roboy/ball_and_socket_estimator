#!/usr/bin/python3
import magjoint
import sys, time, math
import numpy as np
from multiprocessing import Pool, freeze_support, get_context, set_start_method
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray, tools

if len(sys.argv) < 2:
    print("\nUSAGE: ./magnetic_collision_cuda.py ball_joint_config visualize_only, e.g. \n python3 magnetic_collision_cuda.py two_magnets.yaml 1\n")
    sys.exit()

balljoint_config = sys.argv[1]
visualize_only = sys.argv[2]=='1'

ball = magjoint.BallJoint(balljoint_config)

magnets = ball.gen_magnets(ball.config)
if visualize_only:
    ball.plotMagnets(magnets)
    sys.exit()

positions,pos_offsets,angles,angle_offsets = [],[],[],[]
for i in np.arange(-math.pi,math.pi,math.pi/180*5):
    for j in np.arange(-math.pi,math.pi,math.pi/180*5):
        positions.append([25*math.sin(i)*math.cos(j),25*math.sin(i)*math.sin(j),25*math.cos(i)])
        pos_offsets.append([0,0,0])
        angles.append([0,0,90])
        angle_offsets.append([0,0,0])
number_of_sensors = len(positions)
print('number_of_sensors %d'%number_of_sensors)
start = time.time()
sensors = ball.gen_sensors_all(positions,pos_offsets,angles,angle_offsets)
end = time.time()
print('took: %d s or %f min'%(end - start,(end - start)/60))
start = time.time()
sensor_values = np.zeros((number_of_sensors,3),dtype=np.float32,order='C')
for sens,i in zip(sensors,range(0,number_of_sensors)):
    sensor_values[i]=sens.getB(magnets)
end = time.time()
print('took: %d s or %f min'%(end - start,(end - start)/60))
start = time.time()

# start = cuda.Event()
# end   = cuda.Event()

mod = SourceModule("""
  __global__ void distance(int number_of_samples, float3 *p1, float *d)
  {
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;
    if(i>=number_of_samples || j>=number_of_samples || j<i)
        return;
    d[i*number_of_samples+j] = sqrtf(powf(p1[i].x-p1[j].x,2.0) + powf(p1[i].y-p1[j].y,2.0) + powf(p1[i].z-p1[j].z,2.0));
  };
  """)

distance = mod.get_function("distance")

number_of_samples = len(sensor_values)
p1_gpu = gpuarray.to_gpu(sensor_values)
comparisons = int(((number_of_samples-1)*(number_of_samples/2)))
out_gpu = gpuarray.empty(number_of_samples**2, np.float32)
print(out_gpu)
print('calculating %d collisions'%comparisons)
number_of_samples = np.int32(number_of_samples)

bdim = (16, 16, 1)
dx, mx = divmod(number_of_samples, bdim[0])
dy, my = divmod(number_of_samples, bdim[1])
gdim = ( int((dx + (mx>0))), int((dy + (my>0))))
print(bdim)
print(gdim)
distance(number_of_samples, p1_gpu, out_gpu, block=bdim, grid=gdim)
end = time.time()
print('took: %d s or %f min'%(end - start,(end - start)/60))
out = np.reshape(out_gpu.get(),(number_of_samples,number_of_samples))
sum = 0
for val in out_gpu.get():
    sum += val
print(out)
print(sum)
print(gpuarray.sum(out_gpu))
