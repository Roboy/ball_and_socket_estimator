import random
import numpy as np
import pcl
import pcl.pcl_visualization
import scipy
from scipy import spatial
import time
import yaml
import rospy
import std_msgs
import sensor_msgs.msg
import roboy_simulation_msgs.msg
import sys, math, os
import magjoint
import copy
from tqdm import tqdm
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray, tools
from magpylib import Collection, displaySystem, Sensor
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import names

class ParticleSwarm():
    n_particles = 0
    ball_joint = None
    particle = {'magnet_angles':[],'personal_best_score':10000,'personal_best':[],'vel':[],'vel_towards_global_best':[],\
                'vel_towards_personal_best':[], 'color':0, 'name':None}
    particles = []
    global_best_score = 0
    global_best_particle = 0

    global_attraction = 0.001
    personal_attraction = 0.1
    random_speed = 5

    number_of_sensors = 0
    sensors = []
    distance = None
    iteration = 0
    status_bar = None
    magnetic_field_difference_sensitivity = 10
    target_folder = None
    normalize_magnetic_field = True

    def __init__(self, n_particles, sensors, ball_joint):
        self.n_particles = n_particles
        self.ball_joint = ball_joint
        self.number_of_sensors = len(sensors)
        self.sensors = sensors

        self.global_best_score = 10000
        self.global_best_particle = 0

        self.target_folder = time.strftime("%Y%m%d-%H%M%S")
        os.mkdir('pics/'+self.target_folder)

        for i in range(0,n_particles):
            p = copy.deepcopy(self.particle)
            for j in range(0,ball_joint.number_of_magnets):
                p['magnet_angles'].append(np.array([random.uniform(-180,180),random.uniform(-180,180),random.uniform(-180,180)]))
                p['vel'].append(np.array([random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1)]))
                p['vel_towards_global_best'].append(np.array([random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1)]))
                p['vel_towards_personal_best'].append(np.array([random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1)]))
                p['color'] = float(random.randint(50, 255)<<16|random.randint(50, 255)<<8|random.randint(50, 255)<<0)
                p['name'] = names.get_full_name(gender='female')
            os.mkdir('pics/'+self.target_folder+'/'+p['name'])
            p['personal_best'] = p['magnet_angles']
            self.particles.append(p)

        print('initializing cuda kernel')
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

        self.distance = mod.get_function("distance")
        plt.ioff()
    def fitness(self):
        # calculate new particle scores
        i = 0
        for p in self.particles:
            sensor_values = np.zeros((self.number_of_sensors,3),dtype=np.float32,order='C')
            magnets = self.ball_joint.gen_magnets_angle(p['magnet_angles'])
            for sens,i in zip(self.sensors,range(0,self.number_of_sensors)):
                value = sens.getB(magnets)
                if self.normalize_magnetic_field:
                    sensor_values[i]=value/np.linalg.norm(value)
                else:
                    sensor_values[i]=value
            p1_gpu = gpuarray.to_gpu(sensor_values)
            out_gpu = gpuarray.empty(self.number_of_sensors**2, np.float32)
            number_of_samples = np.int32(self.number_of_sensors)
            bdim = (16, 16, 1)
            dx, mx = divmod(number_of_samples, bdim[0])
            dy, my = divmod(number_of_samples, bdim[1])
            gdim = ( int((dx + (mx>0))), int((dy + (my>0))))
            # print(bdim)
            # print(gdim)
            self.distance(number_of_samples, p1_gpu, out_gpu, block=bdim, grid=gdim)
            out = np.reshape(out_gpu.get(),(number_of_samples,number_of_samples))
            # sum = 0
            # for val in out_gpu.get():
            #     sum += val
            # print(out)
            # print(sum)
            # print(gpuarray.sum(out_gpu))
            score = gpuarray.sum(out_gpu).get()
            if score > p['personal_best_score']:
                # print('score of particle %d improved from %d to %d'%(i,p['personal_best_score'],score))
                p['personal_best_score'] = score
                p['personal_best'] = p['magnet_angles']
            i+=1

            fig = plt.figure(figsize=(9,9))
            ax1 = fig.add_subplot(111, projection='3d')
            displaySystem(magnets, subplotAx=ax1, suppress=True, direc=True)
            fig.savefig('pics/'+self.target_folder+'/'+p['name']+'/'+str(self.iteration)+'.png')
            plt.close(fig)
            self.status_bar.update(1)
        # calculate global best score
        i = 0
        for p in self.particles:
            if p['personal_best_score']>self.global_best_score:
                self.global_best_score = p['personal_best_score']
                self.global_best_particle = i
                print('new global best score %d of particle %d'%(self.global_best_score,i))
            i+=1
    def move(self):
        i = 0
        for p in self.particles:
            for j in range(0,self.ball_joint.number_of_magnets):
                p['vel_towards_global_best'][j] = (self.particles[self.global_best_particle]['magnet_angles'][j] - p['magnet_angles'][j])*self.global_attraction
                p['vel_towards_personal_best'][j] = (p['personal_best'][j] - p['magnet_angles'][j])*self.personal_attraction
                random_movement = np.array([random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1)])*self.random_speed
                p['magnet_angles'][j] += p['vel_towards_global_best'][j]+p['vel_towards_personal_best'][j]+random_movement

    def step(self):
        self.move()
        self.status_bar = tqdm(total=self.n_particles, desc='fitness_status', position=0)
        self.fitness()
        self.iteration += 1
