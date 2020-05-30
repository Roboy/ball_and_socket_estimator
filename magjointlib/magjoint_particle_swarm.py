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
import sys
import magjoint
import copy

class ParticleSwarm():
    n_particles = 0
    ball_joint = None
    particle = {'magnet_angles':[],'personal_best_score':10000,'personal_best':[],'vel':[],'vel_towards_global_best':[],\
                'vel_towards_personal_best':[], 'color':0}
    particles = []
    global_best_score = 0
    global_best_particle = 0

    global_attraction = 0.01
    personal_attraction = 0.1
    random_speed = 5

    joint_positions = []
    iteration = 0

    def __init__(self, n_particles, joint_positions, ball_joint):
        self.n_particles = n_particles
        self.ball_joint = ball_joint
        self.joint_positions = joint_positions

        self.global_best_score = 10000
        self.global_best_particle = 0

        for i in range(0,n_particles):
            p = copy.deepcopy(self.particle)
            for j in range(0,ball_joint.number_of_magnets):
                p['magnet_angles'].append(np.array([random.uniform(-180,180),random.uniform(-180,180),random.uniform(-180,180)]))
                p['vel'].append(np.array([random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1)]))
                p['vel_towards_global_best'].append(np.array([random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1)]))
                p['vel_towards_personal_best'].append(np.array([random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1)]))
                p['color'] = float(random.randint(50, 255)<<16|random.randint(50, 255)<<8|random.randint(50, 255)<<0)
            p['personal_best'] = p['magnet_angles']
            self.particles.append(p)

    def fitness(self):
        # calculate new particle scores
        i = 0
        for p in self.particles:
            sensor_values = self.ball_joint.generateSensorData(self.joint_positions,p['magnet_angles'])
            colliders,magnetic_field_differences = self.ball_joint.calculateCollisions(sensor_values,self.joint_positions,30*1.44)
            score = len(colliders)
            if score < p['personal_best_score']:
                print('score of particle %d improved from %d to %d'%(i,p['personal_best_score'],score))
                p['personal_best_score'] = score
                p['personal_best'] = p['magnet_angles']
            i+=1
        # calculate global best score
        i = 0
        for p in self.particles:
            if p['personal_best_score']<self.global_best_score:
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
        self.fitness()
        iteration += 1
