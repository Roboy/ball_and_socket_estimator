import random
import numpy as np
import pcl
import pcl.pcl_visualization
import scipy
from scipy import spatial
import time

class Space():
    def __init__(self, n_particles):
        self.n_particles = n_particles
        self.particles = np.zeros((n_particles,3),dtype=np.float32)
        self.particles_visited = np.zeros((n_particles,4),dtype=np.float32)
        self.best_particle = 0
        self.gbest_value = float('inf')
        self.gbest_position = np.array([0,0,0])
        self.pbest_value = np.ones(n_particles)*float('inf')
        self.pbest_position = np.zeros((n_particles,3))
        self.vel = np.random.rand(n_particles, 3)*0.01
        self.vel_towards_global_best = np.random.rand(n_particles, 3)*0.01
        self.vel_towards_personal_best = np.random.rand(n_particles, 3)*0.01
        self.particles_tree = scipy.spatial.KDTree(self.particles)
        self.visited = np.array([0,0,0], dtype=np.float32)
        self.neighbors = np.ones(n_particles)*float('inf')
        self.colors = np.zeros(n_particles,dtype=np.float32)
        for i in range(self.colors.size):
            self.colors[i] = float(random.randint(50, 255)<<16|random.randint(50, 255)<<8|random.randint(50, 255)<<0)
    def fitness(self):
        if self.visited.shape[0]>3:
            neighbors = self.particles_tree.query_ball_tree(self.visited_tree,0.01)
            for i in range(len(neighbors)):
                self.neighbors[i] = len(neighbors[i])

    def move(self):
        for i in range(self.particles.shape[0]):
            self.vel_towards_global_best[i] = (self.gbest_position - self.particles[i])*0.01
            self.vel_towards_personal_best[i] = (self.pbest_position[i] - self.particles[i])*0.01
            self.vel[i] = self.vel_towards_global_best[i] + self.vel_towards_personal_best[i] + (np.random.rand(1, 3)-0.5)*0.01
            self.particles[i] = self.particles[i]+self.vel[i]
            self.particles_visited = np.vstack([self.particles_visited,np.concatenate((self.particles[i],self.colors[i]),axis=None)])
            self.visited = np.vstack([self.visited,self.particles[i]])
        self.particles_tree = scipy.spatial.KDTree(self.particles)
        self.visited_tree = scipy.spatial.KDTree(self.visited)

    def set_pbest(self):
        for particle in range(self.particles.shape[0]):
            if self.neighbors[particle] < self.pbest_value[particle]:
                self.pbest_value[particle] = self.neighbors[particle]
                self.pbest_position[particle] = self.particles[particle]

    def set_gbest(self):
        new_global_best = False
        for particle in range(self.particles.shape[0]):
            if self.pbest_value[particle] < self.gbest_value:
                self.gbest_value = self.pbest_value[particle]
                self.gbest_position = self.pbest_position[particle]
                new_global_best = True
                self.best_particle = particle
        if not new_global_best:
            random_particle = random.randint(0, self.n_particles-1)
            self.gbest_value = self.pbest_value[random_particle]
            self.gbest_position = self.pbest_position[random_particle]
            print("choosing random particle %d with %d neighbors at (%f %f %f)"%(random_particle,self.gbest_value,
                                                                          self.gbest_position[0],self.gbest_position[1],
                                                                          self.gbest_position[2]))
        else:
            print("choosing particle %d with %d neighbors at (%f %f %f)"%(self.best_particle, self.gbest_value,
                                                                          self.gbest_position[0],self.gbest_position[1],
                                                                          self.gbest_position[2]))


visual = pcl.pcl_visualization.CloudViewing()
search_space = Space(10)
for i in range(10):
    search_space.move()
    search_space.fitness()
    search_space.set_pbest()
    search_space.set_gbest()
    print(search_space.gbest_position)
    pc_1 = pcl.PointCloud_PointXYZRGB()
    pc_1.from_array(search_space.particles_visited)
    visual.ShowColorCloud(pc_1, b'cloud')
    time.sleep(1)


