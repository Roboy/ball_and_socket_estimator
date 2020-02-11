import random
import numpy as np
import pcl
import pcl.pcl_visualization
import scipy
from scipy import spatial
import time
import yaml
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import rospy
import std_msgs
import sensor_msgs.msg
import roboy_simulation_msgs.msg
import sys

model_name = "upper_body"

class Space():
    def __init__(self, n_particles, body_part):
        self.n_particles = n_particles
        self.joint_names = [body_part+"_axis"+str(i) for i in range(3)]
        self.body_part = body_part
        self.particles_visited = np.zeros((n_particles,4),dtype=np.float32)
        self.best_particle = 0
        self.gbest_value = int(1000)
        self.gbest_position = np.array([0,0,0],dtype=np.float32)
        self.pbest_value = np.ones(n_particles)*int(1000)
        self.pbest_position = np.zeros((n_particles,3),dtype=np.float32)
        self.vel = np.random.rand(n_particles, 3)*0.01
        self.vel_towards_global_best = np.random.rand(n_particles, 3)*0.01
        self.vel_towards_personal_best = np.random.rand(n_particles, 3)*0.01
        self.visited = np.array([0,0,0], dtype=np.float32)
        self.visited_colored = np.array([0,0,0,float(255<<24|255<16|255<8)], dtype=np.float32)
        self.neighbors = np.ones(n_particles)*float('inf')
        self.colors = np.zeros(n_particles,dtype=np.float32)
        self.minima = np.zeros((3,1),dtype=np.float32)
        self.maxima = np.zeros((3,1),dtype=np.float32)
        self.minima[2] = -0.1
        self.maxima[2] = 0.1
        self.receiving_data = False
        for i in range(self.colors.size):
            self.colors[i] = float(random.randint(50, 255)<<16|random.randint(50, 255)<<8|random.randint(50, 255)<<0)

        # self.axis0 = rospy.Publisher('/sphere_axis0/sphere_axis0/target', std_msgs.msg.Float32 , queue_size=1)
        # self.axis1 = rospy.Publisher('/sphere_axis1/sphere_axis1/target', std_msgs.msg.Float32 , queue_size=1)
        # self.axis2 = rospy.Publisher('/sphere_axis2/sphere_axis2/target', std_msgs.msg.Float32 , queue_size=1)

        self.joint_targets_pub = rospy.Publisher("/joint_targets", sensor_msgs.msg.JointState, queue_size=1)
        self.joint_targets_msg = sensor_msgs.msg.JointState()
        self.joint_targets_msg.name = self.joint_names; #["shoulder_right_axis0", "shoulder_right_axis1", "shoulder_right_axis2"]
        self.joint_targets_msg.velocity = [0]*len(self.joint_names)
        self.joint_targets_msg.effort = [0]*len(self.joint_names)

        self.received_messages = 0
        #self.visual = pcl.pcl_visualization.CloudViewing()
        self.global_attraction = 0.01
        self.personal_attraction = 0.1
        self.random_speed = 0.1
        global model_name
        with open("/home/letrend/workspace/roboy3/src/robots/"+model_name+"/joint_limits.yaml", 'r') as stream:
            try:
                joint_limits = yaml.safe_load(stream)
                polygon = []
                for i in range(len(joint_limits[self.joint_names[0]])):
                    polygon.append((joint_limits[self.joint_names[0]][i],joint_limits[self.joint_names[1]][i]))
                    if joint_limits[self.joint_names[0]][i]<self.minima[0]:
                        self.minima[0] = joint_limits[self.joint_names[0]][i]
                    if joint_limits[self.joint_names[1]][i]<self.minima[1]:
                        self.minima[1] = joint_limits[self.joint_names[1]][i]
                    if joint_limits[self.joint_names[0]][i]>self.maxima[0]:
                        self.maxima[0] = joint_limits[self.joint_names[0]][i]
                    if joint_limits[self.joint_names[1]][i]>self.maxima[1]:
                        self.maxima[1] = joint_limits[self.joint_names[1]][i]

                self.joint_limits = Polygon(polygon)
            except yaml.YAMLError as exc:
                print(exc)
        self.particles = np.array([random.uniform(self.minima[0], self.maxima[0])[0],random.uniform(self.minima[1], self.maxima[1])[0],random.uniform(self.minima[2], self.maxima[2])[0]],dtype=np.float32)
        for i in range(self.n_particles-1):
            self.particles = np.vstack([self.particles,np.array([random.uniform(self.minima[0], self.maxima[0])[0],random.uniform(self.minima[1], self.maxima[1])[0],random.uniform(self.minima[2], self.maxima[2])[0]],dtype=np.float32)])

    def fitness(self):
        if self.visited.shape[0]>3:
            self.particles_tree = scipy.spatial.KDTree(self.particles)
            self.visited_tree = scipy.spatial.KDTree(self.visited)
            neighbors = self.particles_tree.query_ball_tree(self.visited_tree,0.1)
            for i in range(len(neighbors)):
                self.neighbors[i] = len(neighbors[i])

    def move(self):
        for i in range(self.particles.shape[0]):
            random_movement = (np.random.rand(1, 3)-0.5)*self.random_speed
            self.vel_towards_global_best[i] = (self.gbest_position - self.particles[i])*self.global_attraction
            self.vel_towards_personal_best[i] = (self.pbest_position[i] - self.particles[i])*self.personal_attraction
            if self.joint_limits.contains(Point(self.particles[i][0],self.particles[i][1])) and \
                    self.particles[i][2]>self.minima[2] and self.particles[i][2]<self.maxima[2]:
                self.vel[i] = self.vel_towards_global_best[i] + self.vel_towards_personal_best[i] + random_movement
                self.particles[i] = self.particles[i]+self.vel[i]
            else:
                self.particles[i] = np.array([random.uniform(self.minima[0], self.maxima[0])[0],random.uniform(self.minima[1], self.maxima[1])[0],random.uniform(self.minima[2], self.maxima[2])[0]],dtype=np.float32)

            # self.visited = np.vstack([self.visited,np.array([self.particles[i][0],self.particles[i][1],self.particles[i][2]], dtype=np.float32)])
            # self.visited_colored = np.vstack([self.visited_colored,np.array([self.particles[i][0],self.particles[i][1],self.particles[i][2],self.colors[i]], dtype=np.float32)])

    def set_pbest(self):
        for particle in range(self.particles.shape[0]):
            if self.neighbors[particle] <= self.pbest_value[particle]:
                self.pbest_value[particle] = self.neighbors[particle]
                self.pbest_position[particle] = self.particles[particle]

    def set_gbest(self):
        new_global_best = False
        for particle in range(self.particles.shape[0]):
            if (self.neighbors[particle] <= self.gbest_value) and self.best_particle!=particle:
                self.gbest_value = self.neighbors[particle]
                self.gbest_position = self.particles[particle]
                self.best_particle = particle
                new_global_best = True
        if not new_global_best:
            random_particle = random.randint(0, self.n_particles-1)
            self.best_particle = random_particle
            self.gbest_value = self.neighbors[random_particle]
            self.gbest_position = self.particles[random_particle]
            rospy.loginfo_throttle(1,"choosing random particle %d with %d neighbors at (%f %f %f)"%(random_particle,self.gbest_value,
                                                                          self.gbest_position[0],self.gbest_position[1],
                                                                          self.gbest_position[2]))
        else:
            rospy.loginfo_throttle(1,"choosing particle %d with %d neighbors at (%f %f %f)"%(self.best_particle, self.gbest_value,
                                                                          self.gbest_position[0],self.gbest_position[1],
                                                                          self.gbest_position[2]))
        # self.axis0.publish(self.gbest_position[0])
        # self.axis1.publish(self.gbest_position[1])
        # self.axis2.publish(self.gbest_position[2])
        self.joint_targets_msg.position = [self.gbest_position[0], self.gbest_position[1], self.gbest_position[2]]
        self.joint_targets_pub.publish(self.joint_targets_msg)
        time.sleep(1)

    def trackingCallback(self,data):
        # if moself.received_messages,10) ==0:
        self.visited = np.vstack([self.visited,np.array([data.position[0],data.position[1],data.position[2]], dtype=np.float32)])
        self.visited_colored = np.vstack([self.visited_colored,np.array([data.position[0],data.position[1],data.position[2],self.colors[self.best_particle]], dtype=np.float32)])
        self.receiving_data = True
        rospy.loginfo_throttle(5, "receiving tracking data for %s"%self.body_part)
        search_space.move()
        search_space.fitness()
        search_space.set_pbest()
        search_space.set_gbest()
        # print(search_space.gbest_position)
        pc_1 = pcl.PointCloud_PointXYZRGB()
        pc_1.from_array(search_space.visited_colored)
        #self.visual.ShowColorCloud(pc_1, b'particle_swarm')
        # time.sleep(0.5)

    def run(self):
        trackingSubscriber = rospy.Subscriber("external_joint_states", sensor_msgs.msg.JointState, self.trackingCallback, queue_size=1)
        rospy.spin()


if len(sys.argv) < 2:
    print("\nUSAGE: python particle_swarm.py body_part, e.g. \n python particle_swarm.py shoulder_left \n")
    sys.exit()
body_part = sys.argv[1] #joint_names[0].split("_")[0] + "_" + oint_names[0].split("_")[1]
rospy.init_node(body_part+'_particle_swarm')
search_space = Space(10, body_part)
search_space.run()
