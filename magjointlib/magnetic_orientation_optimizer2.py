#!/usr/bin/python3
import magjoint
from magjoint_particle_swarm2 import ParticleSwarm
import sys, random, time, math
import numpy as np
from tqdm import tqdm
from yaml import load,dump,Loader,Dumper


if len(sys.argv) < 5:
    print("\nUSAGE: ./magnetic_orientation_optimizer.py ball_joint_config x_step y_step visualize_only, e.g. \n python3 magnetic_orientation_optimizer.py two_magnets.yaml 10 10 1\n")
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

positions,pos_offsets,angles,angle_offsets = [],[],[],[]
for i in np.arange(-math.pi,math.pi,math.pi/180*x_step):
    for j in np.arange(-math.pi,math.pi,math.pi/180*y_step):
        positions.append([25*math.sin(i)*math.cos(j),25*math.sin(i)*math.sin(j),25*math.cos(i)])
        pos_offsets.append([0,0,0])
        angles.append([0,0,90])
        angle_offsets.append([0,0,0])
number_of_sensors = len(positions)
print('number_of_sensors %d'%number_of_sensors)
start = time.time()
sensors = ball.gen_sensors_all(positions,pos_offsets,angles,angle_offsets)


particle_swarm = ParticleSwarm(50,sensors,ball)
status_bar = tqdm(total=100, desc='particle_status', position=1)
while particle_swarm.iteration <100:
    particle_swarm.step()
    status_bar.update(1)
    if particle_swarm.global_best_score ==0:
        print('particle %d reached best possible score 0'%particle_swarm.global_best_particle)
        break
magnets = ball.gen_magnets_angle(particle_swarm.particles[particle_swarm.global_best_particle]['magnet_angles'])
ball.plotMagnets(magnets)
ball.config['magnet_angle'] = particle_swarm.particles[particle_swarm.global_best_particle]['magnet_angles']
filename = time.strftime("%Y%m%d-%H%M%S_particle_swarm_winner.yaml")
input("Enter to write particle swarm winner to config file %s..."%(filename))

with open(filename, 'w') as file:
    documents = dump(ball.config, file)
