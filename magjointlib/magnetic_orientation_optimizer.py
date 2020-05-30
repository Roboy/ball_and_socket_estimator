#!/usr/bin/python3
import magjoint
from magjoint_particle_swarm import ParticleSwarm
import sys, random, time
import numpy as np
from tqdm import tqdm
from yaml import load,dump,Loader,Dumper


if len(sys.argv) < 6:
    print("\nUSAGE: ./magnetic_orientation_optimizer.py ball_joint_config x_step y_step z_step visualize_only, e.g. \n python3 magnetic_orientation_optimizer.py two_magnets.yaml 10 10 10 1\n")
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

joint_positions = []
for i in np.arange(0,180,x_step):
    for j in np.arange(0,180,y_step):
        for k in np.arange(0,180,z_step):
            joint_positions.append([i,j,k])

# start = time.time()
# sensor_values = ball.generateSensorData(joint_positions,ball.config['magnet_angle'])
# print('data generation took: %d'%(time.time() - start))
#
# start = time.time()
# colliders,magnetic_field_differences = ball.calculateCollisions(sensor_values,joint_positions,30*1.44)
# print('collision took: %d'%(time.time() - start))

magnetic_field_difference_sensitivity = 1.44*(x_step+y_step+z_step)/3 # noise times average step size
print('magnetic_field_difference_sensitivity: %f'%magnetic_field_difference_sensitivity)

particle_swarm = ParticleSwarm(20,joint_positions,ball,magnetic_field_difference_sensitivity)
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
