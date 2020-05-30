#!/usr/bin/python3
import magjoint
from magjoint_particle_swarm import ParticleSwarm
import sys, random, time
import numpy as np


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
for i in np.arange(0,360,x_step):
    for j in np.arange(0,360,y_step):
        for k in np.arange(0,360,z_step):
            joint_positions.append([i,j,k])

# start = time.time()
# sensor_values = ball.generateSensorData(joint_positions,ball.config['magnet_angle'])
# print('data generation took: %d'%(time.time() - start))
#
# start = time.time()
# colliders,magnetic_field_differences = ball.calculateCollisions(sensor_values,joint_positions,30*1.44)
# print('collision took: %d'%(time.time() - start))

particle_swarm = ParticleSwarm(30,joint_positions,ball)
while particle_swarm.iteration <100:
    particle_swarm.step()
