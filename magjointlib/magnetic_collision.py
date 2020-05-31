#!/usr/bin/python3
import magjoint
import sys
import numpy as np

if len(sys.argv) < 2:
    print("\nUSAGE: ./magnetic_collision.py ball_joint_config visualize_only, e.g. \n python3 magnetic_collision.py two_magnets.yaml 1\n")
    sys.exit()

balljoint_config = sys.argv[1]
visualize_only = sys.argv[2]=='1'

ball = magjoint.BallJoint(balljoint_config)

magnets = ball.gen_magnets()
if visualize_only:
    ball.plotMagnets(magnets)
    sys.exit()

print('\n----------------first course search\n')
grid_positions = []
for i in np.arange(-80,80,10):
    for j in np.arange(-80,80,10):
        for k in np.arange(-80,80,10):
            grid_positions.append([i,j,k])

sensor_values,pos = ball.generateMagneticDataGrid(grid_positions)
colliders,magnetic_field_differences = ball.calculateCollisions(sensor_values,pos,1.44*10)

if len(colliders)>0:
    print('there are %d collisions'%len(colliders))
    for c,dif in zip(colliders,magnetic_field_differences):
        pos_diff = ((c[0][0]-c[1][0])**2+(c[0][0]-c[1][0])**2+(c[0][0]-c[1][0])**2)**1/2
        if pos_diff>20:
            print(c)
            print("magnetic dif %f"%dif)
            print("pos_dif %f"%pos_diff)
            magnet_A = ball.gen_magnets()
            ball.rotateMagnets(magnet_A,c[0])
            magnet_B = ball.gen_magnets()
            ball.rotateMagnets(magnet_B,c[1])
            ball.compareMagnets(magnet_A,magnet_B)
else:
    print('no collisions detected in course search, congrats!')

print('\n----------------second fine search\n')
grid_positions = []
for i in np.arange(-50,50,5):
    for j in np.arange(-50,50,5):
        for k in np.arange(-90,90,5):
            grid_positions.append([i,j,k])

sensor_values,pos = ball.generateMagneticDataGrid(grid_positions)
colliders,magnetic_field_differences = ball.calculateCollisions(sensor_values,pos,1.44*5)

if len(colliders)>0:
    print('there are %d collisions'%len(colliders))
    min_value = min(magnetic_field_differences)
    index = magnetic_field_differences.index(min_value)
    print('minimum: %f index %d'%(min_value,index))
    print(colliders[index])

grid_positions = []
magnetic_diffs = []

for c,dif in zip(colliders,magnetic_field_differences):
    grid_positions.append(c[0])
    magnetic_diffs.append(dif)
    grid_positions.append(c[1])
    magnetic_diffs.append(dif)
ball.visualizeCloud(magnetic_diffs,grid_positions)

print('\n----------------second even finer search\n')
grid_positions = []
for i in np.arange(-50,50,3):
    for j in np.arange(-50,50,3):
        for k in np.arange(-90,90,3):
            grid_positions.append([i,j,k])

sensor_values,pos = ball.generateMagneticDataGrid(grid_positions)
colliders,magnetic_field_differences = ball.calculateCollisions(sensor_values,pos,1.44*3)

if len(colliders)>0:
    print('there are %d collisions'%len(colliders))
    min_value = min(magnetic_field_differences)
    index = magnetic_field_differences.index(min_value)
    print('minimum: %f index %d'%(min_value,index))
    print(colliders[index])

grid_positions = []
magnetic_diffs = []

for c,dif in zip(colliders,magnetic_field_differences):
    grid_positions.append(c[0])
    magnetic_diffs.append(dif)
    grid_positions.append(c[1])
    magnetic_diffs.append(dif)
ball.visualizeCloud(magnetic_diffs,grid_positions)
