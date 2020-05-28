#!/usr/bin/python3
import magjoint
import sys
import numpy as np

if len(sys.argv) < 3:
    print("\nUSAGE: ./magnetic_collision.py dataset_name ball_joint_config visualize_only, e.g. \n python3 magnetic_collision.py head_data0.log two_magnets.yaml 1\n")
    sys.exit()

dataset_name = sys.argv[1]
balljoint_config = sys.argv[2]
visualize_only = sys.argv[3]=='1'

ball = magjoint.BallJoint(balljoint_config)

magnets = ball.gen_magnets()
if visualize_only:
    ball.plotMagnets(magnets)
    sys.exit()

print('\nfirst course search\n')

grid_positions = []
for i in np.arange(-55,55,5):
    for j in np.arange(-55,55,5):
        for k in np.arange(-90,90,10):
            grid_positions.append([i,j,k])

sensor_values,pos = ball.generateMagneticDataGrid(grid_positions)
colliders,magnetic_field_differences = ball.calculateCollisions(sensor_values,pos,15)

if len(colliders)>0:
    min_value = min(magnetic_field_differences)
    index = magnetic_field_differences.index(min_value)
    print('minimum: %f index %d'%(min_value,index))
    print(colliders[index])

grid_positions = []
for c,dif in zip(colliders,magnetic_field_differences):
    pos_diff = ((c[0][0]-c[1][0])**2+(c[0][0]-c[1][0])**2+(c[0][0]-c[1][0])**2)**1/2
    if pos_diff>20:
        print('there are symmetry problems')
        print(c)
        print("magnetic dif %f"%dif)
        print("pos_dif %f"%pos_diff)
        magnet_A = ball.gen_magnets()
        ball.rotateMagnets(magnet_A,c[0])
        magnet_B = ball.gen_magnets()
        ball.rotateMagnets(magnet_B,c[1])
        ball.compareMagnets(magnet_A,magnet_B)
    for i in np.arange(-3,3.1,3):
        for j in np.arange(-3,3.1,3):
            for k in np.arange(-3,3.1,3):
                grid_positions.append([c[0][0]+i,c[0][1]+j,c[0][2]+k])
                grid_positions.append([c[1][0]+i,c[1][1]+j,c[1][2]+k])
if len(colliders)==0:
    print('no collisions detected in course search, congrats!')
    sys.exit()
else:
    print('\nsecond fine search\n')

sensor_values,pos = ball.generateMagneticDataGrid(grid_positions)
colliders,magnetic_field_differences = ball.calculateCollisions(sensor_values,pos,1.44)

grid_positions = []
magnetic_diffs = []

for c,dif in zip(colliders,magnetic_field_differences):
    grid_positions.append(c[0])
    magnetic_diffs.append(dif)
    grid_positions.append(c[1])
    magnetic_diffs.append(dif)
ball.visualizeCloud(magnetic_diffs,grid_positions)
