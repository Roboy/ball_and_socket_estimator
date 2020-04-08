import numpy as np
import matplotlib.pyplot as plt
from magpylib.source.magnet import Box,Cylinder
from magpylib import Collection, displaySystem, Sensor
from scipy.optimize import fsolve, least_squares
import matplotlib.animation as manimation
import random, math

iterations = 300000
normalize_magnetic_strength = True

# define sensor
sensor_pos = [[-22.4, 7.25, 0.25],[-14.0, -19.5, 0.25],[14.3, -19.6, 0.25],[22.325, 6.675, 0.25]]#[[-22.4, 7.25, 0.9],[-14.0, -19.5, 0.9],[14.3, -19.6, 0.9],[22.325, 6.675, 0.9]]
# sensor_rot = [[0,[0,0,1]],[0,[0,0,1]],[0,[0,0,1]],[0,[0,0,1]],[0,[0,0,1]],[0,[0,0,1]],[0,[0,0,1]],[0,[0,0,1]]]
sensors = []
i = 0
for pos in sensor_pos:
    # sensors.append(Sensor(pos=pos,angle=sensor_rot[i][0], axis=sensor_rot[i][1]))
    s = Sensor(pos=pos,angle=90,axis=(0,0,1))
    sensors.append(s)
# def gen_magnets():
#     return [Box(mag=(500,0,0),dim=(10,10,10),pos=(0,12,0)), Box(mag=(0,500,0),dim=(10,10,10),pos=(10.392304845,-6,0),angle=60, axis=(0,0,1)), Box(mag=(0,0,500),dim=(10,10,10),pos=(-10.392304845,-6,0),angle=-60, axis=(0,0,1))]

# # hallbach 0, works well
# def gen_magnets():
#     magnets = []
#     magnets.append(Box(mag=(0,0,-field_strenght),dim=(5,5,5),pos=(0,0,0)))
#     magnets.append(Box(mag=(-field_strenght,0,0),dim=(5,5,5),pos=(-6,6,0)))
#     magnets.append(Box(mag=(-field_strenght,0,0),dim=(5,5,5),pos=(-6,0,0)))
#     magnets.append(Box(mag=(-field_strenght,0,0),dim=(5,5,5),pos=(-6,-6,0)))
#     magnets.append(Box(mag=(field_strenght,0,0),dim=(5,5,5),pos=(6,0,0)))
#     magnets.append(Box(mag=(field_strenght,0,0),dim=(5,5,5),pos=(6,-6,0)))
#     magnets.append(Box(mag=(0,-field_strenght,0),dim=(5,5,5),pos=(0,-6,0)))
#     magnets.append(Box(mag=(field_strenght,0,0),dim=(5,5,5),pos=(6,6,0)))
#     magnets.append(Box(mag=(0,field_strenght,0),dim=(5,5,5),pos=(0,6,0)))
#     return magnets

field_strenght = 1300
# hallbach 0, works well
def gen_magnets():
    magnets = []
    magnets.append(Box(mag=(0,0,field_strenght),dim=(10,10,10),pos=(0,0,0)))
    magnets.append(Box(mag=(0,field_strenght,0),dim=(10,10,10),pos=(-(10+0.5),0,0)))
    magnets.append(Box(mag=(0,-field_strenght,0),dim=(10,10,10),pos=(10+0.5,0,0)))
    return magnets

# calculate B-field on a grid
xs = np.linspace(-40,40,33)
ys = np.linspace(-40,40,44)
zs = np.linspace(-40,40,44)
POS0 = np.array([(x,0,z) for z in zs for x in xs])
POS1 = np.array([(x,y,0) for y in ys for x in xs])

body_part = "test"

record = open("/home/letrend/workspace/roboy3/"+body_part+"_data0.log","w")
record.write("mx0 my0 mz0 mx1 my1 mz1 mx2 my2 mz3 mx3 my3 mz3 roll pitch yaw\n")

first = True

for iter in range(iterations):
    rot = [random.uniform(-80,80),random.uniform(-80,80),random.uniform(-80,80)]
    # rot = [0,0,0]

    c = Collection(gen_magnets())
    c.rotate(rot[0],(1,0,0), anchor=(0,0,0))
    c.rotate(rot[1],(0,1,0), anchor=(0,0,0))
    c.rotate(rot[2],(0,0,1), anchor=(0,0,0))
    data = []
    for sens in sensors:
        val = sens.getB(c)
        if normalize_magnetic_strength:
            val /= np.linalg.norm(val)
        data.append(val)
    if first:
        print("\n%.3f %.3f %.3f\n%.3f %.3f %.3f\n%.3f %.3f %.3f\n%.3f %.3f %.3f"%(data[0][0],data[0][1],data[0][2],data[1][0],data[1][1],data[1][2],data[2][0],data[2][1],data[2][2],data[3][0],data[3][1],data[3][2]))
        # create figure
        fig = plt.figure(figsize=(18,7))
        ax1 = fig.add_subplot(131, projection='3d')  # 3D-axis
        ax2 = fig.add_subplot(132)                   # 2D-axis
        ax3 = fig.add_subplot(133)                   # 2D-axis
        Bs = c.getB(POS0).reshape(44,33,3)     #<--VECTORIZED
        X,Y = np.meshgrid(xs,ys)
        U,V = Bs[:,:,0], Bs[:,:,2]
        ax2.streamplot(X, Y, U, V, color=np.log(U**2+V**2))

        Bs = c.getB(POS1).reshape(44,33,3)     #<--VECTORIZED
        X,Z = np.meshgrid(xs,zs)
        U,V = Bs[:,:,0], Bs[:,:,2]
        ax3.streamplot(X, Z, U, V, color=np.log(U**2+V**2))
        sensor_visualization = []
        i = 0
        for pos in sensor_pos:
             sensor_visualization.append(Box(mag=(0,0,0.001),dim=(1,1,1),pos=sensor_pos[i]))
             i = i+1
        d = Collection(c,sensor_visualization)
        displaySystem(d, subplotAx=ax1, suppress=True, sensors=sensors, direc=True)
        plt.show()
        first = False

    record.write(str(data[0][0])+ " " + str(data[0][1]) + " " + str(data[0][2])+ " " + str(data[1][0])+ " " + str(data[1][1])+ " " + str(data[1][2])+ " " + str(data[2][0])+ " " + str(data[2][1])+ " " + str(data[2][2])  + " " + str(data[3][0])+ " " + str(data[3][1])+ " " + str(data[3][2])+ " " + str(rot[0]/180.0*math.pi) + " " + str(rot[1]/180.0*math.pi) + " " + str(rot[2]/180.0*math.pi) + "\n")

    if iter%1000==0:
        print("(%d/%d)"%(iter,iterations))
record.close()
