import numpy as np
import matplotlib.pyplot as plt
from magpylib.source.magnet import Box,Cylinder
from magpylib import Collection, displaySystem, Sensor
from scipy.optimize import fsolve, least_squares
import matplotlib.animation as manimation
import random

iterations = 100

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=1, metadata=metadata)

# define sensor
sensor_pos = [[-22.7,7.7,0],[-14.7,19.4,0],[14.7,19.4,0],[22.7,7.7,0]]
# sensor_rot = [[0,[0,0,1]],[0,[0,0,1]],[0,[0,0,1]],[0,[0,0,1]],[0,[0,0,1]],[0,[0,0,1]],[0,[0,0,1]],[0,[0,0,1]]]
sensors = []
i = 0
for pos in sensor_pos:
    # sensors.append(Sensor(pos=pos,angle=sensor_rot[i][0], axis=sensor_rot[i][1]))
    sensors.append(Sensor(pos=pos))

def gen_magnets():
    return [Box(mag=(500,0,0),dim=(30,30,30),pos=(0,0,0))]

# calculate B-field on a grid
xs = np.linspace(-30,30,33)
zs = np.linspace(-30,30,44)
POS = np.array([(x,0,z) for z in zs for x in xs])

# create figure
fig = plt.figure(figsize=(9,5))
ax1 = fig.add_subplot(121, projection='3d')  # 3D-axis
ax2 = fig.add_subplot(122)                   # 2D-axis

first = True

with writer.saving(fig, "writer_test.mp4", 100):
    for iter in range(iterations):
        rot = [random.uniform(-180,180),random.uniform(-180,180),random.uniform(-180,180)]

        c = Collection(gen_magnets())
        c.rotate(rot[0],(1,0,0), anchor=(0,0,0))
        c.rotate(rot[1],(0,1,0), anchor=(0,0,0))
        c.rotate(rot[2],(0,0,1), anchor=(0,0,0))
        b_target = []
        for sens in sensors:
            b_target.append(sens.getB(c))
        # print(b_target)

        fig.clear()
        ax1 = fig.add_subplot(121, projection='3d')  # 3D-axis
        ax2 = fig.add_subplot(122)                   # 2D-axis


        Bs = c.getB(POS).reshape(44,33,3)     #<--VECTORIZED
        X,Z = np.meshgrid(xs,zs)
        U,V = Bs[:,:,0], Bs[:,:,2]
        ax2.streamplot(X, Z, U, V, color=np.log(U**2+V**2))

        def func(x):
            c = Collection(gen_magnets())
            c.rotate(x[0],(1,0,0), anchor=(0,0,0))
            c.rotate(x[1],(0,1,0), anchor=(0,0,0))
            c.rotate(x[2],(0,0,1), anchor=(0,0,0))
            b_error = 0
            i = 0
            for sens in sensors:
                b_error = b_error + np.linalg.norm(sens.getB(c)-b_target[i])
                i=i+1
            # print(b_error)
            return [b_error,b_error,b_error]

        res = least_squares(func, [1,1,1], bounds = ((-180, -180, -180), (180, 180, 180)))
        print("iteration (%d/%d) target %d %d %d result %d %d %d"%(iter,iterations,rot[0],rot[1],rot[2],res.x[0],res.x[1],res.x[2]))
        c = Collection(gen_magnets())
        c.rotate(rot[0],(1,0,0), anchor=(0,0,0))
        c.rotate(rot[1],(0,1,0), anchor=(0,0,0))
        c.rotate(rot[2],(0,0,1), anchor=(0,0,0))
        result = Collection(gen_magnets())
        result.rotate(res.x[0],(1,0,0), anchor=(0,0,0))
        result.rotate(res.x[1],(0,1,0), anchor=(0,0,0))
        result.rotate(res.x[2],(0,0,1), anchor=(0,0,0))
        d = Collection(c,result)
        displaySystem(d, subplotAx=ax1, suppress=True)
        if first:
            plt.show()
            first = False
        writer.grab_frame()
