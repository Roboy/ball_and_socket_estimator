import numpy as np
import matplotlib.pyplot as plt
from magpylib.source.magnet import Box,Cylinder
from magpylib import Collection, displaySystem, Sensor
from scipy.optimize import fsolve, least_squares
import matplotlib.animation as manimation
import random
import MDAnalysis
import MDAnalysis.visualization.streamlines_3D
import mayavi, mayavi.mlab
from multiprocessing import Pool

# define sensor
sensor_pos = [(-22.7,7.7,0),(-14.7,-19.4,0),(14.7,-19.4,0),(22.7,7.7,0)]
# sensor_rot = [[0,[0,0,1]],[0,[0,0,1]],[0,[0,0,1]],[0,[0,0,1]],[0,[0,0,1]],[0,[0,0,1]],[0,[0,0,1]],[0,[0,0,1]]]
sensors = []
i = 0
for pos in sensor_pos:
    # sensors.append(Sensor(pos=pos,angle=sensor_rot[i][0], axis=sensor_rot[i][1]))
    sensors.append(Sensor(pos=pos))

def gen_magnets():
    return [Box(mag=(0,500,0),dim=(10,10,10),pos=(0,12,0)), Box(mag=(0,-500,0),dim=(10,10,10),pos=(10.392304845,-6,0),angle=60, axis=(0,0,1)), Box(mag=(0,0,500),dim=(10,10,10),pos=(-10.392304845,-6,0),angle=-60, axis=(0,0,1))]

c = Collection(gen_magnets())

x_lower,x_upper = -20, 20
y_lower,y_upper = -20, 20
z_lower,z_upper = -20, 20
grid_spacing_value = 0.1
xd = int((x_upper-x_lower)/grid_spacing_value)
yd = int((y_upper-y_lower)/grid_spacing_value)
zd = int((z_upper-z_lower)/grid_spacing_value)

x, y, z = np.mgrid[x_lower:x_upper:xd*1j,
                  y_lower:y_upper:yd*1j,
                  z_lower:z_upper:zd*1j]

xs = np.linspace(x_lower,x_upper,xd)
ys = np.linspace(y_lower,y_upper,yd)
zs = np.linspace(z_lower,z_upper,zd)

U = np.zeros((xd,yd,zd))
V = np.zeros((xd,yd,zd))
W = np.zeros((xd,yd,zd))
i = 0

pool = Pool(processes=4)

def getB(i):
    POS = np.array([(x,y,zs[i]) for x in xs for y in ys])
    Bs = c.getB(POS).reshape(len(xs),len(ys),3)
    U[:,:,i]=Bs[:,:,0]
    V[:,:,i]=Bs[:,:,1]
    W[:,:,i]=Bs[:,:,2]
    print(i)

multiple_results = [pool.apply_async(getB, ()) for i in range(4)]

print("simulating magnetic data")
for zi in zs:
    POS = np.array([(x,y,zi) for x in xs for y in ys])
    Bs = c.getB(POS).reshape(len(xs),len(ys),3)
    U[:,:,i]=Bs[:,:,0]
    V[:,:,i]=Bs[:,:,1]
    W[:,:,i]=Bs[:,:,2]
    i=i+1
    print("(%d/%d)"%(i,zd))
    # print(Bs)

# plot with mayavi:
fig = mayavi.mlab.figure(bgcolor=(1.0, 1.0, 1.0), size=(800, 800), fgcolor=(0, 0, 0))
i = 0
print("generating flow")
for z_value in np.arange(z_lower, z_upper, grid_spacing_value):

    st = mayavi.mlab.flow(x, y, z, U,V,W, line_width=0.1,
                          seedtype='plane', integration_direction='both')
    st.streamline_type = 'tube'
    st.tube_filter.radius = 0.1
    st.seed.widget.origin = np.array([ x_lower,  y_upper,   z_value])
    st.seed.widget.point1 = np.array([ x_upper, y_upper,  z_value])
    st.seed.widget.point2 = np.array([ x_lower, y_lower,  z_value])
    st.seed.widget.resolution = 200
    st.seed.widget.enabled = False
    print("(%d/%d)"%(i,zd))
    i= i+1
mayavi.mlab.axes(extent = [x_lower, x_upper, y_lower, y_upper, z_lower, z_upper])
fig.scene.z_plus_view()

print("done")
