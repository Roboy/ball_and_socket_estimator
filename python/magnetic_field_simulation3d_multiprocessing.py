import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from magpylib.source.magnet import Box,Cylinder
from magpylib import Collection, displaySystem, Sensor
from scipy.optimize import fsolve, least_squares
import matplotlib.animation as manimation
import random
import MDAnalysis
import MDAnalysis.visualization.streamlines_3D
import mayavi
from mayavi import mlab
import math, os, time
from multiprocessing import Pool, freeze_support
from pathlib import Path

# define sensor
sensor_pos = [(-22.7,7.7,0),(-14.7,-19.4,0),(14.7,-19.4,0),(22.7,7.7,0)]
# sensor_rot = [[0,[0,0,1]],[0,[0,0,1]],[0,[0,0,1]],[0,[0,0,1]],[0,[0,0,1]],[0,[0,0,1]],[0,[0,0,1]],[0,[0,0,1]]]
sensors = []
i = 0
for pos in sensor_pos:
    # sensors.append(Sensor(pos=pos,angle=sensor_rot[i][0], axis=sensor_rot[i][1]))
    sensors.append(Sensor(pos=pos))

iterations = 360
num_processes = 60
printouts = False
movie_path = '/home/letrend/Videos/magnetic_arrangements/cylindrical_60degrees_ring'

def gen_magnets():
    # return [Box(mag=(500,0,0),dim=(10,10,10),pos=(0,0,10)), Box(mag=(0,500,0),dim=(10,10,10),pos=(0,0,-10))]
    # return [Box(mag=(0,0,500),dim=(10,10,10),pos=(0,0,12)), Box(mag=(0,500,0),dim=(10,10,10),pos=(0,12,0)), Box(mag=(0,500,0),dim=(10,10,10),pos=(10.392304845,-6,0),angle=60, axis=(0,0,1)), Box(mag=(0,500,0),dim=(10,10,10),pos=(-10.392304845,-6,0),angle=-60, axis=(0,0,1))]
    magnets = []
    for i in range(0,360,60):
        magnets.append(Cylinder(mag=(500,0,0),dim=(4,4),angle=i,axis=(0,0,1),pos=(math.sin(i/180.0*math.pi)*12,math.cos(i/180.0*math.pi)*12,0)))
    return magnets

matplotlib.use('Agg')
mlab.options.offscreen = True


def func1(iter):
    c = Collection(gen_magnets())
    rot = [0,iter,0]
    c.rotate(rot[0],(1,0,0), anchor=(0,0,0))
    c.rotate(rot[1],(0,1,0), anchor=(0,0,0))
    c.rotate(rot[2],(0,0,1), anchor=(0,0,0))
    x_lower,x_upper = -30, 30
    y_lower,y_upper = -30, 30
    z_lower,z_upper = -30, 30
    grid_spacing_value = 2#0.25
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
    if printouts:
        print("simulating magnetic data")
    for zi in zs:
        POS = np.array([(x,y,zi) for x in xs for y in ys])
        Bs = c.getB(POS).reshape(len(xs),len(ys),3)
        U[:,:,i]=Bs[:,:,0]
        V[:,:,i]=Bs[:,:,1]
        W[:,:,i]=Bs[:,:,2]
        i=i+1
        if printouts:
            print("(%d/%d)"%(i,zd))


    i = 0
    if printouts:
        print("generating flow whole cube")

    fig = mlab.figure(bgcolor=(1,1,1), size=(1500, 1500), fgcolor=(0, 0, 0))
    fig.scene.disable_render = True

    for xi in xs:
        st = mlab.flow(x, y, z, U,V,W, line_width=0.1,
                       seedtype='plane', integration_direction='both',figure=fig,opacity=0.05)
        st.streamline_type = 'tube'
        st.seed.visible = False
        st.tube_filter.radius = 0.1
        st.seed.widget.origin = np.array([ xi, y_lower,  z_upper])
        st.seed.widget.point1 = np.array([ xi, y_upper,  z_upper])
        st.seed.widget.point2 = np.array([ xi, y_lower,  z_lower])
        st.seed.widget.resolution = 10#int(xs.shape[0])
        st.seed.widget.enabled = True
        st.seed.widget.handle_property.opacity = 0
        st.seed.widget.plane_property.opacity = 0
        if printouts:
            print("(%d/%d)"%(i,xd))
        i= i+1
    fig.scene.disable_render = False
    mayavi.mlab.view(azimuth=-50, elevation=70)
    mlab.axes(extent = [x_lower, x_upper, y_lower, y_upper, z_lower, z_upper],figure=fig)
    mlab.savefig(movie_path+'/movie001/'+'anim%05d.png'%(iter))
    mlab.clf(fig)
    fig = mlab.figure(bgcolor=(1,1,1), size=(1500, 1500), fgcolor=(0, 0, 0))
    fig.scene.disable_render = True
    if printouts:
        print("generating flow sensor plane")
    for zi in np.linspace(-1,1,10):
        st = mlab.flow(x, y, z, U,V,W, line_width=0.1,
                       seedtype='plane', integration_direction='both',figure=fig,opacity=0.05)
        st.streamline_type = 'tube'
        st.seed.visible = False
        st.tube_filter.radius = 0.1
        st.seed.widget.origin = np.array([ x_lower,  y_upper, zi])
        st.seed.widget.point1 = np.array([ x_upper,  y_upper, zi])
        st.seed.widget.point2 = np.array([ x_lower,  y_lower, zi])
        st.seed.widget.resolution = 10#int(xs.shape[0])
        st.seed.widget.enabled = True
        st.seed.widget.handle_property.opacity = 0
        st.seed.widget.plane_property.opacity = 0
        if printouts:
            print("(%d/%d)"%(i,xd))
        i= i+1
    fig.scene.disable_render = False
    mayavi.mlab.view(azimuth=-50, elevation=70)
    mlab.axes(extent = [x_lower, x_upper, y_lower, y_upper, z_lower, z_upper],figure=fig)
    mlab.savefig(movie_path+'/movie002/'+'anim%05d.png'%(iter))
    mlab.clf(fig)

angle_error = np.zeros(360)
b_field_error = np.zeros(360)

def func2(iter):
    c = Collection(gen_magnets())
    rot = [0,iter,0]
    c.rotate(rot[0],(1,0,0), anchor=(0,0,0))
    c.rotate(rot[1],(0,1,0), anchor=(0,0,0))
    c.rotate(rot[2],(0,0,1), anchor=(0,0,0))
    b_target = []
    for sens in sensors:
        b_target.append(sens.getB(c))

    # calculate B-field on a grid
    xs = np.linspace(-30,30,33)
    ys = np.linspace(-30,30,44)
    POS = np.array([(x,y,0) for y in ys for x in xs])

    fig = plt.figure(figsize=(24,15))
    ax1 = fig.add_subplot(121, projection='3d')  # 3D-axis
    ax2 = fig.add_subplot(122)                   # 2D-axis

    Bs = c.getB(POS).reshape(44,33,3)     #<--VECTORIZED
    X,Y = np.meshgrid(xs,ys)
    U,V = Bs[:,:,0], Bs[:,:,1]
    plt.xlabel("x")
    plt.ylabel("y")
    ax2.streamplot(X,Y,U,V, color=np.log(U**2+V**2))

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

    if printouts:
        print("starting pose estimator")
    res = least_squares(func, [0,0,0], bounds = ((-360,-360,-360), (360, 360, 360)))
    angle_error[iter] = ((rot[0]-res.x[0])**2+(rot[1]-res.x[1])**2+(rot[2]-res.x[2])**2)**0.5
    b_field_error[iter] = res.cost
    if printouts:
        print("target %.3f %.3f %.3f result %.3f %.3f %.3f b-field error %.3f, angle_error %.3f"%(rot[0],rot[1],rot[2],res.x[0],res.x[1],res.x[2],res.cost,angle_error[iter]))
    c = Collection(gen_magnets())
    c.rotate(rot[0],(1,0,0), anchor=(0,0,0))
    c.rotate(rot[1],(0,1,0), anchor=(0,0,0))
    c.rotate(rot[2],(0,0,1), anchor=(0,0,0))
    result = Collection(gen_magnets())
    result.rotate(res.x[0],(1,0,0), anchor=(0,0,0))
    result.rotate(res.x[1],(0,1,0), anchor=(0,0,0))
    result.rotate(res.x[2],(0,0,1), anchor=(0,0,0))
    d = Collection(c,result)
    displaySystem(d, subplotAx=ax1, suppress=True, sensors=sensors)
    fig.savefig(movie_path+'/movie003/'+'anim%05d.png'%(iter))
    return (angle_error[iter],b_field_error[iter])

def func3(iter):
    fig = plt.figure(figsize=(6,15))
    ax1 = plt.subplot(211)
    ax1.set_ylim(0, 360)
    plt.plot(angle_error[0:iter],color='r',linewidth=3)
    plt.ylabel('angle error in degree')
    plt.xlabel('angle in degree')
    plt.subplot(212)
    plt.plot(b_field_error[0:iter],color='r',linewidth=3)
    plt.ylabel('b-field error')
    plt.xlabel('angle in degree')
    fig.savefig(movie_path+'/movie004/'+'anim%05d.png'%(iter))

def main():
    Path(movie_path).mkdir(parents=True, exist_ok=True)

    Path(movie_path+'/movie001').mkdir(parents=True, exist_ok=True)
    Path(movie_path+'/movie002').mkdir(parents=True, exist_ok=True)
    Path(movie_path+'/movie003').mkdir(parents=True, exist_ok=True)
    Path(movie_path+'/movie004').mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    print("starting flow plot")
    for i in range(0,iterations,num_processes):
        args = range(i,i+num_processes,1)
        with Pool(processes=num_processes) as pool:
            results = pool.starmap(func1, zip(args))
        elapsed_time = time.time() - start_time
        print("done batch %d-%d, elapsed time %.3f"%(i,i+num_processes,elapsed_time))
    elapsed_time = time.time() - start_time
    print("done flow plot, elapsed time %.3f"%elapsed_time)

    print("starting poseestimator")
    args = range(0,iterations,1)
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(func2, zip(args))
        for i in range(0,iterations):
            angle_error[i] = results[i][0]
            b_field_error[i] = results[i][1]

    elapsed_time = time.time() - start_time
    print("done poseestimator, elapsed time %.3f"%elapsed_time)

    with Pool(processes=num_processes) as pool:
        results = pool.starmap(func3, zip(args))
    elapsed_time = time.time() - start_time
    print("done error plot, elapsed time %.3f"%elapsed_time)

    os.system("ffmpeg -framerate 20 -i " + movie_path+"/movie001/anim%05d.png -c:v libx264 -r 30 -pix_fmt yuv420p "+ movie_path+"/movie001.mp4 -y")
    os.system("ffmpeg -framerate 20 -i " + movie_path+"/movie002/anim%05d.png -c:v libx264 -r 30 -pix_fmt yuv420p "+ movie_path+"/movie002.mp4 -y")
    os.system("ffmpeg -framerate 20 -i " + movie_path+"/movie003/anim%05d.png -c:v libx264 -r 30 -pix_fmt yuv420p "+ movie_path+"/movie003.mp4 -y")
    os.system("ffmpeg -framerate 20 -i " + movie_path+"/movie004/anim%05d.png -c:v libx264 -r 30 -pix_fmt yuv420p "+ movie_path+"/movie004.mp4 -y")
    os.system("ffmpeg -i "+movie_path+"/movie001.mp4 -i "+movie_path+"/movie002.mp4 -filter_complex hstack "+movie_path+"/combined0.mp4 -y")
    os.system("ffmpeg -i "+movie_path+"/movie004.mp4 -i "+movie_path+"/movie003.mp4 -filter_complex hstack "+movie_path+"/combined1.mp4 -y")
    os.system("ffmpeg -i "+movie_path+"/combined1.mp4 -i "+movie_path+"/combined0.mp4 -filter_complex vstack "+movie_path+"/result.mp4 -y")
    print("finished, elapsed time %.3f"%elapsed_time)
if __name__=="__main__":
    freeze_support()
    main()