import numpy as np
import rospy
from roboy_middleware_msgs.msg import MagneticSensor
import std_msgs.msg, sensor_msgs.msg
import rospkg
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
import tf2_ros
from mayavi import mlab
import math, os, time
import geometry_msgs
from multiprocessing import Pool, freeze_support
from pathlib import Path
import tf_conversions
import tf

rospy.init_node('magnetic_field_pose_estimation')

broadcaster = tf2_ros.TransformBroadcaster()
b_target = [(0,0,0),(0,0,0),(0,0,0),(0,0,0)]
# define sensor
sensor_pos = [[-22.7,7.7,0],[-14.7,-19.4,0],[14.7,-19.4,0],[22.7,7.7,0]]#[[22.7,7.7,0],[14.7,-19.4,0],[-14.7,-19.4,0],[-22.7,7.7,0]]
# sensor_rot = [[0,[0,0,1]],[0,[0,0,1]],[0,[0,0,1]],[0,[0,0,1]],[0,[0,0,1]],[0,[0,0,1]],[0,[0,0,1]],[0,[0,0,1]]]
sensors = []
i = 0
for pos in sensor_pos:
    # sensors.append(Sensor(pos=pos,angle=sensor_rot[i][0], axis=sensor_rot[i][1]))
    s = Sensor(pos=pos,angle=90,axis=(0,0,1))
    sensors.append(s)
# def gen_magnets():
#     return [Box(mag=(500,0,0),dim=(10,10,10),pos=(0,12,0)), Box(mag=(0,500,0),dim=(10,10,10),pos=(10.392304845,-6,0),angle=60, axis=(0,0,1)), Box(mag=(0,0,500),dim=(10,10,10),pos=(-10.392304845,-6,0),angle=-60, axis=(0,0,1))]

field_strenght = -1000
# hallbach 0, works well
def gen_magnets():
    magnets = []
    magnets.append(Box(mag=(0,0,-field_strenght),dim=(5,5,5),pos=(0,0,0)))
    magnets.append(Box(mag=(-field_strenght,0,0),dim=(5,5,5),pos=(-5,5,0)))
    magnets.append(Box(mag=(-field_strenght,0,0),dim=(5,5,5),pos=(-5,0,0)))
    magnets.append(Box(mag=(-field_strenght,0,0),dim=(5,5,5),pos=(-5,-5,0)))
    magnets.append(Box(mag=(field_strenght,0,0),dim=(5,5,5),pos=(5,0,0)))
    magnets.append(Box(mag=(field_strenght,0,0),dim=(5,5,5),pos=(5,-5,0)))
    magnets.append(Box(mag=(0,-field_strenght,0),dim=(5,5,5),pos=(0,-5,0)))
    magnets.append(Box(mag=(field_strenght,0,0),dim=(5,5,5),pos=(5,5,0)))
    magnets.append(Box(mag=(0,field_strenght,0),dim=(5,5,5),pos=(0,5,0)))
    return magnets

def func(x):
    c = Collection(gen_magnets())
    c.rotate(x[0],(1,0,0), anchor=(0,0,0))
    c.rotate(x[1],(0,1,0), anchor=(0,0,0))
    c.rotate(x[2],(0,0,1), anchor=(0,0,0))
    b_error = 0
    i = 0
    for sens in sensors:
        b_error = b_error + np.dot(sens.getB(c),b_target[i])#,np.linalg.norm(sens.getB(c)-b_target[i])
        i=i+1
    # print(b_error)
    return [b_error,b_error,b_error]

def magneticsCallback(data):
    for i in range(0,4):
        b_target[i] = (data.x[i],data.y[i],data.z[i])
    res = least_squares(func, [0,0,0], bounds = ((-360,-360,-360), (360, 360, 360)))
    b_field_error = res.cost
    print("result %.3f %.3f %.3f b-field error %.3f\nb_target %.3f %.3f %.3f\t%.3f %.3f %.3f\t%.3f %.3f %.3f\t%.3f %.3f %.3f"%(res.x[0],res.x[1],res.x[2],res.cost,b_target[0][0],b_target[0][1],b_target[0][2],b_target[1][0],b_target[1][1],b_target[1][2],b_target[2][0],b_target[2][1],b_target[2][2],b_target[3][0],b_target[3][1],b_target[3][2]))
    t = geometry_msgs.msg.TransformStamped()

    t.header.stamp = rospy.Time.now()
    t.header.frame_id = "world"
    t.child_frame_id = "predict"
    t.transform.translation.x = 0.0
    t.transform.translation.y = 0.0
    t.transform.translation.z = 0.0
    q = tf.transformations.quaternion_from_euler(res.x[0]/180*math.pi,res.x[1]/180*math.pi,res.x[2]/180*math.pi)
    t.transform.rotation.x = q[0]
    t.transform.rotation.y = q[1]
    t.transform.rotation.z = q[2]
    t.transform.rotation.w = q[3]

    broadcaster.sendTransform(t)
    # result = Collection(gen_magnets())
    # result.rotate(res.x[0],(1,0,0), anchor=(0,0,0))
    # result.rotate(res.x[1],(0,1,0), anchor=(0,0,0))
    # result.rotate(res.x[2],(0,0,1), anchor=(0,0,0))
    # c = Collection(result)
    # # create figure
    # fig = plt.figure(figsize=(18,7))
    # ax1 = fig.add_subplot(121, projection='3d')  # 3D-axis
    # ax2 = fig.add_subplot(122)                   # 2D-axis
    #
    # # calculate B-field on a grid
    # xs = np.linspace(-30,30,33)
    # ys = np.linspace(-30,30,44)
    # POS = np.array([(x,y,0) for y in ys for x in xs])
    #
    # Bs = c.getB(POS).reshape(44,33,3)     #<--VECTORIZED
    # X,Y = np.meshgrid(xs,ys)
    # U,V = Bs[:,:,0], Bs[:,:,2]
    # ax2.streamplot(X, Y, U, V, color=np.log(U**2+V**2))
    #
    # displaySystem(c, subplotAx=ax1, suppress=True, sensors=sensors, direc=True)
    # fig.savefig("/home/letrend/Pictures/result.png")

rospy.Subscriber("roboy/middleware/MagneticSensor", MagneticSensor, magneticsCallback,queue_size=1)
rospy.spin()
