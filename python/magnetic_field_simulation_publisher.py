import rospy
from roboy_middleware_msgs.msg import MagneticSensor
import numpy as np
import matplotlib.pyplot as plt
from magpylib.source.magnet import Box,Cylinder
from magpylib import Collection, displaySystem, Sensor
from scipy.optimize import fsolve, least_squares
import matplotlib.animation as manimation
import random, math
import sensor_msgs.msg, std_msgs
from yaml import load,dump,Loader,Dumper
import sys
from os import path

if len(sys.argv) < 3:
    print("\nUSAGE: python3 magnetic_field_simulation_publisher.py balljoint_config_yaml sensor_log_file, e.g. \n python3 magnetic_field_simulation_publisher.py test.yaml sensor_log.yaml\n")
    sys.exit()

balljoint_config = load(open(sys.argv[1], 'r'), Loader=Loader)
print(dump(balljoint_config, Dumper=Dumper))
sensor_log_file = sys.argv[2]

def gen_sensors(offset,angles):
    sensor_pos = [[-22.7,7.7,0],[-14.7,-19.4,0],[14.7,-19.4,0],[22.7,7.7,0]]
    sensors = []
    i = 0
    for pos in sensor_pos:
        s = Sensor(pos=(pos[0]+offset[i][0],pos[1]+offset[i][1],pos[2]+offset[i][2]),angle=90,axis=(0,0,1))
        s.rotate(angle=angles[i][0],axis=(1,0,0))
        s.rotate(angle=angles[i][1],axis=(0,1,0))
        s.rotate(angle=angles[i][2],axis=(0,0,1))
        sensors.append(s)
        i = i+1
    return sensors

def gen_magnets(fieldstrength,offset,angles):
    magnets = []
    field = [(0,fieldstrength[0],0),(0,0,fieldstrength[1]),(0,0,fieldstrength[2])]
    for i in range(0,3):
        magnet = Box(mag=field[i],dim=(10,10,10),\
        pos=(13*math.sin(i*(360/3)/180.0*math.pi)+offset[i][0],13*math.cos(i*(360/3)/180.0*math.pi+offset[i][1]),offset[i][2]),\
        angle=i*(360/3))
        magnet.rotate(angle=angles[i][0],axis=(1,0,0))
        magnet.rotate(angle=angles[i][1],axis=(0,1,0))
        magnet.rotate(angle=angles[i][2],axis=(0,0,1))
        magnets.append(magnet)
    return magnets

sensors = gen_sensors(balljoint_config['sensor_offsets'],balljoint_config['sensor_angles'])
rospy.init_node('magnetic_field_simulation_publisher')
magneticSensor_pub = rospy.Publisher('roboy/middleware/MagneticSensor', MagneticSensor, queue_size=1)

if(path.exists(sensor_log_file)):
    sensor_log = load(open(sensor_log_file, 'r'), Loader=Loader)
else:
    sensor_log = {'position':[],'sensor_values':[]}

def plotMagnets(magnets):
    # calculate B-field on a grid
    xs = np.linspace(-40,40,33)
    ys = np.linspace(-40,40,44)
    zs = np.linspace(-40,40,44)
    POS0 = np.array([(x,0,z) for z in zs for x in xs])
    POS1 = np.array([(x,y,0) for y in ys for x in xs])

    fig = plt.figure(figsize=(18,7))
    ax1 = fig.add_subplot(131, projection='3d')  # 3D-axis
    ax2 = fig.add_subplot(132)                   # 2D-axis
    ax3 = fig.add_subplot(133)                   # 2D-axis
    Bs = magnets.getB(POS0).reshape(44,33,3)     #<--VECTORIZED
    X,Y = np.meshgrid(xs,ys)
    U,V = Bs[:,:,0], Bs[:,:,2]
    ax2.streamplot(X, Y, U, V, color=np.log(U**2+V**2))

    Bs = magnets.getB(POS1).reshape(44,33,3)     #<--VECTORIZED
    X,Z = np.meshgrid(xs,zs)
    U,V = Bs[:,:,0], Bs[:,:,2]
    ax3.streamplot(X, Z, U, V, color=np.log(U**2+V**2))
    displaySystem(magnets, subplotAx=ax1, suppress=True, sensors=sensors, direc=True)
    plt.show()

magnets = Collection(gen_magnets(balljoint_config['field_strength'],balljoint_config['magnet_offsets'], balljoint_config['magnet_angles']))
plotMagnets(magnets)

def JointTarget(data):
    rospy.loginfo("new joint target: %f %f %f"%(data.position[0],data.position[1],data.position[2]))
    magnets = Collection(gen_magnets(balljoint_config['field_strength'],balljoint_config['magnet_offsets'], balljoint_config['magnet_angles']))
    magnets.rotate(angle=data.position[0]*180.0/math.pi,axis = (1,0,0), anchor=(0,0,0))
    magnets.rotate(angle=data.position[1]*180.0/math.pi,axis = (0,1,0), anchor=(0,0,0))
    magnets.rotate(angle=data.position[2]*180.0/math.pi,axis = (0,0,1), anchor=(0,0,0))
    msg = MagneticSensor()
    msg.id = int(balljoint_config['id'])
    values = []
    for sens in sensors:
        val = sens.getB(magnets)
        msg.x.append(val[0])
        msg.y.append(val[1])
        msg.z.append(val[2])
        values.append((val[0],val[1],val[2]))
    magneticSensor_pub.publish(msg)
    sensor_log['position'].append((data.position[0]*180.0/math.pi,data.position[1]*180.0/math.pi,data.position[2]*180.0/math.pi))
    sensor_log['sensor_values'].append(values)
    print(sensor_log)

joint_state_sub = rospy.Subscriber('joint_targets', sensor_msgs.msg.JointState, JointTarget, queue_size=1)

rospy.spin()

rospy.loginfo('writing sensor log to file %s'%sensor_log_file)
with open(sensor_log_file, 'w') as file:
    documents = dump(sensor_log, file, default_flow_style=False)

rospy.loginfo('done')
