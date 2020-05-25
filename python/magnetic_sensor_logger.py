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
    print("\nUSAGE: python3 magnetic_sensor_logger.py balljoint_config sensor_log_file, e.g. \n python3 magnetic_sensor_logger.py balljoint_config.yaml sensor_log.yaml\n")
    sys.exit()
balljoint_config = load(open(sys.argv[1], 'r'), Loader=Loader)
print(dump(balljoint_config, Dumper=Dumper))
sensor_log_file = sys.argv[2]
if(path.exists(sensor_log_file)):
    sensor_log = load(open(sensor_log_file, 'r'), Loader=Loader)
else:
    sensor_log = {'position':[],'sensor_values':[]}

ball_pos = [[0,0,0],[0,0,math.pi/2],[0,0,-math.pi/2]]
k = 0

def MagneticSensorCallback(data):
    global k

    if k>=len(ball_pos):
        rospy.signal_shutdown("no more")
        return

    msg = MagneticSensor()
    if data.id is not int(balljoint_config['id']):
        return

    values = []
    i =0
    for sens in data.x:
        values.append((data.x[i],data.y[i],data.z[i]))
        i = i+1
    sensor_log['position'].append(ball_pos[k])
    sensor_log['sensor_values'].append(values)
    k = k+1
    print(sensor_log)

    input("Press Enter for next ball position: %f %f %f"%(ball_pos[k][0],ball_pos[k][1],ball_pos[k][2]))

rospy.init_node('magnetic_sensor_logger')
magneticSensor_sub = rospy.Subscriber('roboy/middleware/MagneticSensor', MagneticSensor, MagneticSensorCallback, queue_size=1)

rospy.spin()

rospy.loginfo('writing sensor log to file %s'%sensor_log_file)
with open(sensor_log_file, 'w') as file:
    documents = dump(sensor_log, file, default_flow_style=False)

rospy.loginfo('done')
