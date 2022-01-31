import rospy
import numpy as np

import message_filters
import std_msgs.msg
from sensor_msgs.msg import JointState
from roboy_middleware_msgs.msg import MagneticSensor
from utils import BodyPart, MagneticId

rospy.init_node("write_rosbag")

body_part = "shoulder_right"
joint_names = [body_part + "_axis" + str(i) for i in range(3)]

record = open("./data/training_data/" + body_part + "_data_9.log", "w")
record.write("mx0 my0 mz0 mx1 my1 mz1 mx2 my2 mz3 mx3 my3 mz3 roll pitch yaw\n")

numberOfSamples = 300000
samples = np.zeros((numberOfSamples, 9))
roll = 0
pitch = 0
yaw = 0
sample = 0

tracker_down = False
tracker_down_cnt = 0
last_tracker_down = 0

magnetic_data = None


def tracker_down_callback(data):
    global tracker_down, tracker_down_cnt
    tracker_down = data.data


def tracking_callback(tracking_data):
    if not tracking_data.name[0] == joint_names[0]:
        return

    global sample, magnetic_data

    position = [0, 0, 0]
    for i in range(3):
        idx = tracking_data.name.index(joint_names[i])
        position[i] = tracking_data.position[idx]

    roll = position[0]
    pitch = position[1]
    yaw = position[2]

    if tracker_down:
        rospy.loginfo("Tracker down " + str(roll) + " " + str(pitch) + " " + str(yaw))
        return

    if magnetic_data is None:
        return

    s0_norm = ((magnetic_data.x[0] + magnetic_data.y[0] + magnetic_data.z[0]) ** 2) ** 0.5
    s1_norm = ((magnetic_data.x[1] + magnetic_data.y[1] + magnetic_data.z[1]) ** 2) ** 0.5
    s2_norm = ((magnetic_data.x[2] + magnetic_data.y[2] + magnetic_data.z[2]) ** 2) ** 0.5
    s3_norm = ((magnetic_data.x[3] + magnetic_data.y[3] + magnetic_data.z[3]) ** 2) ** 0.5
    if s0_norm == 0 or s1_norm == 0 or s2_norm == 0 or s3_norm == 0:
        return

    record.write(
        str(magnetic_data.x[0]) + " " + str(magnetic_data.y[0]) + " " + str(magnetic_data.z[0]) + " " +
        str(magnetic_data.x[1]) + " " + str(magnetic_data.y[1]) + " " + str(magnetic_data.z[1]) + " " +
        str(magnetic_data.x[2]) + " " + str(magnetic_data.y[2]) + " " + str(magnetic_data.z[2]) + " " +
        str(magnetic_data.x[3]) + " " + str(magnetic_data.y[3]) + " " + str(magnetic_data.z[3]) + " " +
        str(roll) + " " + str(pitch) + " " + str(yaw) + "\n")

    sample = sample + 1
    rospy.loginfo_throttle(5, "%s: \n Data collection progress: %f%%" % (
        body_part, float(sample) / float(numberOfSamples) * 100.0))


def magnetic_callback(data):
    if not BodyPart[MagneticId(data.id).name] == body_part:
        return

    global magnetic_data
    magnetic_data = data


tracking_sub = rospy.Subscriber('/roboy/pinky/sensing/external_joint_states', JointState, tracking_callback)
magnetic_sub = rospy.Subscriber('/roboy/pinky/middleware/MagneticSensor', MagneticSensor, magnetic_callback)

# ts = message_filters.ApproximateTimeSynchronizer([tracking_sub, magnetic_sub], 10, 0.5)
# ts.registerCallback(data_callback)

tracking_loss_sub = rospy.Subscriber('/tracking_loss', std_msgs.msg.Bool, tracker_down_callback)
rospy.spin()
