import rospy
import numpy as np

import message_filters
import std_msgs.msg
import argparse
from argparse import RawTextHelpFormatter
from sensor_msgs.msg import JointState
from roboy_middleware_msgs.msg import MagneticSensor
from utils import BodyPart, MagneticId

# Global variables and are defined in main function
body_part = None
joint_names = None
record = None

numberOfSamples = 300000
samples = np.zeros((numberOfSamples, 9))
roll = 0
pitch = 0
yaw = 0
sample = 0

tracker_down = False
tracker_down_cnt = 0
last_tracker_down = 0


def tracker_down_callback(data):
    global tracker_down, tracker_down_cnt
    tracker_down = data.data


def data_callback(tracking_data, magnetic_data):
    global tracker_down, last_tracker_down
    global record, body_part, joint_names

    if not BodyPart[MagneticId(magnetic_data.id).name] == body_part:
        return

    if not tracking_data.name[0] == joint_names[0]:
        return

    global sample

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Write data from rosbag, use python write_data_from_rosbag.py -h for more information',
        formatter_class=RawTextHelpFormatter)
    parser.add_argument('--body_part', type=BodyPart, choices=list(BodyPart),
                        help='Body part: head, shoulder_left, shoulder_right, hand_left, hand_right', required=True)
    parser.add_argument('--output', type=str,
                        help='Output name, the output path will be in ./data/training_data/<body_part>_<output>.log',
                        required=True)
    args = parser.parse_args()

    body_part = args.body_part
    joint_names = [body_part + "_axis" + str(i) for i in range(3)]
    record = open("./data/training_data/" + body_part + "_" + args.output + ".log", "w")
    record.write("mx0 my0 mz0 mx1 my1 mz1 mx2 my2 mz3 mx3 my3 mz3 roll pitch yaw\n")

    rospy.init_node("write_rosbag_" + body_part)

    tracking_sub = message_filters.Subscriber('/roboy/pinky/sensing/external_joint_states', JointState)
    magnetic_sub = message_filters.Subscriber('/roboy/pinky/middleware/MagneticSensor', MagneticSensor)

    ts = message_filters.ApproximateTimeSynchronizer([tracking_sub, magnetic_sub], 10, 0.5)
    ts.registerCallback(data_callback)

    tracking_loss_sub = rospy.Subscriber('/tracking_loss', std_msgs.msg.Bool, tracker_down_callback)
    rospy.spin()
