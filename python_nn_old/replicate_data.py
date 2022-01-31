import os
import rospy
import sensor_msgs.msg
from utils import BodyPart
import argparse
from argparse import RawTextHelpFormatter

rospy.init_node('replicate_data')

outPublisher = rospy.Publisher("/roboy/pinky/simulation/joint_targets", sensor_msgs.msg.JointState)
in_ = None
out_ = None
in_idxs = None
out_idxs = None

def joint_targets_callback(data):
    global in_idxs, out_idxs

    if out_idxs is None:
        in_idxs = [i for i, name in enumerate(data.name) if name.find(in_) != -1]
        out_idxs = [i for i, name in enumerate(data.name) if name.find(out_) != -1]

    position = list(data.position)
    for in_idx, out_idx in zip(in_idxs, out_idxs):
        position[out_idx] = position[in_idx]
        # position[in_idx] = 0

    data.position = tuple(position)
    outPublisher.publish(data)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rqt plot data, use replicate_data.py -h for more information',
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument('--input', type=str,
                        help='left or right', required=True)
    parser.add_argument('--output', type=str,
                        help='left or right', required=True)
    args = parser.parse_args()

    in_ = args.input
    out_ = args.output

    joint_targets_sub = rospy.Subscriber('/roboy/pinky/simulation/in_joint_targets', sensor_msgs.msg.JointState,
                                 joint_targets_callback)

    rospy.spin()
