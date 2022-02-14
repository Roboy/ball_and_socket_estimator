import os.path

import numpy as np
import rospkg
import rospy
import joblib
import std_msgs.msg
import sensor_msgs.msg
import message_filters
import torch
import time
from visualization_msgs.msg import Marker
from sensor_msgs.msg import JointState
from roboy_middleware_msgs.msg import MagneticSensor
# from nn_model import FFNeuralNetworkModel, LSTMNeuralNetworkModel
from libs.deploy_utils import BodyPart, MagneticId
from libs.nn_models import LSTMRegressor
from libs.dvbf_models import DVBF
from libs.orientation_utils import compute_rotation_matrix_from_ortho6d, compute_euler_angles_from_rotation_matrices

MAX_ERROR = 0.3
RESET_AFTER_N_REJECTIONS = 5

filter_n = 3
history_n = 5
init_n = 5

rospy.init_node('ball_socket_neural_network')
sensors_scaler_dvbf = [None for _ in MagneticId]
actions_scaler_dvbf = [None for _ in MagneticId]
sensors_scaler_lstm = [None for _ in MagneticId]
model_dvbf = [None for _ in MagneticId]
dvbf_state = [None for _ in MagneticId]
init_dvbf = [False for _ in MagneticId]
model_lstm = [None for _ in MagneticId]
history_dvbf = [None for _ in MagneticId]
history_lstm = [None for _ in MagneticId]
filter_dvbf = [None for _ in MagneticId]
filter_lstm = [None for _ in MagneticId]
reject_count_dvbf = [0 for _ in MagneticId]
reject_count_lstm = [0 for _ in MagneticId]

# dvbfPublisher = rospy.Publisher("/roboy/pinky/sensing/dvbf_joint_states", sensor_msgs.msg.JointState, queue_size=1)
lstmPublisher = rospy.Publisher("/roboy/pinky/sensing/lstm_joint_states", sensor_msgs.msg.JointState, queue_size=1)
dvbfPublisher = rospy.Publisher("/roboy/pinky/sensing/external_joint_states", sensor_msgs.msg.JointState, queue_size=1)
joint_target = None


def load_data(name, id, dvbf=False):
    global sensors_scaler_dvbf, actions_scaler_dvbf, sensors_scaler_lstm, filter, history
    if dvbf:
        sensors_scaler_dvbf[id], actions_scaler_dvbf[id] = joblib.load(name + '/scaler.pkl')
    else:
        sensors_scaler_lstm[id], _ = joblib.load(name + '/scaler.pkl')

    filter_dvbf[id] = np.zeros((1, 3))
    filter_lstm[id] = np.zeros((1, 3))

    history_lstm[id] = np.zeros((1, 12))
    history_dvbf[id] = np.zeros((1, 4, 3))


def target_data_callback(target_data):
    global joint_target

    # Check shoulder left
    if 'shoulder_left_axis0' in target_data.name:
        idx = target_data.name.index('shoulder_left_axis0')
        joint_target = np.array(target_data.position[idx:idx+3])


def magentic_data_callback(magnetic_data):
    global filter_dvbf, filter_lstm, history_dvbf, history_lstm, init_dvbf, reject_count_dvbf, reject_count_lstm

    if sensors_scaler_lstm[magnetic_data.id] is None:
        return

    x_test = np.array(
        [magnetic_data.x[0], magnetic_data.y[0], magnetic_data.z[0],
         magnetic_data.x[1], magnetic_data.y[1], magnetic_data.z[1],
         magnetic_data.x[2], magnetic_data.y[2], magnetic_data.z[2],
         magnetic_data.x[3], magnetic_data.y[3], magnetic_data.z[3]])

    rospy.loginfo_throttle(1, x_test)

    # Normalize input data
    x_test = x_test.reshape((1, len(x_test)))
    x_dvbf_test = sensors_scaler_dvbf[magnetic_data.id].transform(x_test).astype('float32').reshape(-1, 4, 3)
    x_lstm_test = sensors_scaler_lstm[magnetic_data.id].transform(x_test).astype('float32')

    #################### LSTM #####################
    # Make sure we have a long enough trajectory to do the prediction
    history_lstm[magnetic_data.id] = np.append(history_lstm[magnetic_data.id], x_lstm_test, axis=0)
    if len(history_lstm[magnetic_data.id]) > history_n:
        history_lstm[magnetic_data.id] = np.delete(history_lstm[magnetic_data.id], 0, axis=0)
    
        x_in = torch.tensor(history_lstm[magnetic_data.id][None, :, :], dtype=torch.float32)
        out_set = model_lstm[magnetic_data.id](x_in)[:, -1]
        out_set = compute_rotation_matrix_from_ortho6d(out_set)
        output_lstm = compute_euler_angles_from_rotation_matrices(out_set).detach().numpy()

        if len(filter_lstm[magnetic_data.id]) < filter_n:
            filter_lstm[magnetic_data.id] = np.append(filter_lstm[magnetic_data.id], output_lstm, axis=0)

        error = np.abs(filter_lstm[magnetic_data.id][-1] - output_lstm)

        if error.max() < MAX_ERROR:
            filter_lstm[magnetic_data.id] = np.append(filter_lstm[magnetic_data.id], output_lstm, axis=0)
            filter_lstm[magnetic_data.id] = np.delete(filter_lstm[magnetic_data.id], 0, axis=0)

            # Publish messages
            msg = sensor_msgs.msg.JointState()
            msg.header = std_msgs.msg.Header()
            msg.header.stamp = rospy.Time.now()

            output_lstm = output_lstm[0]
            msg.name = [BodyPart[MagneticId(magnetic_data.id).name] + "_axis" + str(i) for i in range(3)]
            msg.position = [output_lstm[0], output_lstm[1], output_lstm[2]]
            msg.velocity = [0, 0, 0]
            msg.effort = [0, 0, 0]

            lstmPublisher.publish(msg)
        else:
            reject_count_lstm[magnetic_data.id] += 1
            rospy.logwarn("Reject lstm {} with error_{}={}".format(BodyPart[MagneticId(magnetic_data.id).name], error.argmax(), error.max()))

            # Auto reset
            if reject_count_lstm[magnetic_data.id] > RESET_AFTER_N_REJECTIONS:
                reject_count_lstm[magnetic_data.id] = 0
                filter_lstm = [output_lstm for _ in MagneticId]

    ################### DVBF #######################
    if len(history_dvbf[magnetic_data.id]) <= init_n:
        history_dvbf[magnetic_data.id] = np.append(history_dvbf[magnetic_data.id], x_dvbf_test, axis=0)
    else:
        if joint_target is None:
            return 

        if not init_dvbf[magnetic_data.id]:
            x_in = torch.tensor(history_dvbf[magnetic_data.id][None, :, :], dtype=torch.float32)
            dvbf_state[magnetic_data.id], _  = model_dvbf[magnetic_data.id].predict_initial(x_in)
            init_dvbf[magnetic_data.id] = True

            return

        action = actions_scaler_dvbf[magnetic_data.id].transform(joint_target[None]).astype('float32')
        x_in = torch.tensor(x_dvbf_test, dtype=torch.float32)
        u_in = torch.tensor(action, dtype=torch.float32)
        out_set, dvbf_state[magnetic_data.id] = model_dvbf[magnetic_data.id].predict_belief(dvbf_state[magnetic_data.id], u_in, x_in)

        out_set = compute_rotation_matrix_from_ortho6d(out_set)
        output_dvbf = compute_euler_angles_from_rotation_matrices(out_set).detach().numpy()

        if len(filter_dvbf[magnetic_data.id]) < filter_n:
            filter_dvbf[magnetic_data.id] = np.append(filter_dvbf[magnetic_data.id], output_dvbf, axis=0)

        error = np.abs(filter_dvbf[magnetic_data.id][-1] - output_dvbf)

        if error.max() < MAX_ERROR:
            filter_dvbf[magnetic_data.id] = np.append(filter_dvbf[magnetic_data.id], output_dvbf, axis=0)
            filter_dvbf[magnetic_data.id] = np.delete(filter_dvbf[magnetic_data.id], 0, axis=0)

            # Publish messages
            msg = sensor_msgs.msg.JointState()
            msg.header = std_msgs.msg.Header()
            msg.header.stamp = rospy.Time.now()

            output_dvbf = output_dvbf[0]
            msg.name = [BodyPart[MagneticId(magnetic_data.id).name] + "_axis" + str(i) for i in range(3)]
            msg.position = [output_dvbf[0], output_dvbf[1], output_dvbf[2]]
            msg.velocity = [0, 0, 0]
            msg.effort = [0, 0, 0]

            dvbfPublisher.publish(msg)
        else:
            reject_count_dvbf[magnetic_data.id] += 1
            rospy.logwarn("Reject dvbf {} with error_{}={}".format(BodyPart[MagneticId(magnetic_data.id).name], error.argmax(), error.max()))

            # Auto reset
            if reject_count_dvbf[magnetic_data.id] > RESET_AFTER_N_REJECTIONS:
                reject_count_dvbf[magnetic_data.id] = 0
                filter_dvbf = [output_dvbf for _ in MagneticId]


if __name__ == '__main__':
    rate = rospy.Rate(300)

    rospack = rospkg.RosPack()
    base_path = rospack.get_path('ball_in_socket_estimator') + '/python/'

    # Search models and its corresponding body_parts.
    for i in range(len(model_dvbf)):
        body_part = BodyPart[MagneticId(i).name].value

        if body_part not in ["shoulder_left"]:
            continue

        dvbf_path = f'{base_path}/outputs_idp/{body_part}_dvbf_long_trajs_ad_with_target_rot6D'
        lstm_path = f'{base_path}/outputs_idp/{body_part}_lstm_rot6D'

        # Load dvbf
        if os.path.isdir(dvbf_path):
            model_dvbf[i] = DVBF.load_from_checkpoint(checkpoint_path=f'{dvbf_path}/best.ckpt')
            torch.set_grad_enabled(False)
            model_dvbf[i].eval()
            print("Loading DVBF model " + dvbf_path + " from disk")
            load_data(dvbf_path, i, dvbf=True)

        if os.path.isdir(lstm_path):
            model_lstm[i] = LSTMRegressor.load_from_checkpoint(checkpoint_path=f'{lstm_path}/best.ckpt')
            torch.set_grad_enabled(False)
            model_lstm[i].eval()
            print("Loading LSTM model " + lstm_path + " from disk")
            load_data(lstm_path, i)

    # target_sub = message_filters.Subscriber('/roboy/pinky/simulation/joint_targets', JointState)
    # magnetic_sub = message_filters.Subscriber('/roboy/pinky/middleware/MagneticSensor', MagneticSensor)

    # ts = message_filters.ApproximateTimeSynchronizer([target_sub, magnetic_sub], 10, 0.1)
    # ts.registerCallback(data_callback)

    rospy.Subscriber("/roboy/pinky/simulation/joint_targets", JointState, target_data_callback, queue_size=1)
    rospy.Subscriber("/roboy/pinky/middleware/MagneticSensor", MagneticSensor, magentic_data_callback, queue_size=1)

    rospy.spin()
