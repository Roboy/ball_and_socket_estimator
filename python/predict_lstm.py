import os.path
import gc
import numpy as np
import rospkg
import rospy
import joblib
import std_msgs.msg
import sensor_msgs.msg
import torch
import time
from sensor_msgs.msg import JointState
from roboy_middleware_msgs.msg import MagneticSensor
from libs.deploy_utils import BodyPart, MagneticId
from libs.nn_models import LSTMRegressor
from libs.orientation_utils import compute_rotation_matrix_from_ortho6d, compute_euler_angles_from_rotation_matrices

MAX_ERROR = 0.3
RESET_AFTER_N_REJECTIONS = 5

filter_n = 3
history_n = 5
init_n = 5

rospy.init_node('ball_socket_neural_network')
sensors_scaler_lstm = [None for _ in MagneticId]
model_lstm = [None for _ in MagneticId]
history_lstm = [None for _ in MagneticId]
filter_lstm = [None for _ in MagneticId]
reject_count_lstm = [0 for _ in MagneticId]

lstmPublisher = rospy.Publisher("/roboy/oxford/prediction/external_joint_states", sensor_msgs.msg.JointState, queue_size=1)
joint_targets = [None for _ in MagneticId]

prev_time = time.time()

def load_data(name, id):
    global sensors_scaler_lstm, filter, history
    sensors_scaler_lstm[id], _ = joblib.load(name + '/scaler.pkl')

    filter_lstm[id] = np.zeros((1, 3))

    history_lstm[id] = np.zeros((1, 12))


def target_data_callback(target_data):
    global joint_target

    for i in range(len(model_lstm)):
        body_part = BodyPart[MagneticId(i).name].value

        if f'{body_part}_axis0' in target_data.name:
            idx = target_data.name.index(f'{body_part}_axis0')
            joint_targets[MagneticId(i).value] = np.array(target_data.position[idx:idx+3])


def magentic_data_callback(magnetic_data):
    global filter_lstm, history_lstm, reject_count_lstm, prev_time

    if sensors_scaler_lstm[magnetic_data.id] is None:
        return
    
    if magnetic_data.id != 0:
        return

    x_test = np.array(
        [magnetic_data.x[0], magnetic_data.y[0], magnetic_data.z[0],
         magnetic_data.x[1], magnetic_data.y[1], magnetic_data.z[1],
         magnetic_data.x[2], magnetic_data.y[2], magnetic_data.z[2],
         magnetic_data.x[3], magnetic_data.y[3], magnetic_data.z[3]])
    rospy.loginfo_throttle(1, magnetic_data.id)
    rospy.loginfo_throttle(1, x_test)

    # Normalize input data
    x_test = x_test.reshape((1, len(x_test)))
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
            # msg.name = [BodyPart[MagneticId(magnetic_data.id).name] + "_axis" + str(i) for i in range(3)]
            msg.name = ["axis" + str(i) for i in range(3)]
            msg.position = [output_lstm[0], output_lstm[1], output_lstm[2]]
            msg.velocity = [0, 0, 0]
            msg.effort = [0, 0, 0]
            rospy.loginfo_throttle(1, msg.position)

            lstmPublisher.publish(msg)
        else:
            reject_count_lstm[magnetic_data.id] += 1
            rospy.logwarn("Reject lstm {} with error_{}={}".format(BodyPart[MagneticId(magnetic_data.id).name], error.argmax(), error.max()))

            # Auto reset
            if reject_count_lstm[magnetic_data.id] > RESET_AFTER_N_REJECTIONS:
                reject_count_lstm[magnetic_data.id] = 0
                filter_lstm = [output_lstm for _ in MagneticId]
                
    # Clean up the memory after few seconds
    if time.time() - prev_time > 3:
        gc.collect()
        prev_time = time.time()
        print("Clean up")

if __name__ == '__main__':
    rate = rospy.Rate(300)

    rospack = rospkg.RosPack()
    base_path = rospack.get_path('ball_in_socket_estimator') + '/python/'

    # Search models and its corresponding body_parts.
    for i in range(len(model_lstm)):
        body_part = "shoulder" #BodyPart[MagneticId(i).name].value
        lstm_path = f'{base_path}/outputs_new/{body_part}_lstm_rot6D'

        if os.path.isdir(lstm_path):
            model_lstm[i] = LSTMRegressor.load_from_checkpoint(checkpoint_path=f'{lstm_path}/500epochs.ckpt')
            torch.set_grad_enabled(False)
            model_lstm[i].eval()
            print("Loading LSTM model " + lstm_path + " from disk")
            load_data(lstm_path, i)

    rospy.Subscriber("/roboy/oxford/simulation/joint_targets", JointState, target_data_callback, queue_size=1)
    rospy.Subscriber("/roboy/oxford/middleware/MagneticSensor", MagneticSensor, magentic_data_callback, queue_size=1)

    rospy.spin()
