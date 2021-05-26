import numpy as np
import rospkg
import rospy
import joblib
import std_msgs.msg
import sensor_msgs.msg
from visualization_msgs.msg import Marker
from roboy_middleware_msgs.msg import MagneticSensor
from nn_model import NeuralNetworkModel


rospy.init_node('ball_socket_neural_network')
body_part = "shoulder_right"
sensors_scaler = None
model = NeuralNetworkModel()
prediction_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=1)
trackingPublisher = rospy.Publisher("/roboy/pinky/external_joint_states_nn", sensor_msgs.msg.JointState)
targetPublisher = rospy.Publisher("/roboy/pinky/control/joint_targets", sensor_msgs.msg.JointState)


def load_data(name):
    global sensors_scaler
    sensors_scaler = joblib.load(name+'/scaler.pkl')


def sin_cos_to_angle(sin_cos):
    return np.array([
        np.arctan2(sin_cos[:, 0], sin_cos[:, 3]),
        np.arctan2(sin_cos[:, 1], sin_cos[:, 4]),
        np.arctan2(sin_cos[:, 2], sin_cos[:, 5]),
    ]).T


def magentic_data_callback(data):
    x_test = np.array(
        [data.x[0], data.y[0], data.z[0],
         data.x[1], data.y[1], data.z[1],
         data.x[2], data.y[2], data.z[2],
         data.x[3], data.y[3], data.z[3]])

    x_test = x_test.reshape((1, len(x_test)))
    x_test = sensors_scaler.transform(x_test).astype('float32')

    output = model.predict(x_test)
    output = sin_cos_to_angle(output)[0]

    msg = sensor_msgs.msg.JointState()
    msg.header = std_msgs.msg.Header()
    msg.header.stamp = rospy.Time.now()

    msg.name = [body_part + "_axis" + str(i) for i in range(3)]
    msg.position = [output[0], output[1], output[2]]
    msg.velocity = [0, 0, 0]
    msg.effort = [0, 0, 0]

    if not data.id == 1:
        return

    trackingPublisher.publish(msg)


if __name__ == '__main__':
    rate = rospy.Rate(100)

    rospack = rospkg.RosPack()
    base_path = rospack.get_path('ball_in_socket_estimator') + '/python/'
    model_path = './output/'+body_part+'_tanh'

    model.restore_model(model_path + "/best_model")
    print("Loaded model from disk")
    load_data(model_path)

    rospy.Subscriber("/roboy/pinky/middleware/MagneticSensor", MagneticSensor, magentic_data_callback, queue_size=1)
    rospy.spin()

