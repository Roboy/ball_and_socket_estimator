import numpy as np
from keras.models import model_from_json
import tensorflow
import rospy
from roboy_middleware_msgs.msg import MagneticSensor
import std_msgs.msg, sensor_msgs.msg

base_path= '/home/roboy/workspace/scooping_ws/src/ball_in_socket_estimator/python/'
network_name = 'shoulder_left'
model_name = 'shoulder_left'
rospy.init_node('3dof predictor', anonymous=True)

# load json and create model
json_file = open(base_path+network_name+'.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(base_path+model_name+".h5") #_checkpoint
rospy.loginfo("Loaded model from disk")
model.summary()

class ball_in_socket_estimator:
    graph = tensorflow.get_default_graph()
    base_path= '/home/roboy/workspace/scooping_ws/src/ball_in_socket_estimator/python/'
    network_name = 'shoulder_left'
    model_name = 'shoulder_left'
    joint_state = rospy.Publisher('/joint_states', sensor_msgs.msg.JointState , queue_size=1)
    def __init__(self):
        # load json and create model
        json_file = open(self.base_path+self.network_name+'.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights(self.base_path+self.model_name+".h5")
        print("Loaded model from disk")
        self.listener()
    def magneticsCallback(self, data):
        x_test = np.array([data.x[0], data.y[0], data.z[0], data.x[1], data.y[1], data.z[1], data.x[2], data.y[2], data.z[2]])
        x_test=x_test.reshape((1,9))
        with self.graph.as_default(): # we need this otherwise the precition does not work ros callback
            euler = self.model.predict(x_test)
            #            pos = self.model.predict(x_test)
            rospy.loginfo_throttle(1, (euler[0,0],euler[0,1],euler[0,2]))
            msg = sensor_msgs.msg.JointState()
            msg.header = std_msgs.msg.Header()
            msg.header.stamp = rospy.Time.now()
            msg.name = ['sphere_axis0', 'sphere_axis1', 'sphere_axis2']
            msg.position = [euler[0,0], euler[0,1], euler[0,2]]
            msg.velocity = [0,0,0]
            msg.effort = [0,0,0]
            self.joint_state.publish(msg)

    def listener(self):
        rospy.Subscriber("roboy/middleware/MagneticSensor", MagneticSensor, self.magneticsCallback)
        rospy.spin()

estimator = ball_in_socket_estimator()
