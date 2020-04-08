import numpy as np
from keras.models import model_from_json
import tensorflow
import rospy
from roboy_middleware_msgs.msg import MagneticSensor
import std_msgs.msg, sensor_msgs.msg
import rospkg

rospy.init_node('shoulder_predictor')
normalize_magnetic_strength = True
# # load json and create model
# json_file = open(base_path+network_name+'.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)
# # load weights into new model
# model.load_weights(base_path+model_name+".h5") #_checkpoint
# rospy.loginfo("Loaded model from disk")
# model.summary()

class ball_in_socket_estimator:
    rospack = rospkg.RosPack()
    base_path = rospack.get_path('ball_in_socket_estimator')+'/python/'

    # base_path= '/home/letrend/workspace/roboy3/src/ball_in_socket_estimator/python/'
    network_name = 'testmodel'
    model_name = 'testmodel'
    model_to_publish_name = 'head'
    offset = [0,0,0]
    graph = tensorflow.get_default_graph()
    # base_path= '/home/letrend/workspace/roboy3/src/ball_in_socket_estimator/python/'
    joint_state = rospy.Publisher('/external_joint_states', sensor_msgs.msg.JointState , queue_size=1)
    msg = sensor_msgs.msg.JointState()
    def __init__(self):
        # load json and create model
        json_file = open(self.base_path+self.network_name+'.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights(self.base_path+self.model_name+".h5")
        print("Loaded model from disk")
        self.msg.position = [0,0,0]
        self.listener()

    def magneticsCallback(self, data):
        values = []
        for i in range(0,4):
            val = np.array((data.x[i], data.y[i], data.z[i]))
            if normalize_magnetic_strength:
                val /= np.linalg.norm(val)
            values.append(val[0])
            values.append(val[1])
            values.append(val[2])
        x_test = np.array(values)#np.array([data.x[0], data.y[0], data.z[0], data.x[1], data.y[1], data.z[1], data.x[2], data.y[2], data.z[2], data.x[3], data.y[3], data.z[3]])
        x_test=x_test.reshape((1,12))
        with self.graph.as_default(): # we need this otherwise the precition does not work ros callback
            euler = self.model.predict(x_test)
            #            pos = self.model.predict(x_test)
            rospy.loginfo_throttle(1, (euler[0,0],euler[0,1],euler[0,2]))

            self.msg.header = std_msgs.msg.Header()
            self.msg.header.stamp = rospy.Time.now()
            self.msg.name = [self.model_to_publish_name+'_axis0', self.model_to_publish_name+'_axis1', self.model_to_publish_name+'_axis2']
            self.msg.position = [0.9*self.msg.position[0]+0.1*euler[0,0], 0.9*self.msg.position[1]+0.1*euler[0,1], 0.9*self.msg.position[2]+0.1*euler[0,2]]
            for i in range(len(self.msg.position)):
                self.msg.position[i] += self.offset[i]
            self.msg.velocity = [0,0,0]
            self.msg.effort = [0,0,0]
            self.joint_state.publish(self.msg)

    def listener(self):
        rospy.Subscriber("roboy/middleware/MagneticSensor", MagneticSensor, self.magneticsCallback)
        rospy.spin()

estimator = ball_in_socket_estimator()
