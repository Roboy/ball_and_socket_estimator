
# coding: utf-8

# In[30]:


from keras.models import Sequential
import numpy as np
import pandas
import rospy
import tensorflow as tf

# In[33]:

from keras.layers import Dense, Activation
from roboy_communication_middleware.msg import MagneticSensor
from visualization_msgs.msg import Marker

class ball_in_socket_estimator:
    model = Sequential()
    graph = tf.get_default_graph() # we need this otherwise the precition does not work ros callback
    dataset1 = pandas.read_csv("/home/roboy/workspace/neural_net_test/data.log", delim_whitespace=True, header=None)
#    dataset2 = pandas.read_csv("/home/roboy/workspace/neural_net_test/data1.log", delim_whitespace=True, header=None)
#    dataset3 = pandas.read_csv("/home/roboy/workspace/neural_net_test/data3.log", delim_whitespace=True, header=None)
    prediction_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=1)
    def __init__(self):
        self.model.add(Dense(units=64, input_dim=9,kernel_initializer='normal', activation='relu'))
        self.model.add(Dense(100, kernel_initializer='normal', activation='relu'))
        self.model.add(Dense(units=4,kernel_initializer='normal'))
        
        quaternion_set1 = self.dataset1.values[1:,1:5]
#        quaternion_set2 = self.dataset2.values[1:,1:5]
#        quaternion_set3 = self.dataset3.values[1:,1:5]
        sensors_set1 = self.dataset1.values[1:,8:]
#        sensors_set2 = self.dataset2.values[1:,8:]
#        sensors_set3 = self.dataset3.values[1:,8:]
        
        self.model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])
        # x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
        self.model.fit(sensors_set1, quaternion_set1, epochs=500)
#        self.model.fit(sensors_set2, quaternion_set2, epochs=30)
#        self.model.fit(sensors_set3, quaternion_set3, epochs=30)
        self.listener()
    def ros_callback(self, data):
        x_test = np.array([data.x[0], data.y[0], data.z[0], data.x[1], data.y[1], data.z[1], data.x[2], data.y[2], data.z[2]])
        x_test=x_test.reshape((1,9))
#        print "input: ",(x_test[0,0],x_test[0,1],x_test[0,2],x_test[0,3],x_test[0,4],x_test[0,5],x_test[0,6],x_test[0,7],x_test[0,8])
        with self.graph.as_default(): # we need this otherwise the precition does not work ros callback
            quat = self.model.predict(x_test)
            print "predicted: ",(quat[0,0],quat[0,1],quat[0,2],quat[0,3])
            msg = Marker()
            msg.header.frame_id = "world"
            msg.header.stamp = rospy.Time.now()
            msg.type = Marker.CYLINDER
            msg.pose.position.x = 0
            msg.pose.position.y = 0
            msg.pose.position.z = 0
            msg.pose.orientation.x = quat[0,0];
            msg.pose.orientation.y = quat[0,1];
            msg.pose.orientation.z = quat[0,2];
            msg.pose.orientation.w = quat[0,3];
            msg.scale.x = 0.1
            msg.scale.y = 0.1
            msg.scale.z = 0.1
            msg.color.a = 1.0
            msg.color.r = 1.0
            msg.color.g = 1.0
            msg.color.b = 1.0
            self.prediction_pub.publish(msg)
    def listener(self):
        rospy.init_node('listener', anonymous=True)
        rospy.Subscriber("roboy/middleware/MagneticSensor", MagneticSensor, self.ros_callback)
        rospy.spin()


# In[34]:
estimator = ball_in_socket_estimator()
# In[34]:
#estimator.listener()
# In[13]:
#
#
#
#model = Sequential()
#from keras.layers import Dense, Activation
#
#model.add(Dense(units=64, input_dim=9,kernel_initializer='normal', activation='relu'))
##model.add(Activation('relu'))
#model.add(Dense(6, kernel_initializer='normal', activation='relu'))
#model.add(Dense(units=4,kernel_initializer='normal'))
##model.add(Activation('softmax'))
#
#
## In[18]:
#
#
#dataset = pandas.read_csv("/home/roboy/workspace/neural_net_test/data0.log", delim_whitespace=True, header=None)
#dataset2 = pandas.read_csv("/home/roboy/workspace/neural_net_test/data1.log", delim_whitespace=True, header=None)
#dataset3 = pandas.read_csv("/home/roboy/workspace/neural_net_test/data3.log", delim_whitespace=True, header=None)
#
#
#quaternion_set = dataset.values[1:,1:5]
#q_test = dataset2.values[1:,1:5]
#sensors_set = dataset.values[1:,8:]
#s_test = dataset2.values[1:,8:]
## In[26]:
#
#
#model.compile(loss='mean_squared_error',
#              optimizer='adam',
#              metrics=['accuracy'])
## x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
#model.fit(sensors_set, quaternion_set, epochs=100)
#
#
## In[9]:
#
#
#x_test=dataset2.values[25,8:]
#x_test=x_test.reshape((1,9))
#x_test
#
#
## In[10]:
#
#
#classes = model.predict(x_test)
#classes

# In[ ]:



