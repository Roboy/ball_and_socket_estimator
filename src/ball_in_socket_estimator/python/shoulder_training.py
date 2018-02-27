# coding: utf-8
# In[30]:
from keras.models import Sequential
import numpy as np
import pandas
import rospy
import tensorflow
import tf
import geometry_msgs.msg
from pyquaternion import Quaternion
from keras.layers import Dense, Activation
from roboy_communication_middleware.msg import MagneticSensor
from visualization_msgs.msg import Marker
import sys, select
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
import numpy
global record
record = True
# In[33]:
rospy.init_node('shoulder_magnetics_training')
listener = tf.TransformListener()
rate = rospy.Rate(60.0)


if record is True:
    print("recording training data")
    global numberOfSamples
    numberOfSamples = 1000
    global sample
    global samples
    samples = np.zeros((numberOfSamples,9))
    hl, = plt.plot([], [])
    plt.ion()
    #Set up plot
    figure, ax = plt.subplots()
    lines, = ax.plot([],[], 'o')
            #Autoscale on unknown axis and known lims on the other
    ax.set_autoscaley_on(True)
    ax.set_xlim(0, numberOfSamples)
    ax.grid()
    
    global magneticsSubscriber
    global quaternion_set
    quaternion_set = np.zeros((numberOfSamples,4))
    global sensors_set
    sensors_set = np.zeros((numberOfSamples,9))
    record = open("data0.log","w") 
    record.write("qx qy qz qw mx0 my0 mz0 mx1 my1 mz1 mx2 my2 mz2")
    sample = 0
    def magneticsCallback(data):
        global sample
        global samples
        global numberOfSamples
        global record
        try:
            (trans,rot) = listener.lookupTransform('/world', '/sphereTracker_VO', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return
    
        if sample<numberOfSamples:
            record.write(str(rot[0]) + " " + str(rot[1])+ " " + str(rot[2])+ " " + str(rot[3]) + " " + str(data.x[0])+ " " + str(data.y[0]) + " " + str(data.z[0])+ " " + str(data.x[1])+ " " + str(data.y[1])+ " " + str(data.z[1])+ " " + str(data.x[2])+ " " + str(data.y[2])+ " " + str(data.z[2]) + "\n")
            print(str(rot[0]) + " " + str(rot[1])+ " " + str(rot[2])+ " " + str(rot[3]) + " " + str(data.x[0])+ " " + str(data.y[0]) + " " + str(data.z[0])+ " " + str(data.x[1])+ " " + str(data.y[1])+ " " + str(data.z[1])+ " " + str(data.x[2])+ " " + str(data.y[2])+ " " + str(data.z[2]))
            print(str(sample))
            sensor = np.array([data.x[0], data.y[0], data.z[0], data.x[1], data.y[1], data.z[1], data.x[2], data.y[2], data.z[2]])
            q = np.array([rot[0], rot[1], rot[2], rot[3]])
            sensors_set[sample,:] = sensor.reshape(1,9)
            quaternion_set[sample,:] = q.reshape(1,4)            
            samples[sample,:]= sample
            sample = sample + 1
            
    magneticsSubscriber = rospy.Subscriber("roboy/middleware/MagneticSensor", MagneticSensor, magneticsCallback)
    
    sys.stdin.read(1)
    lines.set_xdata(samples)
    lines.set_ydata(sensors_set)
    #Need both of these in order to rescale
    ax.relim()
    ax.autoscale_view()
    #We need to draw *and* flush
    figure.canvas.draw()
    figure.canvas.flush_events()

class ball_in_socket_estimator:
    model = Sequential()
    graph = tensorflow.get_default_graph() # we need this otherwise the precition does not work ros callback
    if record is False:
        dataset = pandas.read_csv("data0.log", delim_whitespace=True, header=1)
        dataset = dataset.values[1:,:]
    prediction_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=1)
    br = tf.TransformBroadcaster()
    def __init__(self):
        self.model.add(Dense(units=60, input_dim=9,kernel_initializer='normal', activation='relu'))
        self.model.add(Dense(60, kernel_initializer='normal', activation='relu'))
        self.model.add(Dense(units=4,kernel_initializer='normal'))
        global quaternion_set
        global sensors_set
        global record
        if record is False:
            quaternion_set = self.dataset[1:,0:4]
            sensors_set = self.dataset[1:,4:]
        
        self.model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['acc'])

        self.model.fit(sensors_set, quaternion_set, epochs=50)
        self.listener()
    def ros_callback(self, data):
        x_test = np.array([data.x[0], data.y[0], data.z[0], data.x[1], data.y[1], data.z[1], data.x[2], data.y[2], data.z[2]])
        x_test=x_test.reshape((1,9))
        try:
            (trans,rot) = listener.lookupTransform('/world', '/sphereTracker_VO', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return
#        print "input: ",(x_test[0,0],x_test[0,1],x_test[0,2],x_test[0,3],x_test[0,4],x_test[0,5],x_test[0,6],x_test[0,7],x_test[0,8])
        with self.graph.as_default(): # we need this otherwise the precition does not work ros callback
            quat = self.model.predict(x_test)
#            pos = self.model.predict(x_test)
            rospy.loginfo_throttle(1, (quat[0,0],quat[0,1],quat[0,2],quat[0,3]))
            norm = numpy.linalg.norm(quat)
            q = (quat[0,0]/norm,quat[0,1]/norm,quat[0,2]/norm,quat[0,3]/norm)
#            print "predicted: ",(pos[0,0],pos[0,1],pos[0,2])
            self.br.sendTransform(trans,
                     (q[0],q[1],q[2],q[3]),
                     rospy.Time.now(),
                     "shoulderOrientation",
                     "world")
    def listener(self):
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



