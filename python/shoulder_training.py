# coding: utf-8
# In[30]:
from keras.models import Sequential
import numpy as np
import pandas
from pandas import DataFrame
from pandas import concat
import rospy
import tensorflow
import tf
import geometry_msgs.msg
from pyquaternion import Quaternion
from keras.layers import Dense, Activation, Dropout, LSTM, CuDNNLSTM
from keras.models import model_from_json
from roboy_middleware_msgs.msg import MagneticSensor
from visualization_msgs.msg import Marker
import sys, select
from pyquaternion import Quaternion
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
import numpy
import wrangle as wr
from matplotlib import pyplot
global record
global sensors_set
global quaternion_set
import std_msgs, sensor_msgs

import pdb
record = True
train = False

# In[33]:
rospy.init_node('shoulder_magnetics_training', anonymous=True)
listener = tf.TransformListener()
rate = rospy.Rate(60.0)

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

if record is True:
    print("recording training data")
    global numberOfSamples
    numberOfSamples = 500000
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
    global trackingSubscriber
    global quaternion_set
    quaternion_set = np.zeros((numberOfSamples,4))
    global sensors_set
    sensors_set = np.zeros((numberOfSamples,9))
    record = open("/home/letrend/workspace/roboy_control/data0.log","w")
    record.write("qx qy qz qw mx0 my0 mz0 mx1 my1 mz1 mx2 my2 mz2 roll pitch yaw qx_top qy_top qz_top qw_top\n")
    roll = 0
    pitch = 0
    yaw = 0
    sample = 0
    def magneticsCallback(data):
        global sample
        global samples
        global numberOfSamples
        global record
        try:
            (trans,rot) = listener.lookupTransform('/world', '/top_estimate', rospy.Time(0))
            (trans,rot2) = listener.lookupTransform('/world', '/top', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return
    
        record.write(str(rot[0]) + " " + str(rot[1])+ " " + str(rot[2])+ " " + str(rot[3]) + " " + str(data.x[0])+ " " + str(data.y[0]) + " " + str(data.z[0])+ " " + str(data.x[1])+ " " + str(data.y[1])+ " " + str(data.z[1])+ " " + str(data.x[2])+ " " + str(data.y[2])+ " " + str(data.z[2]) + " " + str(roll) + " " + str(pitch) + " " + str(yaw) + " " +  str(rot2[0]) + " " + str(rot2[1])+ " " + str(rot2[2])+ " " + str(rot2[3]) + "\n")
        rospy.loginfo_throttle(5,str(sample) + " " + str(rot[0]) + " " + str(rot[1])+ " " + str(rot[2])+ " " + str(rot[3]) + " " + str(data.x[0])+ " " + str(data.y[0]) + " " + str(data.z[0])+ " " + str(data.x[1])+ " " + str(data.y[1])+ " " + str(data.z[1])+ " " + str(data.x[2])+ " " + str(data.y[2])+ " " + str(data.z[2]) + " " + str(roll) + " " + str(pitch) + " " + str(yaw) + " " +  str(rot2[0]) + " " + str(rot2[1])+ " " + str(rot2[2])+ " " + str(rot2[3]))
        sensor = np.array([data.x[0], data.y[0], data.z[0], data.x[1], data.y[1], data.z[1], data.x[2], data.y[2], data.z[2]])
        q = np.array([rot[0], rot[1], rot[2], rot[3]])
        # sensors_set[sample,:] = sensor.reshape(1,9)
        # quaternion_set[sample,:] = q.reshape(1,4)
        # samples[sample,:]= sample
        sample = sample + 1
    def trackingCallback(data):
        global roll
        global pitch
        global yaw
        roll = data.position[0]
        pitch = data.position[1]
        yaw = data.position[2]
        rospy.loginfo_throttle(5, "receiving tracking data")
            
    magneticsSubscriber = rospy.Subscriber("roboy/middleware/MagneticSensor", MagneticSensor, magneticsCallback)
    trackingSubscriber = rospy.Subscriber("joint_states_training", sensor_msgs.msg.JointState, trackingCallback)

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
        # pdb.set_trace()
    prediction_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=1)
    br = tf.TransformBroadcaster()
    joint_state = rospy.Publisher('/joint_states', sensor_msgs.msg.JointState , queue_size=1)
    def __init__(self):
        global train
        if train:
            self.model = Sequential()
            self.model.add(Dense(units=30, input_dim=9,kernel_initializer='normal', activation='relu'))
            # model.add(Dropout(dropout))
            self.model.add(Dense(units=30, kernel_initializer='normal', activation='tanh'))
            self.model.add(Dense(units=30, kernel_initializer='normal', activation='tanh'))
            # model.add(Dropout(dropout))
            self.model.add(Dense(units=3,kernel_initializer='normal'))

            self.model.compile(loss='mean_squared_error',
                          optimizer='adam',
                          metrics=['acc'])
            global quaternion_set
            global sensors_set
            global record
            rospy.loginfo("loading data")
            dataset = pandas.read_csv("/home/letrend/workspace/roboy_control/data0.log", delim_whitespace=True, header=1)

            dataset = dataset.values[:-1,0:]
            euler_set = np.array(dataset[:,13:16])
            sensors_set = np.array(dataset[:,4:13])
            print(sensors_set[0,:])
            print(euler_set[0,:])
            # sensors_set = wr.mean_zero(pandas.DataFrame(sensors_set)).values
            data_in_train = sensors_set[:int(len(sensors_set)*0.7),:]
            data_in_test = sensors_set[int(len(sensors_set)*0.7):,:]
            data_out_train = euler_set[:int(len(sensors_set)*0.7),:]
            data_out_test = euler_set[int(len(sensors_set)*0.7):,:]

            # self.model = Sequential()
            # self.model.add(CuDNNLSTM(units=100, input_shape=(train_X.shape[1], train_X.shape[2])))
            # self.model.add(Dense(train_y.shape[1], activation="relu"))
            self.model.compile(loss='mse', optimizer='adam')
            # out = self.model.predict(train_X)
            # print(out)

            # fit network
            history = self.model.fit(data_in_train, data_out_train, epochs=50, batch_size=500, validation_data=(data_in_test, data_out_test), verbose=2, shuffle=True)
            # plot history
            pyplot.plot(history.history['loss'], label='train')
            pyplot.plot(history.history['val_loss'], label='test')
            pyplot.legend()
            pyplot.show()

            result = self.model.predict(data_in_test[0,:])
            print(result)

            # serialize model to JSON
            model_json = self.model.to_json()
            with open("model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            self.model.save("model.h5")
            print("Saved model to disk")
        else:
            # load json and create model
            json_file = open('/home/letrend/workspace/roboy_control/src/ball_in_socket_estimator/python/model.json', 'r')

            loaded_model_json = json_file.read()
            json_file.close()
            self.model = model_from_json(loaded_model_json)
            # load weights into new model
            self.model.load_weights("/home/letrend/workspace/roboy_control/src/ball_in_socket_estimator/python/model.h5")

            print("Loaded model from disk")

        self.listener()
    def ros_callback(self, data):
        x_test = np.array([data.x[0], data.y[0], data.z[0], data.x[1], data.y[1], data.z[1], data.x[2], data.y[2], data.z[2]])
        x_test=x_test.reshape((1,9))
        show_ground_truth = True
        try:
            (trans,rot2) = listener.lookupTransform('/world', '/top_estimate', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return
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
            # joint_state.publish(msg)
#             norm = numpy.linalg.norm(quat)
#             q = (quat[0,0]/norm,quat[0,1]/norm,quat[0,2]/norm,quat[0,3]/norm)
# #            print "predicted: ",(pos[0,0],pos[0,1],pos[0,2])
#             self.br.sendTransform(trans,
#                      (q[0],q[1],q[2],q[3]),
#                      rospy.Time.now(),
#                      "top_NN",
#                      "world")

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



