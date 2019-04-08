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
import math
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

import pdb
record = False
train = True

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
    record = open("/home/roboy/workspace/roboy_control/data0.log","w")
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
    trackingSubscriber = None
    roll = 0
    pitch = 0
    yaw = 0
    def __init__(self):
        global train
        if train:
            self.model = Sequential()
            self.model.add(Dense(units=45, input_dim=6,kernel_initializer='normal', activation='relu'))
            self.model.add(Dropout(0.01))
            # self.model.add(Dense(units=600, input_dim=6,kernel_initializer='normal', activation='relu'))
            # self.model.add(Dropout(0.1))
            # self.model.add(Dense(units=100, kernel_initializer='normal', activation='relu'))
            self.model.add(Dense(units=45, kernel_initializer='normal', activation='relu'))
            self.model.add(Dropout(0.01))
#            self.model.add(Dense(units=200, kernel_initializer='normal', activation='relu'))
            # self.model.add(Dense(units=400, kernel_initializer='normal', activation='tanh'))
            # self.model.add(Dropout(0.1))
            self.model.add(Dense(units=3,kernel_initializer='normal'))

            self.model.compile(loss='mean_squared_error',
                          optimizer='adam',
                          metrics=['acc'])
            global quaternion_set
            global sensors_set
            global record
            rospy.loginfo("loading data")
            dataset = pandas.read_csv("/home/roboy/workspace/roboy_control/data0.log", delim_whitespace=True, header=1)

            dataset = dataset.values[1:len(dataset)-1,0:]
            print('%d values'%(len(dataset)))
            dataset = dataset[abs(dataset[:,13])<=0.7,:]
            dataset = dataset[abs(dataset[:,14])<=0.7,:]
            dataset = dataset[abs(dataset[:,15])<=1.5,:]
            # print('%d values after filtering outliers'%(len(dataset)))
            # dataset = dataset[0:200000,:]
            euler_set = np.array(dataset[:,13:16])
            # mean_euler = euler_set.mean(axis=0)
            # std_euler = euler_set.std(axis=0)
            # euler_set = (euler_set - mean_euler) / std_euler
            print('max euler ' + str(np.amax(euler_set)))
            print('min euler ' + str(np.amin(euler_set)))
            # sensors_set = np.array([dataset[:,4],dataset[:,5],dataset[:,6],dataset[:,7],dataset[:,8],dataset[:,9],dataset[:,10],dataset[:,11],dataset[:,12]])
            sensors_set = np.array([dataset[:,4],dataset[:,5],dataset[:,7],dataset[:,8],dataset[:,10],dataset[:,11]])
            sensors_set = np.transpose(sensors_set)

            # mean_sensor = sensors_set.mean(axis=0)
            # std_sensor = sensors_set.std(axis=0)
            # sensors_set = (sensors_set - mean_sensor) / std_sensor
            np.set_printoptions(precision=8)
            # print(mean_sensor)
            # print(std_sensor)
            print(sensors_set[0,:])
            print(euler_set[0,:])
            # sensors_set = wr.mean_zero(pandas.DataFrame(sensors_set)).values

            data_split = 0.5

            sensor_train_set = sensors_set[:int(len(sensors_set)*data_split),:]
            euler_train_set = euler_set[:int(len(sensors_set)*data_split),:]
            sensor_test_set = sensors_set[int(len(sensors_set)*data_split):,:]
            euler_test_set = euler_set[int(len(sensors_set)*data_split):,:]

            data_in_train = sensor_train_set[:int(len(sensor_train_set)*0.7),:]
            data_in_test = sensor_train_set[int(len(sensor_train_set)*0.7):,:]
            data_out_train = euler_train_set[:int(len(sensor_train_set)*0.7),:]
            data_out_test = euler_train_set[int(len(sensor_train_set)*0.7):,:]

            # self.model = Sequential()
            # self.model.add(CuDNNLSTM(units=100, input_shape=(train_X.shape[1], train_X.shape[2])))
            # self.model.add(Dense(train_y.shape[1], activation="relu"))
            # optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.1, amsgrad=False)
            self.model.compile(loss='mse', optimizer='adam')
            # out = self.model.predict(train_X)
            # print(out)

            earlyStopping = EarlyStopping(monitor='val_loss', patience=40, verbose=0, mode='min')
            mcp_save = ModelCheckpoint('model.h5', save_best_only=True, monitor='val_loss', mode='min')

            # fit network
            history = self.model.fit(data_in_train, data_out_train, epochs=500, batch_size=150,
                                     validation_data=(data_in_test, data_out_test), verbose=2, shuffle=True,
                                     callbacks=[earlyStopping, mcp_save])

            # serialize model to JSON
            model_json = self.model.to_json()
            with open("model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            # self.model.save("model.h5")
            print("Saved model to disk")
            euler_predict = self.model.predict(sensor_test_set)
            mse = numpy.linalg.norm(euler_predict-euler_test_set)/len(euler_predict)
            print("mse on test_set %f"%(mse))
            # plot history
            pyplot.plot(history.history['loss'], label='train')
            pyplot.plot(history.history['val_loss'], label='test')
            pyplot.legend()
            pyplot.show()

            # result = self.model.predict(data_in_test)
            # print(result)


        else:
            self.trackingSubscriber = rospy.Subscriber("joint_states_training", sensor_msgs.msg.JointState, self.trackingCallback)

            # load json and create model
            json_file = open('/home/roboy/workspace/roboy_control/src/ball_in_socket_estimator/python/model.json', 'r')

            loaded_model_json = json_file.read()
            json_file.close()
            self.model = model_from_json(loaded_model_json)
            # load weights into new model
            self.model.load_weights("/home/roboy/workspace/roboy_control/src/ball_in_socket_estimator/python/model.h5")

            print("Loaded model from disk")
            self.listener()

        self.listener()
    def ros_callback(self, data):
        x_test = np.array([data.x[0], data.y[0], data.x[1], data.y[1], data.x[2], data.y[2]])
        x_test=x_test.reshape((1,6))
        show_ground_truth = True
        # try:
        #     (trans,rot2) = listener.lookupTransform('/world', '/top_estimate', rospy.Time(0))
        # except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        #     return
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
            error_roll = ((self.roll-euler[0,0])**2)**0.5
            error_pitch = ((self.pitch-euler[0,1])**2)**0.5
            error_yaw = ((self.yaw-euler[0,2])**2)**0.5
            rospy.loginfo_throttle(1, str(error_roll) + " " + str(error_pitch) + " " + str(error_yaw) )
            self.publishErrorCube(error_roll,error_pitch,error_yaw)
            self.publishErrorText(error_roll,error_pitch,error_yaw)
            # self.joint_state.publish(msg)
#             norm = numpy.linalg.norm(quat)
#             q = (quat[0,0]/norm,quat[0,1]/norm,quat[0,2]/norm,quat[0,3]/norm)
# #            print "predicted: ",(pos[0,0],pos[0,1],pos[0,2])
#             self.br.sendTransform(trans,
#                      (q[0],q[1],q[2],q[3]),
#                      rospy.Time.now(),
#                      "top_NN",
#                      "world")
    def trackingCallback(self,data):
        self.roll = data.position[0]
        self.pitch = data.position[1]
        self.yaw = data.position[2]
        rospy.loginfo_throttle(5, "receiving tracking data")
    def publishErrorCube(self,error_roll, error_pitch, error_yaw):
        msg2 = Marker()
        msg2.header = std_msgs.msg.Header()
        msg2.header.stamp = rospy.Time.now()
        msg2.action = msg2.ADD
        msg2.ns = 'prediction error'
        msg2.id = 19348720
        msg2.type = msg2.CUBE
        msg2.scale.x = abs(error_roll)
        msg2.scale.y = abs(error_pitch)
        msg2.scale.z = abs(error_yaw)
        msg2.header.frame_id = 'world'
        msg2.color.a = 0.3
        msg2.color.r = 1
        msg2.pose.orientation.w = 1
        msg2.pose.position.z = 0.3
        self.prediction_pub.publish(msg2)
    def publishErrorText(self,error_roll, error_pitch, error_yaw):
        string = "%.3f %.3f %.3f" % (error_roll*180.0/math.pi, error_pitch*180.0/math.pi, error_yaw*180.0/math.pi)
        marker = Marker(
        type=Marker.TEXT_VIEW_FACING,
        id=0,
        lifetime=rospy.Duration(1.5),
        pose=geometry_msgs.msg.Pose(geometry_msgs.msg.Point(0,0,0.3), geometry_msgs.msg.Quaternion(0, 0, 0, 1)),
        scale=geometry_msgs.msg.Vector3(0.01, 0.01, 0.01),
        header=std_msgs.msg.Header(frame_id='world'),
        color=std_msgs.msg.ColorRGBA(1.0, 1.0, 1.0, 1.0),
        text=string)
        self.prediction_pub.publish(marker)
    def listener(self):
        rospy.Subscriber("roboy/middleware/MagneticSensor", MagneticSensor, self.ros_callback)
        trackingSubscriber = rospy.Subscriber("joint_states_training", sensor_msgs.msg.JointState, self.trackingCallback)
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
