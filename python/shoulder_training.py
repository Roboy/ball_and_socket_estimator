# coding: utf-8
# In[30]:
import logging
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
import paramiko
import pdb

# record = True
# train = False

record = False
train = True


use_sftp = False

if len(sys.argv) < 2:
    print("\nUSAGE: python shoulder_training.py body_part, e.g. \n python shoulder_training.py shoulder_left \n")
    sys.exit()

body_part = sys.argv[1]
joint_names = [body_part+"_axis"+str(i) for i in range(3)]
if  body_part == "head":
    id = 0
elif body_part == "shoulder_left":
    id = 3
elif body_part == "shoulder_right":
    id = 4
else:
    print("unknown FPGA id")
    sys.exit()

# In[33]:
rospy.init_node(body_part+'_magnetics_training_training',anonymous=True)
rospy.loginfo("collecting data for %s"%body_part)
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
    print("recording training data for %s"%body_part)
    global numberOfSamples
    numberOfSamples = 1000000
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
    record = open("/home/letrend/workspace/roboy3/"+body_part+"_data0.log","w")
    record.write("mx0 my0 mz0 mx1 my1 mz1 mx2 my2 mz3 mx3 my3 mz3 roll pitch yaw\n")
    roll = 0
    pitch = 0
    yaw = 0
    sample = 0

    def magneticsCallback(data):
        # if (data.id == id):
        global sample
        global samples
        global numberOfSamples
        global record
        # try:
        #     (trans,rot) = listener.lookupTransform('/world', '/top_estimate', rospy.Time(0))
        #     (trans,rot2) = listener.lookupTransform('/world', '/top', rospy.Time(0))
        # except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        #     return

        s0_norm = ((data.x[0]+data.y[0]+data.z[0])**2)**(0.5)
        s1_norm = ((data.x[1]+data.y[1]+data.z[1])**2)**(0.5)
        s2_norm = ((data.x[2]+data.y[2]+data.z[2])**2)**(0.5)
        s3_norm = ((data.x[3]+data.y[3]+data.z[3])**2)**(0.5)
        if s0_norm==0 or s1_norm==0 or s2_norm==0 or s3_norm==0:
            return

        record.write(str(data.x[0])+ " " + str(data.y[0]) + " " + str(data.z[0])+ " " + str(data.x[1])+ " " + str(data.y[1])+ " " + str(data.z[1])+ " " + str(data.x[2])+ " " + str(data.y[2])+ " " + str(data.z[2])  + " " + str(data.x[3])+ " " + str(data.y[3])+ " " + str(data.z[3])+ " " + str(roll) + " " + str(pitch) + " " + str(yaw) + "\n")

        # record.write(str(rot[0]) + " " + str(rot[1])+ " " + str(rot[2])+ " " + str(rot[3]) + " " + str(data.x[0])+ " " + str(data.y[0]) + " " + str(data.z[0])+ " " + str(data.x[1])+ " " + str(data.y[1])+ " " + str(data.z[1])+ " " + str(data.x[2])+ " " + str(data.y[2])+ " " + str(data.z[2]) + " " + str(roll) + " " + str(pitch) + " " + str(yaw) + " " +  str(rot2[0]) + " " + str(rot2[1])+ " " + str(rot2[2])+ " " + str(rot2[3]) + "\n")
        rospy.loginfo_throttle(5,str(sample) + " " + str(data.x[0])+ " " + str(data.y[0]) + " " + str(data.z[0])+ " " + str(data.x[1])+ " " + str(data.y[1])+ " " + str(data.z[1])+ " " + str(data.x[2])+ " " + str(data.y[2])+ " " + str(data.z[2]) + " " + str(data.x[3])+ " " + str(data.y[3])+ " " + str(data.z[3]) + " " + str(roll) + " " + str(pitch) + " " + str(yaw) + "\n")
        sensor = np.array([data.x[0], data.y[0], data.z[0], data.x[1], data.y[1], data.z[1], data.x[2], data.y[2], data.z[2], data.x[3], data.y[3], data.z[3]])
        # q = np.array([rot[0], rot[1], rot[2], rot[3]])
        # sensors_set[sample,:] = sensor.reshape(1,9)
        # quaternion_set[sample,:] = q.reshape(1,4)
        # samples[sample,:]= sample
        sample = sample + 1
        rospy.loginfo_throttle(5, "%s: \n Data collection progress: %f%%"%(body_part, float(sample)/float(numberOfSamples)*100.0))

    def trackingCallback(data):
        global roll
        global pitch
        global yaw
        position = [0,0,0]
        for i in range(3):
            idx = data.name.index(joint_names[i])
            position[i] = data.position[idx]

        roll = position[0]
        pitch = position[1]
        yaw = position[2]
        rospy.loginfo_throttle(10, "%s: receiving tracking data"%body_part)


    magneticsSubscriber = rospy.Subscriber("roboy/middleware/MagneticSensor", MagneticSensor, magneticsCallback)
    trackingSubscriber = rospy.Subscriber("external_joint_states", sensor_msgs.msg.JointState, trackingCallback)
    rospy.spin()

class ball_in_socket_estimator:
    model = Sequential()
    graph = tensorflow.get_default_graph() # we need this otherwise the precition does not work ros callback
        # pdb.set_trace()
    prediction_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=1)
    br = tf.TransformBroadcaster()
    joint_state = rospy.Publisher('/external_joint_states', sensor_msgs.msg.JointState , queue_size=1)
    trackingSubscriber = None
    roll = 0
    pitch = 0
    yaw = 0
    def __init__(self, body_part):
        global train
        self.body_part = body_part
        self.joint_names = [body_part+"_axis"+str(i) for i in range(3)]
        self.trackingPublisher = rospy.Publisher("/external_joint_states", sensor_msgs.msg.JointState)
        if train:
            self.model = Sequential()
            self.model.add(Dense(units=100, input_dim=12,kernel_initializer='normal', activation='relu'))
            # self.model.add(Dropout(0.01))
            # self.model.add(Dense(units=600, input_dim=6,kernel_initializer='normal', activation='relu'))
            # self.model.add(Dropout(0.1))
            # self.model.add(Dense(units=100, kernel_initializer='normal', activation='relu'))
            self.model.add(Dense(units=100, kernel_initializer='normal', activation='relu'))
            # self.model.add(Dropout(0.01))
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
            if use_sftp:
                client = paramiko.SSHClient()
                client.load_system_host_keys()
                client.connect(hostname='192.168.0.224', username='letrend')
                sftp_client = client.open_sftp()
                remote_file = sftp_client.open('/home/letrend/workspace/roboy3/data0.log')
                dataset = pandas.read_csv(remote_file, delim_whitespace=True, header=1)
            else:
                dataset = pandas.read_csv('/home/letrend/workspace/roboy3/'+self.body_part+'_data0.log', delim_whitespace=True, header=1)


            dataset = dataset.values[1:len(dataset)-1,0:]
            numpy.random.shuffle(dataset)
            print('%d values'%(len(dataset)))
            # dataset = dataset[abs(dataset[:,12])<=0.7,:]
            # dataset = dataset[abs(dataset[:,13])<=0.7,:]
            # dataset = dataset[abs(dataset[:,14])<=1.5,:]
            dataset = dataset[abs(dataset[:,12])!=0.0,:]
            dataset = dataset[abs(dataset[:,13])!=0.0,:]
            dataset = dataset[abs(dataset[:,14])!=0.0,:]
            print('%d values after filtering outliers'%(len(dataset)))
            # dataset = dataset[0:200000,:]
            euler_set = np.array(dataset[:,12:15])
            # mean_euler = euler_set.mean(axis=0)
            # std_euler = euler_set.std(axis=0)
            # euler_set = (euler_set - mean_euler) / std_euler
            print('max euler ' + str(np.amax(euler_set)))
            print('min euler ' + str(np.amin(euler_set)))
            sensors_set = np.array([dataset[:,0],dataset[:,1],dataset[:,2],dataset[:,3],dataset[:,4],dataset[:,5],dataset[:,6],dataset[:,7],dataset[:,8],dataset[:,9],dataset[:,10],dataset[:,11]])
            # sensors_set = np.array([dataset[:,4],dataset[:,5],dataset[:,7],dataset[:,8],dataset[:,10],dataset[:,11]])
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

            data_split = 0.7

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

            earlyStopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2, mode='min')
            mcp_save = ModelCheckpoint(self.body_part+'model.h5', save_best_only=True, monitor='val_loss', mode='min')

            # fit network
            history = self.model.fit(data_in_train, data_out_train, epochs=1000, batch_size=1000,
                                     validation_data=(data_in_test, data_out_test), verbose=2, shuffle=True,
                                     callbacks=[earlyStopping, mcp_save])

            # serialize model to JSON
            model_json = self.model.to_json()
            with open(self.body_part+"model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            # self.model.save("model.h5")
            print("Saved model to disk")
            euler_predict = self.model.predict(sensor_test_set)
            mse = numpy.linalg.norm(euler_predict-euler_test_set)/len(euler_predict)*180.0/math.pi
            print("mse on test_set %f degrees"%(mse))
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
            json_file = open('/home/letrend/workspace/roboy3/src/ball_in_socket_estimator/python/'+self.body_part+'model.json', 'r')

            loaded_model_json = json_file.read()
            json_file.close()
            self.model = model_from_json(loaded_model_json)
            # load weights into new model
            self.model.load_weights("/home/letrend/workspace/roboy3/src/ball_in_socket_estimator/python/"+self.body_part+"model.h5")

            print("Loaded model from disk")
            self.listener()

        # self.listener()
    def magentic_data_callback(self, data):
        # if (self.body_part == "shoulder_right" and data.id == 4) or (self.body_part == "shoulder_left" and data.id == 3):
        x_test = np.array([data.x[0], data.y[0], data.z[0], data.x[1], data.y[1], data.z[1], data.x[2], data.y[2], data.z[2], data.x[3], data.y[3], data.z[3]])
        # x_test = np.array([data.x[0], data.y[0], data.x[1], data.y[1], data.x[2], data.y[2]])
        x_test=x_test.reshape((1,len(x_test)))
        show_ground_truth = True
        # try:
        #     (trans,rot2) = listener.lookupTransform('/world', '/top_estimate', rospy.Time(0))
        # except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        #     return
        with self.graph.as_default(): # we need this otherwise the precition does not work ros callback
            euler = self.model.predict(x_test)
#            pos = self.model.predict(x_test)
            rospy.loginfo_throttle(1, (euler[0,0],euler[0,1],euler[0,2]))
            # msg = sensor_msgs.msg.JointState()
            # msg.header = std_msgs.msg.Header()
            # msg.header.stamp = rospy.Time.now()
            # msg.name = self.joint_names
            # msg.position = [euler[0,0], euler[0,1], euler[0,2]]
            # msg.velocity = [0,0,0]
            # msg.effort = [0,0,0]
            # self.trackingPublisher.publish(msg)
            error_roll = (((self.roll-euler[0,0])*180.0/math.pi)**2)**0.5
            error_pitch = (((self.pitch-euler[0,1])*180.0/math.pi)**2)**0.5
            error_yaw = (((self.yaw-euler[0,2])**2)*180.0/math.pi)**0.5
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
        # if (self.body_part == "shoulder_right" and data.id == 4) or (self.body_part == "shoulder_left" and data.id == 3):
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
        msg2.scale.x = abs(error_roll)/10.0
        msg2.scale.y = abs(error_pitch)/10.0
        msg2.scale.z = abs(error_yaw)/10.0
        msg2.header.frame_id = 'world'
        msg2.color.a = 0.3
        if error_pitch>1 or error_roll>1 or error_yaw >1:
            msg2.color.r = 1
        else:
            msg2.color.g = 1
        msg2.pose.orientation.w = 1
        msg2.pose.position.z = 0.4
        self.prediction_pub.publish(msg2)
    def publishErrorText(self,error_roll, error_pitch, error_yaw):
        string = "%.3f %.3f %.3f" % (error_roll, error_pitch, error_yaw)
        color = std_msgs.msg.ColorRGBA(0,1,0, 1.0)
        if error_pitch>1 or error_roll>1 or error_yaw >1:
            color = std_msgs.msg.ColorRGBA(1.0, 0, 0, 1.0)
        marker = Marker(
        type=Marker.TEXT_VIEW_FACING,
        id=0,
        lifetime=rospy.Duration(1.5),
        pose=geometry_msgs.msg.Pose(geometry_msgs.msg.Point(0,0,0.4), geometry_msgs.msg.Quaternion(0, 0, 0, 1)),
        scale=geometry_msgs.msg.Vector3(0.01, 0.01, 0.01),
        header=std_msgs.msg.Header(frame_id='world'),
        color=color,
        text=string)
        self.prediction_pub.publish(marker)
    def listener(self):
        rospy.Subscriber("roboy/middleware/MagneticSensor", MagneticSensor, self.magentic_data_callback)

        trackingSubscriber = rospy.Subscriber("joint_states_training", sensor_msgs.msg.JointState, self.trackingCallback)
        rospy.spin()


# In[34]:
estimator = ball_in_socket_estimator(body_part)

# In[34]:
estimator.listener()
