# coding: utf-8
# In[30]:
from keras.models import Sequential
import numpy as np
import pandas
from keras.layers import Dense, Activation, Dropout, LSTM
import wrangle as wr
from matplotlib import pyplot
from keras import callbacks

model = Sequential()
model.add(Dense(units=60, input_dim=9,kernel_initializer='normal', activation='elu'))
# self.model.add(Dropout(0.1))
model.add(Dense(units=60, kernel_initializer='normal', activation='elu'))
# self.model.add(Dropout(0.1))
model.add(Dense(units=4,kernel_initializer='normal', activation='elu'))
model.compile(loss='mse', optimizer='adam')
model.summary()

model_name = "60x60_elu"

# serialize model to JSON
model_json = model.to_json()
with open(model_name+".json", "w") as json_file:
    json_file.write(model_json)

dataset = pandas.read_csv("/home/letrend/workspace/roboy_control/data0.log", delim_whitespace=True, header=1)
dataset = dataset.values[:500000,0:]
np.random.shuffle(dataset)
quaternion_set = np.array(dataset[:,0:4])
sensors_set = np.array(dataset[:,4:13])
sensors_set = wr.mean_zero(pandas.DataFrame(sensors_set)).values
data_in_train = sensors_set[:int(len(sensors_set)*0.7),:]
data_in_test = sensors_set[int(len(sensors_set)*0.7):,:]
data_out_train = quaternion_set[:int(len(sensors_set)*0.7),:]
data_out_test = quaternion_set[int(len(sensors_set)*0.7):,:]

train_X = data_in_train
test_X = data_in_test
train_y = data_out_train
test_y = data_out_test
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
filepath=model_name+"_checkpoint.h5"
callbacks_list = [callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min'), callbacks.TensorBoard(log_dir='./logs', update_freq='epoch')]
# fit network
history = model.fit(train_X, train_y, epochs=100, batch_size=20, validation_data=(test_X, test_y), verbose=2, shuffle=True, callbacks=callbacks_list)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# result = model.predict(train_X)
# print(result)
# serialize weights to HDF5
model.save(model_name+".h5")
print("Saved model to disk")
