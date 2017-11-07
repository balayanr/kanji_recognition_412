import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras import backend as K
import ../kanji_and_radicals
import ../load_data_rad
import numpy as np
import pickle
import os
import sys
import gc

epochs = 100
batch_size = 500
img_rows, img_cols = 127, 128
input_shape = (img_rows, img_cols, 1)
num_classes = 235

# Metrics
def multilabel_accuracy(y_true, y_pred):
    return K.mean(K.equal(keras.metrics.mean_absolute_error(y_true, K.round(y_pred)), 0))

# Load the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='sigmoid'))
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy',multilabel_accuracy])
curr_model = "small_cnn_radical"

# Test for memory issues
x_train = x_val = np.random.random((batch_size, 127, 128 , 1))
y_train = y_val = np.ones((batch_size, num_classes))
hist = model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=1,
        verbose=1,
        validation_data=(x_val, y_val))

# Load images
data, x_train, y_train, x_val, y_val, x_test, y_test = load_data_rad.load()
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Train
hist = model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_val, y_val))

score = model.evaluate(x_test, y_test, verbose=1)
print('Test accuracy:', score)
score = model.evaluate(x_val, y_val, verbose=1)
print('Validation accuracy:', score)

try:
    with open("%s_hist.pickle"%curr_model, "wb") as fd:
        pickle.dump(hist.history, fd)
except:
    print("History not saved")

try:
    model.save_weights("%s_model_weights.h5"%curr_model)
except:
    with open("%s_model_weights.pickle"%curr_model, "wb") as fd:
        pickle.dump(model.get_weights(), fd)
