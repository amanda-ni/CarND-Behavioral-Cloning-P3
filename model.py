import numpy as np
import os
import csv
import cv2
from generator import generator, get_manifest
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
import sys
import pickle

# parse arguments
if len(sys.argv)<2:
    driving_log='./data/combo-log.csv'
    print('Driving log (CSV) is '+driving_log)
elif len(sys.argv)==2:
    driving_log=sys.argv[1]
else:
    sys.exit()

# read the data and file manifest from csv log
samples = get_manifest(driving_log)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# keras neural network
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Conv2D(16, 8, 8, subsample=(2,2), border_mode='valid', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(32, 4, 4, subsample=(2,2), border_mode='valid', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(24, 5, 5, border_mode='valid', activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
# model.add(Dense(1, activation='tanh'))
model.add(Dense(1, activation=None))

model.compile(loss='mse', optimizer=Adam(lr=0.0001))
history = model.fit_generator(train_generator, samples_per_epoch= \
                  len(train_samples), validation_data=validation_generator, \
                  nb_val_samples=len(validation_samples), nb_epoch=20)

model.save('model.h5')
pickle.dump(history.history, open('epoch-losses.p', "wb"))
