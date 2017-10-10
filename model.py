import numpy as np
import os
import csv
import cv2
from generator import generator, get_manifest
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Conv2D

# read the data and file manifest from csv log
samples = get_manifest('./data/sample/driving_log.csv')
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
ch, row, col = 3, 160, 320  # Trimmed image format

# keras neural network
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Conv2D(8, 8, 8, subsample=(4,4), border_mode='same',
                activation='relu'))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= \
          len(train_samples), validation_data=validation_generator, \
          nb_val_samples=len(validation_samples), nb_epoch=3)

model.save('model.h5')
