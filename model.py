import numpy as np
import os
import csv
import cv2
from generator import generator, get_manifest, LossHistory
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import sys
import pickle

# parse arguments
model_load = None
driving_log='./data/collections_log.csv'

if len(sys.argv)<2:
    print('Driving log (CSV) is '+driving_log)
if len(sys.argv)>=2:
    driving_log=sys.argv[1]
if len(sys.argv)>=3:
    nb_epoch=int(sys.argv[2])
if len(sys.argv)>=4:
    print("Model loading from {}".format(sys.argv[3]))
    model_load=sys.argv[3]
if len(sys.argv)>4:
    sys.exit()

# read the data and file manifest from csv log
samples = get_manifest(driving_log)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# keras neural network
model = Sequential()
model.add(Cropping2D(cropping=((60,20),(0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 127.5) - 0.5))
# model.add(Lambda(lambda x: (x / 127.5) - 0.5, input_shape=(160,320,3)))
model.add(Conv2D(24, 8, 8, border_mode='valid', activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D())
model.add(Conv2D(48, 4, 4, border_mode='valid', activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D())
model.add(Conv2D(64, 5, 5, border_mode='valid', activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D())
model.add(Conv2D(48, 3, 3, border_mode='valid', activation='relu'))
model.add(Conv2D(48, 3, 3, border_mode='valid', activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
# model.add(Dense(1, activation='tanh'))
model.add(Dense(1, activation=None))
model.compile(loss='mse', optimizer=Adam(lr=0.001))

# Callbacks
checkpoint = ModelCheckpoint('model.ckpt.h5', verbose=1, save_best_only=True)
history = LossHistory()
callbacks = [checkpoint, history]

# Number of epochsa
if model_load: 
    model = load_model(model_load)
    print("Model loaded from {}".format(model_load))
if not 'nb_epoch' in locals(): 
    nb_epoch=100
print("Beginning fit to model over {} epochs".format(nb_epoch))
history = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), \
                  callbacks=callbacks, validation_data=validation_generator, \
                  nb_val_samples=len(validation_samples), nb_epoch=nb_epoch)

# Save the final model to "model.h5"
model.save('model.h5')
pickle.dump(history.history, open('epoch-losses.p', "wb"))
