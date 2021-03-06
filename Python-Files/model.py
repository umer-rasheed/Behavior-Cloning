# LOAD DATA
# Import Modules
import argparse
import cv2
import os
import csv
import base64
import numpy as np
import tensorflow as tf
import pickle
import pandas as pd
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.signal import find_peaks_cwt
import random
from sklearn.utils import shuffle
from collections import deque
from scipy.stats import norm
from tqdm import tqdm
import json
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras import backend as K
import h5py

# LOAD DATA
# Load Training, Test & Validation Data from Pickle File
PickleFilename='Training-Data/Drive.pickle'
with open(PickleFilename,'rb') as PickleFile:
    # Load Data from File
    PickleData=pickle.load(PickleFile)

    # Load Training, Test & Validation Data
    XTrain=PickleData['TrainDataset']
    YTrain=PickleData['TrainLabels']
    XValid=PickleData['ValidDataset']
    YValid=PickleData['ValidLabels']
    XTest=PickleData['TestDataset']
    YTest=PickleData['TestLabels']

    # Delete Pickle Data to Free the Memory
    del PickleData  

print('Training, Test & Validation Data Imported from Pickle File')

# LOAD DATA
# Read Data Characteristics
print("Training Data Feature Size:",XTrain.shape)
print("Train Labels Size:",YTrain.shape)
print("Test Features Size:",XTest.shape)
print("Test Labels Size:",YTest.shape)
print("Valid Features Size:",XValid.shape)
print("Valid Labels Size:",YValid.shape)
InputShape=XTrain.shape[1:]
print("Input Shape:",InputShape)

# PREPROCESS DATA
# Divide the Data by 255 & Subtract -0.5 from it
XTrain=XTrain.astype('float32')
XValid=XValid.astype('float32')
XTest=XTest.astype('float32')
XTrain/=255
XValid/=255
XTest/=255
XTrain-=0.5
XValid-=0.5
XTest-=0.5

# TRAINING MODEL
# Set the Parameters and print out the summary of the model
np.random.seed(1337)  
BatchSize=64
Classes=1 
Epoch=15 

# TRAINING MODEL
# Design the Model
print('Model Design!')

# Set Number of Convolutional Filters
Filter01=16
Filter02=8
Filter03=4
Filter04=2

# Set the Size of Pooling Area for Max Pooling
PoolSize=(2,2)

# Set the Kernel Size
KernelSize=(3,3)

# Initiating the Model
Model=Sequential()

# The First Convolutional Layer Converts 3 Channels into 16 Channels
Model.add(Convolution2D(Filter01,KernelSize[0],KernelSize[1],
						border_mode='valid',
						input_shape=InputShape))
						
# Activation via ELU
Model.add(Activation('elu'))

# The Second Convolutional Layer Converts 16 Channels into 8 Channels
Model.add(Convolution2D(Filter02,KernelSize[0],KernelSize[1]))

# Activation via ELU
Model.add(Activation('elu'))

# The Third Convolutional Layer Converts 8 Channels into 4 Channels
Model.add(Convolution2D(Filter03,KernelSize[0],KernelSize[1]))

# Activatoin via ELU
Model.add(Activation('elu'))

# The Fourth Convolutional Layer Converts 4 Channels into 2 Channels
Model.add(Convolution2D(Filter04,KernelSize[0],KernelSize[1]))

# Activatoin via ELU
Model.add(Activation('elu'))

# Apply Max Pooling for each 2x2 Pixels
Model.add(MaxPooling2D(pool_size=PoolSize))

# Apply Dropout of 25%
Model.add(Dropout(0.25))

# Apply Flattening
Model.add(Flatten())

# Add Dense
Model.add(Dense(16))

# Activatoin via ELU
Model.add(Activation('elu'))

# Add Dense
Model.add(Dense(16))

# Activatoin via ELU
Model.add(Activation('elu'))

# Add Dense
Model.add(Dense(16))

# Activatoin via ELU
Model.add(Activation('elu'))

# Apply Dropout of 50%
Model.add(Dropout(0.5))

# Add Dense
Model.add(Dense(Classes))

# Print Model Summary
Model.summary()

# TRAINING MODEL
# Compile Model using Adam Optimizer and Loss Computed by Mean-Squared Error (MSE)
Model.compile(loss='mean_squared_error',
              optimizer=Adam(),
              metrics=['accuracy'])

# TRAINING MODEL  
# Perform Training
ModelFit=Model.fit(XTrain,YTrain,batch_size=BatchSize,nb_epoch=Epoch,verbose=1,validation_data=(XValid,YValid))
Score=Model.evaluate(XTest,YTest,verbose=0)
print('Test Score:',Score[0])
print('Test Accuracy:',Score[1])

# TRAINING MODEL
# Save the Model as JSON File, Save Weights in H5 File
JSONString=Model.to_json()
with open('Model.json','w') as JSONFile:
    json.dump(JSONString,JSONFile)
    Model.save_weights('./Model.h5')

print("Model Trained and Saved!")	
