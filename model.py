
import pickle
import cv2
import tensorflow as tf
import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, BatchNormalization, Convolution2D, Cropping2D, Activation, MaxPooling2D, Dropout

# Read in data
lines = []
# Read in CSV files
print('Extracting log file...')
with open('../driving_log.csv') as csvfile:
    read = csv.reader(csvfile)
    for line in read:
        lines.append(line)
# Parse lines
print('Parsing...')
images = []
measurements = []
for line in lines :
    steering_correction = 0.20
    steering_center = float(line[3])
    steering_left = steering_center + steering_correction
    steering_right = steering_center - steering_correction
    # Lower amount of low steering examples by factor of 10
    # prevents bias toward moving straight
    source_path = line[0]
    # Prepare center image
    filename = source_path.split('/')[-1]
    current_path = '../IMG/' + filename
    # Pass the following images
    #   - All instances of signficant turning
    #   - 5% of all images with little turning (prevents bias)
    if (((abs(steering_center) > 0.1) or ((abs(steering_center) <= 0.1) &                   (np.random.randint(100)<=5)))) :      
        # Append center image
        image = cv2.imread(current_path)
        image_cvt = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image_cvt)
        measurements.append(float(line[3]))
        # Add left image + measurement
        source_path = line[1]
        filename = source_path.split('/')[-1]    
        image = cv2.imread(current_path)
        image_cvt = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image_cvt)
        measurements.append(steering_left)
        # Add right image + measurement
        source_path = line[2]
        filename = source_path.split('/')[-1]    
        image = cv2.imread(current_path)
        image_cvt = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image_cvt)
        measurements.append(steering_right)
    
x_train = np.array(images)
y_train = np.array(measurements)

# Build Model
# Uses NVIDIA model 
print('Building Model..')
model = Sequential()
# Normalize
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
# Crop
model.add(Convolution2D(24,(5,5),subsample=(2,2)))
model.add(Activation('relu'))
model.add(Convolution2D(36,(5,5),subsample=(2,2)))
model.add(Activation('relu'))
model.add(Convolution2D(48,(5,5),subsample=(2,2)))
model.add(Activation('relu'))
model.add(Convolution2D(64,(3,3)))
model.add(Activation('relu'))
model.add(Convolution2D(64,(3,3)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.15))
model.add(Dense(50))
model.add(Dropout(0.15))
model.add(Dense(10))
model.add(Dropout(0.15))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, validation_split = 0.2, nb_epoch = 15, shuffle=True)

model.save('model.h5')