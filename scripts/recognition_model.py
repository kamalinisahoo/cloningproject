# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import cv2
from PIL import Image
from imutils import contours
import os
import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import LearningRateScheduler,ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Dropout, Flatten
from tensorflow.keras.optimizers import Adam

# Creating empty list for loading the images and labels
images = []
labels = []

for i in range(10):
    im = [cv2.imread(file) for file in glob.glob('numbers/{}/*.png'.format(i))]
    im = np.array(im)
    images.append(im)
    labels.append([i] * len(im))

# Variable name to iterate over folders
    
loop_var = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 
            'k', 'l', 'm', 'n', 'p', 'r', 's', 't', 'v', 'w', 'x', 'z']

# Variable Mapping for Inverse Label Encoding

refined_loop_var = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 
            'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'W', 'X', 'Z']

# Saving the encoding information in the dataframe

loop_map = pd.DataFrame()
loop_map['key'] = np.arange(0, 31)
loop_map['value'] = refined_loop_var

count = 10

for i in loop_var:
    im = [cv2.imread(file) for file in glob.glob('chars/{}/*.png'.format(i))]
    im = np.array(im)
    images.append(im)
    labels.append([count] * len(im))
    count += 1

# Preparing the vector of predictions 'y'

import operator
from functools import reduce
label = reduce(operator.concat, labels)

# Preparing feature matrix 'X'

y = np.array(label)
X = images[0]
 
for i in range(1, 31):
    X = np.concatenate([X, images[i]])


# Saving the preprocessed dataset as numpy arrays in the disc for future usecases.
    
np.save('image_array', X)
np.save('image_label', y)

# Feature Scaling

X = X.astype('float64') / 255.0

# Splitting the dataset into train set, test set and validation set
# The split ratio selected is 80:20

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.2, random_state = 1)

# Defining a learning rate scheduler

reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', 
                                patience=3, 
                                verbose=1, 
                                factor=0.2, 
                                min_lr=1e-6)

# Defining the model architecture

model = keras.models.Sequential([
    keras.layers.Conv2D(32, kernel_size = 3, activation = 'relu', padding = 'same', input_shape = [40, 40, 3]),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, kernel_size = 3, activation = 'relu', padding = 'same'),
    keras.layers.MaxPool2D(),
    keras.layers.Conv2D(32, kernel_size = 3, activation = 'relu', padding = 'same'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, kernel_size = 3, activation = 'relu', padding = 'same'),
    keras.layers.MaxPool2D(),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(256, activation = 'relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(31, activation = 'softmax')])

# Compiling the model

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'nadam', metrics = ['accuracy'])

# Fitting the model on the dataset

history = model.fit(X_train, y_train, epochs = 20, validation_data = (X_valid, y_valid),
                    callbacks = [reduce_lr], batch_size = 32)

# Saving the model for future usecases

model.save('my_model_version1')

# Evaluating the model on the test set

model.evaluate(X_test, y_test)

# Randomly selecting 9 images from the dataset for testing and visualization of results
# Dataset is defined in the data_preparation.py script

test_sample = dataset.sample(9)
test_sample_images = []

for path in test_sample['image_path']:
    test_sample_images.append(cv2.imread(path))

# Saving the randomly selected images in a separate folder for analysis
    
for i in range(len(test_sample_images)):
    im = test_sample_images[i]
    im3 = im.copy()

    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)
    
    os.makedirs('X_test/image_{}'.format(i))

    contours_,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    (contours_, _) = contours.sort_contours(contours_, method="left-to-right")
    samples =  np.empty((0,100))
    responses = []
    keys = [i for i in range(48,58)]
    ROI_number = 0
    for cnt in contours_:
        if cv2.contourArea(cnt)>50:
            [x,y,w,h] = cv2.boundingRect(cnt)
    
            if  h>28:
                cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
                roi = thresh[y:y+h,x:x+w]
                roismall = cv2.resize(roi,(28,28))
                #cv2.imshow('norm',im)
                #responses.append(int(chr(key)))
                print(roi.shape)
                if (roi.shape[1] < 80):
                    
                    roismall = cv2.resize(roi,(40,40))
                    cv2.imwrite('X_test/image_{}/image_{}_ROI_{}.png'.format(i, i, ROI_number), roismall)
                    ROI_number += 1 
                    # sample = roismall.reshape((1,100))
                # samples = np.append(samples,sample,0)
                cv2.waitKey(0)
       
    img = Image.fromarray(im)
    img.save('X_test_box/{}.png'.format(i))

# Reading the randomly selected images from the disc and generating predictions

test_image_pred = []

for i in range(9):
    im = [cv2.imread(file) for file in glob.glob('X_test/image_{}/*.png'.format(i))]
    im = np.array(im)
    test_image_pred.append(im)

# List of predictions
    
pred_list = []    

for i in test_image_pred:
    new = i
    new = new.astype('float64') / 255.0
    if new.shape[0] != 0:
        final_pred = model.predict_classes(new)
        pred_list.append(final_pred)

# List of final predictions
pred_list_final = []

# Performing inverse label encoding. Changing integers to characters

for i in range(len(pred_list)):
    pred_list_final.append(list(loop_map['value'][pred_list[i]]))

# Visualizing the final results. 9 randomly selected images along with their predictions

for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(test_sample_images[i])
    plt.xlabel('Prediction : {}'.format(' '.join(pred_list_final[i])), fontsize = 14)
plt.show()















































