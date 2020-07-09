# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import glob
import cv2
import matplotlib
from PIL import Image
from imutils import contours
import os

from sklearn.cluster import KMeans

segmented_characters = [cv2.imread(file) for file in glob.glob('extracted_images/*.png')]
segmented_characters = np.array(segmented_characters)

gray_segmented_characters = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in segmented_characters]
gray_segmented_characters = np.array(gray_segmented_characters)

gray_segmented_characters = gray_segmented_characters.reshape(7684, 1600)

# Using within cluster variation to determine optimal number of clusters
# This code has to be executed multiple times to generate correct clusters
# Filter out all the unique characters and save them in a separate directory

wcv = []

for i in range(1, 40):
    km = KMeans(n_clusters = i)
    km.fit(gray_segmented_characters)
    wcv.append(km.inertia_)

plt.plot(range(1, 40), wcv)
plt.show()

km = KMeans(n_clusters = 31)
y_pred = km.fit_predict(gray_segmented_characters)

test_label = gray_segmented_characters[y_pred == 0]

fig = plt.gcf()
fig.suptitle("Inspecting labels with cluster numbered 1. Looks like '3'. Assign label = 3 to this cluster.", fontsize=14)
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    im = test_label[i]
    im = im.reshape(40, 40)
    plt.imshow(im, "binary")
plt.show()

test_label = test_label.reshape((test_label.shape[0], 40, 40))

# Save the clustered images into separate folders for numbers & characters

for i in range(len(test_label)):
    im = Image.fromarray(test_label[i])
    im.save('numbers/{}.png'.format(i))

for i in range(len(test_label)):
    im = Image.fromarray(test_label[i])
    im.save('chars/{}.png'.format(i))







































