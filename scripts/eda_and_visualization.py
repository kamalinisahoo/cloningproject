# ML libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# DL libraries

import tensorflow as tf
from tensorflow import keras

# Image processing

import glob
import cv2
import matplotlib
from PIL import Image
from imutils import contours
import os

# Exploring the dataset

fig = plt.gcf()
fig.suptitle("Original Images from the Dataset", fontsize=12)
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(dataset['image'][i])
    plt.xlabel(dataset['label'][i])
plt.tight_layout()
plt.show()

# Sharpening the Images using a Kernel

kernel = np.array([[0, -1, 0],
 [-1, 5,-1],
 [0, -1, 0]])

sharp_images = []

sharp_images = [cv2.filter2D(image, -1, kernel) for image in images]

fig = plt.gcf()
fig.suptitle("Sharpened Original Images using a Kernel", fontsize=12)
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(sharp_images[i])
    plt.xlabel(dataset['label'][i])
plt.tight_layout()
plt.show()

# Grayscaling the Images

gray_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

fig = plt.gcf()
fig.suptitle("Grayscaled Images", fontsize=12)
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(gray_images[i], "gray")
    plt.xlabel(dataset['label'][i])
plt.tight_layout()
plt.show()

# Applying Binarization Technique

max_output_value = 255
neighborhood_size = 99
subtract_from_mean = 10

binarized_images = []

binarized_images = [cv2.adaptiveThreshold(image, max_output_value,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY,
                                        neighborhood_size,
                                        subtract_from_mean) for image in gray_images]

fig = plt.gcf()
fig.suptitle("Binarized Grayscaled Images", fontsize=12)
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(binarized_images[i], "binary")
    plt.xlabel(dataset['label'][i])
plt.tight_layout()
plt.show()

fig = plt.gcf()
fig.suptitle("Binarized Grayscaled Images", fontsize=12)
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(binarized_images[i], "gray")
    plt.xlabel(dataset['label'][i])
plt.tight_layout()
plt.show()

# Blurring the Binarized Grayscaled Images

blurry_images = []

blurry_images = [cv2.blur(image, (5,5)) for image in binarized_images]

fig = plt.gcf()
fig.suptitle("Blurry Binarized Grayscaled Images", fontsize=12)
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(blurry_images[i], "gray")
    plt.xlabel(dataset['label'][i])
plt.tight_layout()
plt.show()

# Enhancing Contrast of the Grayscaled Images

enhanced_images = []

enhanced_images = [cv2.equalizeHist(image) for image in gray_images]

fig = plt.gcf()
fig.suptitle("Enhanced Contrast of Grayscaled Images", fontsize=12)
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(enhanced_images[i], "gray")
    plt.xlabel(dataset['label'][i])
plt.tight_layout()
plt.show()

# Enhancing Binarized Grayscaled Images

max_output_value = 255
neighborhood_size = 99
subtract_from_mean = 10

enhanced_binarized_images = []

enhanced_binarized_images = [cv2.adaptiveThreshold(image, max_output_value,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY,
                                        neighborhood_size,
                                        subtract_from_mean) for image in enhanced_images]

fig = plt.gcf()
fig.suptitle("Enhanced Binarized Grayscaled Images", fontsize=12)
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(enhanced_binarized_images[i], "gray")
    plt.xlabel(dataset['label'][i])
plt.tight_layout()
plt.show()


# Detecting Edges using Canny Edge Detector

image_edges = []

for i in range(len(gray_images)):
    median_intensity = np.median(gray_images[i])
    
    lower_threshold = int(max(0, (1.0 - 0.33) * median_intensity))
    upper_threshold = int(min(255, (1.0 + 0.33) * median_intensity))
    
    image_canny = cv2.Canny(gray_images[i], lower_threshold, upper_threshold)
    image_edges.append(image_canny)

fig = plt.gcf()
fig.suptitle("Detecting Edges using Canny Edge Detector Algorithm", fontsize=12)
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image_edges[i], "gray")
    plt.xlabel(dataset['label'][i])
plt.tight_layout()
plt.show()


# Detecting Corners using Harris Corner Detector

image_corners = []

for i in range(len(images)):

    block_size = 2
    aperture = 29
    free_parameter = 0.04
    
    detector_responses = cv2.cornerHarris(gray_images[i], block_size, aperture, free_parameter)
    
    detector_responses = cv2.dilate(detector_responses, None)
    
    threshold = 0.02
    images[i][detector_responses > threshold * detector_responses.max()] = [255,255,255]
    image_gray = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
    image_corners.append(image_gray)

                             
fig = plt.gcf()
fig.suptitle("Detecting Corners using Harris Corner Detector Algorithm", fontsize=12)
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image_corners[i], "gray")
    plt.xlabel(dataset['label'][i])
plt.tight_layout()
plt.show()





























