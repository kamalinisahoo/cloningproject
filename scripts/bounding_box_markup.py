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

# Importing the saved enhanced binarized images

bi = [cv2.imread(file) for file in glob.glob('enhanced_binarized_images/*.png')]

# Loop to create bounding box and save the segmented characters from the license plate
# The segmented characters from the license plate have been resized into 40 * 40 dimension

for i in range(len(bi)):
    im = bi[i]
    im3 = im.copy()

    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)
        
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
                print(roi.shape)
                if (roi.shape[1] < 80):
                    roismall = cv2.resize(roi,(40,40))
                    cv2.imwrite('extracted_images/image_{}_ROI_{}.png'.format(i, ROI_number), roismall)
                    ROI_number += 1 
                    
                
       
    img = Image.fromarray(im)
    img.save('bounding_box_markup/{}.png'.format(i))


# Visualizing some segmented characters of 40 * 40 dimension from the license plate
    
segmented_characters = [cv2.imread(file) for file in glob.glob('extracted_images/*.png')]

fig = plt.gcf()
fig.suptitle("Segmented characters from the license plate (40 * 40)", fontsize=14)
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(segmented_characters[i])
plt.show()













