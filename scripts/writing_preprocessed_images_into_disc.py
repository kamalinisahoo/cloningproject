# Importing the libraries

import glob
import cv2
import matplotlib
from PIL import Image
from imutils import contours
import os

# Writing Grayscaled Images into the Disc

for i in range(len(gray_images)):
    im = Image.fromarray(gray_images[i])
    im.save('grayscaled/{}.png'.format(i))

# Writing Enhanced Images into the Disc
    
for i in range(len(enhanced_images)):
    im = Image.fromarray(enhanced_images[i])
    im.save('enhanced_images/{}.png'.format(i))

# Writing Enhanced Binarized Images into the Disc
    
for i in range(len(enhanced_binarized_images)):
    im = Image.fromarray(enhanced_binarized_images[i])
    im.save('enhanced_binarized_images/{}.png'.format(i))

# Writing Sharpened Images into the Disc
    
for i in range(len(sharp_images)):
    im = Image.fromarray(sharp_images[i])
    im.save('sharpened/{}.png'.format(i))

# Writing Blurry Images into the Disc
    
for i in range(len(blurry_images)):
    im = Image.fromarray(blurry_images[i])
    im.save('blurry/{}.png'.format(i))

# Writing Edge Marked Images into the Disc
    
for i in range(len(image_edges)):
    im = Image.fromarray(image_edges[i])
    im.save('edge/{}.png'.format(i))

# Writing Corner Marked Images into the Disc
    
for i in range(len(image_corners)):
    im = Image.fromarray(image_corners[i])
    im.save('corner/{}.png'.format(i))
