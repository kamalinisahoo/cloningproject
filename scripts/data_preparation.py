# ML libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2

# Preparing the dataset

dataset = pd.read_csv('dataset.csv', names = ['image_path', 'label'])

dataset['image'] = np.nan

images = []

for path in dataset['image_path']:
    images.append(cv2.imread(path))

dataset['image'] = images

