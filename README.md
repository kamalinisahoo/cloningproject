# License-Plate-Character-Recognition-without-using-any-OCR-service-or-Pretrained-Model

The project segments the characters on a typical license plate using image processing techniques and the opencv library. It then uses a state-of-the-art CNN architecture to perform license plate character recognition. The project neither uses any OCR service for character segmentation nor a pre-trained model like YOLO for recognition. Everything has been coded from scratch using basic scientific python library stack.

![vehicle_license_plate_recognition](https://github.com/iamrahul29/License-Plate-Character-Recognition-without-using-any-OCR-service-or-Pretrained-Model/blob/master/images/final_prediction.png)

## Getting Started

The project requires Anaconda distribution and is built using the scientific computing IDE known as Spyder. It uses basic image processing techniques to extract characters from the license plate. It then uses unsupervised learning for label generation. A popular clustering algorithm known as 'K-Means Clustering is implemented.' Finally, the project introduces a state-of-the-art CNN (Convolutional Neural Network) architecture for classifying and recognizing the characters on a vehicle license plate.

The model scores  

```
99.80% on the Training Set
99.02% on the Validation Set
99.30% on the Testing Set
```

![vehicle_license_plate_recognition](https://github.com/iamrahul29/License-Plate-Character-Recognition-without-using-any-OCR-service-or-Pretrained-Model/blob/master/images/train.PNG)

![vehicle_license_plate_recognition](https://github.com/iamrahul29/License-Plate-Character-Recognition-without-using-any-OCR-service-or-Pretrained-Model/blob/master/images/test.PNG)


### Prerequisites

```
Anaconda Distribution
Numpy
Matplotlib
Pandas
Glob
OpenCV
Tensorflow
Keras
Sklearn
```

### Installations

The deep learning and image processing libraries are not prebundled with Anaconda Distribution. They need to be installed separately using either 'pip' or 'conda' package manager

```
pip install opencv-python
pip install tensorflow
pip install glob
pip install imutils
```

### Image Processing Techniques Applied

```
Sharpening using a Kernel
Grayscaling
Binarization
Blurring
Enhancing Contrast
Edge Detection
Corner Detection
```

## Using Contours to create bounding box and extracting the fragments from the license plate

The extracted fragments from the license plates that showcase the segmented characters have been resized in the 40 * 40 dimension. These fragments have been separately stored in the disc for visualization and label generation


![vehicle_license_plate_recognition](https://github.com/iamrahul29/License-Plate-Character-Recognition-without-using-any-OCR-service-or-Pretrained-Model/blob/master/images/segmented_chars.png)

A large proportion of the images in the dataset are untidy and contain a lot of noise. It is difficult to recognize the characters on the license plate even for a human.
The project uses a popular clustering algorithm known as 'K-Means Clustering' for dividing the extracted images into homogeneous clusters. Filtering is used to identify and mark the clusters with appropriate labels. 

## Exploring the Images from the Original Dataset

![vehicle_license_plate_recognition](https://github.com/iamrahul29/License-Plate-Character-Recognition-without-using-any-OCR-service-or-Pretrained-Model/blob/master/images/original.png)

## Applying Various Image Processing Techniques

### Sharpening the Images using a Kernel

![vehicle_license_plate_recognition](https://github.com/iamrahul29/License-Plate-Character-Recognition-without-using-any-OCR-service-or-Pretrained-Model/blob/master/images/sharp_original.png)

### Grayscaling the Images

![vehicle_license_plate_recognition](https://github.com/iamrahul29/License-Plate-Character-Recognition-without-using-any-OCR-service-or-Pretrained-Model/blob/master/images/gray.png)

### Applying the Binarization Technique

![vehicle_license_plate_recognition](https://github.com/iamrahul29/License-Plate-Character-Recognition-without-using-any-OCR-service-or-Pretrained-Model/blob/master/images/binarized_gray_1.png)

![vehicle_license_plate_recognition](https://github.com/iamrahul29/License-Plate-Character-Recognition-without-using-any-OCR-service-or-Pretrained-Model/blob/master/images/binarized_gray_2.png)

### Blurring the  Binarized Images

![vehicle_license_plate_recognition](https://github.com/iamrahul29/License-Plate-Character-Recognition-without-using-any-OCR-service-or-Pretrained-Model/blob/master/images/blur_binarized_gray.png)

### Enhancing Contrast of Images

![vehicle_license_plate_recognition](https://github.com/iamrahul29/License-Plate-Character-Recognition-without-using-any-OCR-service-or-Pretrained-Model/blob/master/images/enhanced_contrast.png)

![vehicle_license_plate_recognition](https://github.com/iamrahul29/License-Plate-Character-Recognition-without-using-any-OCR-service-or-Pretrained-Model/blob/master/images/enhanced_binarized.png)

### Detecting Edges using Canny Edge Detector

![vehicle_license_plate_recognition](https://github.com/iamrahul29/License-Plate-Character-Recognition-without-using-any-OCR-service-or-Pretrained-Model/blob/master/images/edge_images.png)

### Detecting Corners using Harris Corner Detector

![vehicle_license_plate_recognition](https://github.com/iamrahul29/License-Plate-Character-Recognition-without-using-any-OCR-service-or-Pretrained-Model/blob/master/images/corner.png)


## Applying Unsupervised Learning Technique for Label Generation

A large proportion of the images in the dataset are untidy and contain a lot of noise. It is difficult to recognize the characters on the license plate even for a human.
The project uses a popular clustering algorithm known as 'K-Means Clustering' for dividing the extracted images into homogeneous clusters. Filtering is used to identify and mark the clusters with appropriate labels. 

![vehicle_license_plate_recognition](https://github.com/iamrahul29/License-Plate-Character-Recognition-without-using-any-OCR-service-or-Pretrained-Model/blob/master/images/label.png)

![vehicle_license_plate_recognition](https://github.com/iamrahul29/License-Plate-Character-Recognition-without-using-any-OCR-service-or-Pretrained-Model/blob/master/images/label_1.png)

## Built With

* [Anaconda](https://www.anaconda.com/) - The IDE used
* [OpenCV](https://opencv.org/) - Image Processing
* [Python](https://www.python.org/) - Programming Language
* [Tensorflow](https://www.tensorflow.org/) - Deep Learning Framework

Thank you for reading!
Hope you find it helpful.
