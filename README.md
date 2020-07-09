# License-Plate-Character-Recognition-without-using-any-OCR-service-or-Pretrained-Model

The project segments the characters on a typical license plate using image processing techniques and the opencv library. It then uses a state-of-the-art CNN architecture to perform license plate character recognition. The project neither uses any OCR service for character segmentation nor a pre-trained model like YOLO for recognition. Everything has been coded from scratch using basic scientific python library stack.

![vehicle_license_plate_recognition](https://github.com/iamrahul29/License-Plate-Character-Recognition-without-using-any-OCR-service-or-Pretrained-Model/blob/master/images/final_prediction.png)

## Getting Started

The project requires Anaconda distribution and is built using the scientific computing IDE known as Spyder. It uses basic image processing techniques to extract characters from the license plate. It then uses unsupervised learning for label generation. A popular clustering algorithm known as 'K-Means Clustering is implemented.' Finally, the project introduces a state-of-the-art CNN (Convolutional Neural Network) architecture for classifying and recognizing the characters on a vehicle license plate.

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

### Installing

The deep learning and image processing libraries are not prebundled with Anaconda Distribution. They need to be installed separately using either 'pip' or 'conda' package manager

```
pip install opencv-python
pip install tensorflow
pip install glob
pip install imutils
```

## Exploring the Images from the Original Dataset

![vehicle_license_plate_recognition](https://github.com/iamrahul29/License-Plate-Character-Recognition-without-using-any-OCR-service-or-Pretrained-Model/blob/master/images/original.png)

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

