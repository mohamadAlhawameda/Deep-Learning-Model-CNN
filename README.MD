# Introduction

##### In this project, the goal is to develop a deep learning model for the mnsit dataset using a convolutional neural network (CNN). In particular, we will make use of the ResNet50 architecture, a cutting-edge CNN model known for its exceptional results on a range of visual identification tasks.

# About the data set

### Context

##### MNIST is a subset of a larger set available from NIST (it's copied from http://yann.lecun.com/exdb/mnist/)

### Content

##### The MNIST database of handwritten digits has a training set of 60,000 examples and a test set of 10,000 examples.

### Four files are available:

##### train-images-idx3-ubyte.gz: training set images (9912422 bytes)

##### train-labels-idx1-ubyte.gz: training set labels (28881 bytes)

##### t10k-images-idx3-ubyte.gz: test set images (1648877 bytes)

##### t10k-labels-idx1-ubyte.gz: test set labels (4542 bytes)

# DL model pre tranined model

##### ResNet-50 is a 50-layer convolutional neural network (48 convolutional layers, one MaxPool layer, and one average pool layer) followed by additional layers for fine-tuning. include_top=False: Excludes the fully connected layers at the top of the network, which are usually responsible for classification. By excluding them, you can add your classification layers.

##### pooling='avg': Utilizes global average pooling to reduce the spatial dimensions of the output feature map to a vector. weights='imagenet': Initializes the model with weights pre-trained on the ImageNet dataset, which contains a large variety of images across multiple classes. In addition, I have added a dropout layer with a dropout rate of 0.5.

# Conclusion

##### Using the pre-trained model I was able to achieve a good accuracy for the MNSIT image predictions which came about 89 accuracies. the described deep learning model is a powerful architecture for image classification tasks, leveraging the ResNet-50 convolutional neural network as a feature extractor. By using a pre-trained ResNet-50 model, the network benefits from learned features from a large-scale dataset (ImageNet) without the need for extensive training on the target dataset. To mitigate overfitting and enable the model to adapt to the particular classification job, fully linked dense layers with dropout regularisation are included. Before the fully linked layers, the global average pooling layer creates a compact representation by reducing the spatial dimensions of the feature maps
