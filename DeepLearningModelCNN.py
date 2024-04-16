import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import numpy as np

''' 
large database of handwritten digits that is commonly used for training various image processing systems.
The database is also widely used for training and testing in the field of machine learning.
'''


''' 
Reshaping: The data is reshaped to have a single channel (grayscale) for each image.
Type Conversion: The data type is converted to float32.
Channels Conversion: The grayscale images are converted to RGB format by stacking the same image thrice along the channel axis.
One-hot Encoding: The integer labels representing classes are converted into a one-hot encoded format.
'''

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# reshape the data
X_train = x_train.reshape((x_train.shape[0], 28, 28))
X_test = x_test.reshape((x_test.shape[0], 28, 28))
# change the type to float
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# convert data to 3 channels
''' 
converts the grayscale images in the MNIST dataset to RGB format by 
replicating the single-channel image three times along the channel axis.
'''
X_train = np.stack((X_train,)*3, axis=-1)
X_test = np.stack((X_test,)*3, axis=-1)

''' 
it converts the integer labels representing classes into a one-hot encoded format. 
One-hot encoding is a representation of categorical variables as binary vectors where 
each element in the vector represents a category, and only one element is 'hot' 
'''

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


''' 
This is a common technique used in deep learning for image classification tasks t
o artificially increase the diversity of your training dataset, which can help 
improve the generalization capability of your model and prevent overfitting.
'''
# data augmentation with generator
train_generator = ImageDataGenerator(
    rescale = 1./255,  # normalization of images
    rotation_range = 40, # augmention of images to avoid overfitting
    shear_range = 0.2,
    zoom_range = 0.2,
    fill_mode = 'nearest'
)

val_generator = ImageDataGenerator(rescale = 1./255)

train_iterator = train_generator.flow(X_train, y_train, batch_size=512, shuffle=True)

val_iterator = val_generator.flow(X_test, y_test, batch_size=512, shuffle=False)



plt.imshow(x_train[5].reshape(28,28),cmap='gray')
y_train[5]

plt.imshow(x_train[10].reshape(28,28),cmap='gray')
y_train[10]
print(x_train.shape, x_test.shape)
from tensorflow import keras

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

#normalize the data 
x_train = x_train / 255 
x_test = x_test / 255 


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dense, Dropout



from tensorflow.keras.applications.resnet50 import ResNet50
'''ResNet-50 is a 50-layer convolutional neural network 
(48 convolutional layers, one MaxPool layer, and one average pool layer).'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
# add the pretrained model
model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))

# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.5))  # Add dropout layer with dropout rate of 0.5
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))  # Add dropout layer with dropout rate of 0.5
model.add(Dense(10, activation='softmax'))

# set resnet layers not trainable
model.layers[0].trainable=False
model.summary()

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_iterator, epochs=10, validation_data=val_iterator)


import matplotlib.pyplot as plt

# Evaluate model on training data
train_loss, train_accuracy = model.evaluate(x_train, y_train, verbose=0)
print("Training Loss:", train_loss)
print("Training Accuracy:", train_accuracy)

# Evaluate model on test data
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


