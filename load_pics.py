"""loads images to np.arrays
todo: experiment with https://keras.io/preprocessing/image/

useful sources:
https://keras.io/layers/convolutional/
https://keras.io/getting-started/sequential-model-guide/
"""

import os

import numpy as np
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.models import Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import to_categorical


def load_images_into_array(path, size=100):
    """iterates over a directory and reads all images
    into an ndarray with dimensions (nfiles, width, height, 3)
    assumes all files in the given directory are images"""
    files = os.listdir(path)
    return np.stack([img_to_array(load_img(f"{path}//{file}", target_size=(20, 20))) for file in files], axis=0)


train_cats = load_images_into_array(path="C:\\magisterka_data\\dogscats\\train\\cats")
train_doggos = load_images_into_array(path="C:\\magisterka_data\\dogscats\\train\\dogs")
valid_cats = load_images_into_array(path="C:\\magisterka_data\\dogscats\\valid\\cats")
valid_doggos = load_images_into_array(path="C:\\magisterka_data\\dogscats\\valid\\dogs")

print(train_cats.shape)
print(valid_cats.shape)
print(train_doggos.shape)
print(valid_doggos.shape)

x_train = np.concatenate((train_cats, train_doggos), axis=0)
train_labels = np.array([0 for i in range(train_cats.shape[0])] + [1 for j in range(train_doggos.shape[0])])

x_valid = np.concatenate((valid_cats, valid_doggos), axis=0)
valid_labels = np.array([0 for k in range(valid_cats.shape[0])] + [1 for l in range(valid_doggos.shape[0])])

print(x_train.shape)
print(train_labels)
print(x_valid.shape)
print(valid_labels)

# ====================================
# train discriminator

y_train = to_categorical(train_labels, num_classes=2)
y_valid = to_categorical(valid_labels, num_classes=2)

discriminator = Sequential()
discriminator.add(Conv2D(32, (3, 3), activation="relu", data_format="channels_last", input_shape=(20, 20, 3)))
discriminator.add(Dropout(0.25))
discriminator.add(Flatten())
discriminator.add(Dense(64, activation='relu'))
discriminator.add(Dropout(0.25))
discriminator.add(Dense(32, activation='relu'))
discriminator.add(Dropout(0.25))
discriminator.add(Dense(2, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
discriminator.compile(loss='categorical_crossentropy', optimizer=sgd)

discriminator.fit(x_train, y_train, batch_size=128, epochs=1)

score = discriminator.evaluate(x_valid, y_valid)

print(score)
