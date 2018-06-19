"""loads images to np.arrays

todo: try https://keras.io/preprocessing/image/
todo: improve accuracy (better pic size, more layers, more epochs)

useful resources:
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


def load_images_into_array(path, pic_size=(20, 20)):
    """iterates over a directory and reads all images
    into an ndarray with dimensions (nfiles, width, height, 3)
    assumes all files in the given directory are images"""
    files = os.listdir(path)
    return np.stack([img_to_array(load_img(f"{path}//{file}", target_size=pic_size)) for file in files], axis=0)


PIC_SIZE = (20, 20)

train_cats = load_images_into_array(path="C:\\magisterka_data\\dogscats\\train\\cats", pic_size=PIC_SIZE)
train_doggos = load_images_into_array(path="C:\\magisterka_data\\dogscats\\train\\dogs", pic_size=PIC_SIZE)
valid_cats = load_images_into_array(path="C:\\magisterka_data\\dogscats\\valid\\cats", pic_size=PIC_SIZE)
valid_doggos = load_images_into_array(path="C:\\magisterka_data\\dogscats\\valid\\dogs", pic_size=PIC_SIZE)

print(train_cats.shape)
print(valid_cats.shape)
print(train_doggos.shape)
print(valid_doggos.shape)

# %%
# preprocess arrays to make them proper keras inputs

# concatenate dogs and cats
x_train = np.concatenate((train_cats, train_doggos), axis=0)
x_valid = np.concatenate((valid_cats, valid_doggos), axis=0)

# create label arrays, cat=0, dog=1
train_labels = np.array([0 for i in range(train_cats.shape[0])] + [1 for j in range(train_doggos.shape[0])])
valid_labels = np.array([0 for k in range(valid_cats.shape[0])] + [1 for l in range(valid_doggos.shape[0])])

# shuffle images to improve training process
shuffle_train = np.random.permutation(x_train.shape[0])
shuffle_valid = np.random.permutation(x_valid.shape[0])

x_train = x_train[shuffle_train, :, :, :]
train_labels = train_labels[shuffle_train]

x_valid = x_valid[shuffle_valid, :, :, :]
valid_labels = valid_labels[shuffle_valid]

# make labels more keras-friendly
y_train = to_categorical(train_labels, num_classes=2)
y_valid = to_categorical(valid_labels, num_classes=2)

print(x_train.shape)
print(y_train)
print(x_valid.shape)
print(y_valid)

# %%
# train discriminator

discriminator = Sequential()
discriminator.add(
    Conv2D(32, (3, 3), activation="relu", data_format="channels_last", input_shape=(PIC_SIZE[0], PIC_SIZE[1], 3)))
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

# %%
# calculate loss
score = discriminator.evaluate(x_valid, y_valid)
print(score)

# calculate accuracy
preds = discriminator.predict(x_valid)
print(sum(np.equal(np.round(preds[:, 0]), y_valid[:, 0])) / preds.shape[0])
